// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "camxiom/init/dlt_pnp.hpp"

#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace camxiom::init
{

namespace
{

/// Minimum number of correspondences for DLT-PnP.
///
/// The 2N x 12 design matrix has rank at most min(2N, 12). To leave a
/// 1-dimensional null space we need rank == 11, hence 2N >= 11, i.e.
/// N >= 6. Below this threshold the pose is not identifiable.
///
/// The planar (9-DOF) path technically only needs 2N >= 8 (N >= 4), but
/// we keep the same minimum for both paths because (a) downstream MS1-4..7
/// already supplies many more points and (b) sharing the threshold avoids
/// a class of "works for non-planar but not planar" surprises near the
/// boundary.
constexpr Eigen::Index kMinPoints = 6;

/// Threshold for declaring the DLT-PnP linear system rank deficient.
///
/// For the 12-DOF path: A is 2N x 12, we expect sigma(11) (the null
/// direction) to be near zero and sigma(10) (the smallest in-range
/// singular value) to be O(1) under Hartley normalisation. For the
/// 9-DOF planar path: A is 2N x 9, we examine sigma(8) (null) and
/// sigma(7) (smallest in-range). The configuration is treated as
/// degenerate when the in-range minimum divided by the maximum drops
/// below this threshold. Mirrors MS1-1's `kDegenerateSingularRatio` and
/// MS1-2's `kIacDegenerateSingularRatio`.
constexpr double kPnpDegenerateSingularRatio = 1e-12;

/// Procrustes scale floor.
///
/// During the orthogonalisation step we divide t_raw_n by the recovered
/// scale (mean of singular values for the full 3D case, mean of the first
/// two for the planar case). If that scale is too small (effectively
/// zero), the raw rotation block is itself near-zero and the SVD recovered
/// no usable pose information. Treated as a numerical pathology.
constexpr double kProcrustesScaleFloor = 1e-14;

/// Planarity detection threshold.
///
/// After Hartley normalisation, the world-point covariance has three
/// non-negative eigenvalues (ascending). When the smallest divided by
/// the largest is below this ratio we deem the world points coplanar and
/// dispatch to the 9-DOF planar DLT path. A ratio of 1e-6 means "smallest
/// spread axis is at least 1000x smaller than largest" — true planar
/// inputs sit near machine zero, while noisy near-planar boards typically
/// sit at 1e-3..1e-2, so the gap is comfortable.
constexpr double kPlanarRatioThreshold = 1e-6;

/// Project a 3x3 matrix onto SO(3) via SVD and recover the scale.
///
/// Given R_raw such that R_raw = s * R_proper for some proper rotation
/// R_proper and scale s > 0, returns (R_proper, s) using
///   SVD: R_raw = U * Sigma * V^T
///   s = mean(Sigma_diag)
///   R_proper = U * diag(1, 1, det(U * V^T)) * V^T
/// The diagonal correction guarantees det(R_proper) = +1 even when the
/// raw rotation has a reflection component from SVD sign ambiguity.
///
/// Returns false if the recovered scale is below kProcrustesScaleFloor.
bool procrustesToSO3(const Eigen::Matrix3d &R_raw, Eigen::Matrix3d &R_proper, double &scale)
{
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d sv = svd.singularValues();
  scale = (sv(0) + sv(1) + sv(2)) / 3.0;
  if (!std::isfinite(scale) || scale < kProcrustesScaleFloor)
  {
    return false;
  }

  const Eigen::Matrix3d &U = svd.matrixU();
  const Eigen::Matrix3d &V = svd.matrixV();
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (U * V.transpose()).determinant();
  R_proper = U * D * V.transpose();
  return true;
}

/// Planar variant of Procrustes: orthogonalise a 3x3 block whose first two
/// columns are noisy estimates of r1, r2 (with the third synthesised from
/// the cross product).
///
/// Steps (from the spec):
///   1. M = [r1_raw, r2_raw, r1_raw x r2_raw].
///   2. SVD(M) = U * Sigma * V^T.
///   3. R_proper = U * diag(1, 1, det(U V^T)) * V^T.
///   4. scale = (Sigma(0) + Sigma(1)) / 2. (Sigma(2) is excluded because it
///      comes from the cross-product column, not directly from data, and
///      is not on the same footing as Sigma(0..1).)
///
/// Returns false if scale < kProcrustesScaleFloor.
bool procrustesPlanar(
  const Eigen::Vector3d &r1_raw, const Eigen::Vector3d &r2_raw, Eigen::Matrix3d &R_proper,
  double &scale
)
{
  Eigen::Matrix3d M;
  M.col(0) = r1_raw;
  M.col(1) = r2_raw;
  M.col(2) = r1_raw.cross(r2_raw);

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d sv = svd.singularValues();
  scale = 0.5 * (sv(0) + sv(1));
  if (!std::isfinite(scale) || scale < kProcrustesScaleFloor)
  {
    return false;
  }

  const Eigen::Matrix3d &U = svd.matrixU();
  const Eigen::Matrix3d &V = svd.matrixV();
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = (U * V.transpose()).determinant();
  R_proper = U * D * V.transpose();
  return true;
}

/// Count the number of correspondences whose camera-frame point lies in
/// front of the camera (positive depth along the bearing).
///
/// Used by both 12-DOF and 9-DOF sign resolution. Operates on the
/// Hartley-normalised world points because the depth comparison is
/// sign-only and dimension-free.
Eigen::Index countPositiveDepth(
  const Eigen::Matrix3d &R_cand, const Eigen::Vector3d &t_cand, const Eigen::Matrix3Xd &world_pts_n,
  const Eigen::Matrix3Xd &rays
)
{
  const Eigen::Index n = world_pts_n.cols();
  Eigen::Index positive = 0;
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const Eigen::Vector3d p_cam = R_cand * world_pts_n.col(i) + t_cand;
    if (p_cam.dot(rays.col(i)) > 0.0)
    {
      ++positive;
    }
  }
  return positive;
}

/// Pick the two rows of the cross-product matrix [ray]_x whose coefficients
/// involve the ray's dominant component. Any two of the three rows are
/// algebraically independent, but the fixed (row0, row1) pair degenerates as
/// rz -> 0 (90-degree incidence, routine for >180-deg-FOV models): row0 ->
/// (0, 0, ry) and row1 -> (0, 0, -rx) become parallel and the per-point
/// constraint collapses from rank 2 towards rank 1. Selecting the pair that
/// contains the largest |component| keeps the two rows well-conditioned for
/// every direction. For the usual rz-dominant rays this returns (0, 1), the
/// historical choice, so nothing changes there.
inline std::pair<int, int> selectCrossProductRows(const double rx, const double ry, const double rz)
{
  const double ax = std::abs(rx);
  const double ay = std::abs(ry);
  const double az = std::abs(rz);
  if (ax >= ay && ax >= az)
  {
    return {1, 2};  // rx appears in rows 1 and 2
  }
  if (ay >= az)
  {
    return {0, 2};  // ry appears in rows 0 and 2
  }
  return {0, 1};  // rz appears in rows 0 and 1
}

/// Shared denormalisation step.
///
/// World normalisation:  world_pts_n = scale_W * (world_pts - c_W).
/// The ray-direction constraint only fixes camera-frame points up to a
/// positive scale, so dividing through by scale_W gives the unnormalised
/// pose with R = R_n and t = t_n / scale_W - R_n * c_W.
void denormalisePose(
  const Eigen::Matrix3d &R_n, const Eigen::Vector3d &t_n, const Eigen::Vector3d &c_W,
  double scale_W, Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out
)
{
  R_out = R_n;
  t_out = t_n / scale_W - R_n * c_W;
}

/// 12-DOF DLT-PnP (non-coplanar world points).
///
/// Builds the 2N x 12 design matrix from the cross-product constraint
/// ray_i x (R * X_i_n + t_n) = 0 with p = [r11 r12 r13 t1 | r21 r22 r23
/// t2 | r31 r32 r33 t3]. Solves via SVD, projects onto SO(3) via
/// Procrustes, resolves the sign ambiguity by majority positive-depth
/// vote, then denormalises.
///
/// All inputs are already Hartley-normalised. The full world-point
/// metadata (c_W, scale_W) is forwarded for the final denormalisation.
StatusCode estimatePoseDLT12DoF(
  const Eigen::Matrix3Xd &world_pts_n, const Eigen::Matrix3Xd &rays, const Eigen::Vector3d &c_W,
  double scale_W, Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out
)
{
  const Eigen::Index n = world_pts_n.cols();

  // Build the 2N x 12 design matrix. With
  //   [a]_x = [[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]],
  // the first two rows of e_k^T [ray]_x (R X + t) = 0 give:
  //   row 0:  -rz * (R_row2 X + t2) + ry * (R_row3 X + t3) = 0
  //   row 1:   rz * (R_row1 X + t1) - rx * (R_row3 X + t3) = 0
  Eigen::MatrixXd A(2 * n, 12);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const double rx = rays(0, i);
    const double ry = rays(1, i);
    const double rz = rays(2, i);
    const double wx = world_pts_n(0, i);
    const double wy = world_pts_n(1, i);
    const double wz = world_pts_n(2, i);

    // Two rows of [ray]_x, picked per point (dominant-component rule) so
    // near-90-degree rays stay well-conditioned. Row k contributes
    // coefficient C[k][j] to parameter block j = (R_row_{j+1}, t_{j+1}),
    // i.e. C[k][j] * (wx, wy, wz, 1).
    const double C[3][3] = {{0.0, -rz, ry}, {rz, 0.0, -rx}, {-ry, rx, 0.0}};
    const std::pair<int, int> ks = selectCrossProductRows(rx, ry, rz);
    for (int r = 0; r < 2; ++r)
    {
      const int k = (r == 0) ? ks.first : ks.second;
      const Eigen::Index row = 2 * i + r;
      for (int j = 0; j < 3; ++j)
      {
        A(row, 4 * j + 0) = C[k][j] * wx;
        A(row, 4 * j + 1) = C[k][j] * wy;
        A(row, 4 * j + 2) = C[k][j] * wz;
        A(row, 4 * j + 3) = C[k][j];
      }
    }
  }

  Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();

  // sigma.size() == min(2N, 12) == 12 (precondition N >= kMinPoints == 6
  // guarantees 2N >= 12), so no defensive guard is needed here.
  const double sigma_largest = sigma(0);
  const double sigma_min_nondegenerate = sigma(10);

  if (!(sigma_largest > 0.0) || !std::isfinite(sigma_largest))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  if (sigma_min_nondegenerate / sigma_largest < kPnpDegenerateSingularRatio)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  Eigen::VectorXd p = svd.matrixV().col(11);

  // Reshape p (row-major) into R_raw_n (3x3) and t_raw_n (3x1).
  auto extractRt =
    [](const Eigen::VectorXd &p_vec, Eigen::Matrix3d &R_raw, Eigen::Vector3d &t_raw) {
      R_raw << p_vec(0), p_vec(1), p_vec(2), p_vec(4), p_vec(5), p_vec(6), p_vec(8), p_vec(9),
        p_vec(10);
      t_raw << p_vec(3), p_vec(7), p_vec(11);
    };

  Eigen::Matrix3d R_raw_n;
  Eigen::Vector3d t_raw_n;
  extractRt(p, R_raw_n, t_raw_n);

  Eigen::Matrix3d R_n;
  double scale = 0.0;
  if (!procrustesToSO3(R_raw_n, R_n, scale))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  Eigen::Vector3d t_n = t_raw_n / scale;

  // Sign resolution by majority positive depth.
  Eigen::Index pos_pos = countPositiveDepth(R_n, t_n, world_pts_n, rays);
  // Strict majority: 2*pos < n means MORE than half have negative depth.
  // Exact tie (pos == n/2 for even n) favours the first candidate.
  if (2 * pos_pos < n)
  {
    Eigen::VectorXd p_neg = -p;
    Eigen::Matrix3d R_raw_n_neg;
    Eigen::Vector3d t_raw_n_neg;
    extractRt(p_neg, R_raw_n_neg, t_raw_n_neg);

    Eigen::Matrix3d R_n_alt;
    double scale_alt = 0.0;
    if (!procrustesToSO3(R_raw_n_neg, R_n_alt, scale_alt))
    {
      return StatusCode::NUMERIC_ERROR;
    }
    const Eigen::Vector3d t_n_alt = t_raw_n_neg / scale_alt;
    const Eigen::Index pos_alt = countPositiveDepth(R_n_alt, t_n_alt, world_pts_n, rays);

    if (pos_alt > pos_pos)
    {
      R_n = R_n_alt;
      t_n = t_n_alt;
      pos_pos = pos_alt;
    }
  }

  if (2 * pos_pos < n)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  denormalisePose(R_n, t_n, c_W, scale_W, R, t);

  if (!R.allFinite() || !t.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  R_out = R;
  t_out = t;
  return StatusCode::OK;
}

/// 9-DOF planar DLT-PnP (coplanar world points).
///
/// The solve runs in the plane's own basis: R_plane (proper rotation whose
/// third column is the unit plane normal) rotates the normalised points so
/// their third coordinate is ~0 *by construction* — the coplanar set may be
/// arbitrarily tilted in the world frame. Hartley normalisation only shifts
/// and scales, so without this rotation a tilted plane (z = a*x + b*y) would
/// silently lose its z-content and yield a wrong pose with status OK.
///
/// With wz = 0 in the plane basis the constraint reduces to
///   ray_i x (r1 * wx_i + r2 * wy_i + t) = 0
/// where r1, r2 are the first two columns of R. The unknown vector is
///   p = [r1; r2; t] (length 9)
///     = [r11, r21, r31, r12, r22, r32, t1, t2, t3].
/// Each correspondence contributes two independent rows (1st and 2nd of
/// [ray]_x). After SVD we recover r1, r2, t up to a common scale; r3 is
/// synthesised by Procrustes (M = [r1, r2, r1 x r2]); the basis change is
/// folded back into the rotation before denormalisation.
StatusCode estimatePoseDLTPlanar(
  const Eigen::Matrix3Xd &world_pts_n, const Eigen::Matrix3Xd &rays, const Eigen::Vector3d &c_W,
  double scale_W, const Eigen::Matrix3d &R_plane, Eigen::Matrix3d &R_out, Eigen::Vector3d &t_out
)
{
  const Eigen::Index n = world_pts_n.cols();

  // Rotate into the plane basis; the third row of world_pts_p is ~0.
  const Eigen::Matrix3Xd world_pts_p = R_plane.transpose() * world_pts_n;

  // Build the 2N x 9 design matrix. Layout of p:
  //   p = [r11 r21 r31 | r12 r22 r32 | t1 t2 t3].
  // Row 0 of [ray]_x (R*X + t)  (with X = (wx, wy, 0)):
  //   -rz * (r21*wx + r22*wy + t2) + ry * (r31*wx + r32*wy + t3) = 0
  //   =>  coefficients:
  //         r11:  0
  //         r21: -rz*wx
  //         r31:  ry*wx
  //         r12:  0
  //         r22: -rz*wy
  //         r32:  ry*wy
  //         t1 :  0
  //         t2 : -rz
  //         t3 :  ry
  // Row 1 of [ray]_x:
  //    rz * (r11*wx + r12*wy + t1) - rx * (r31*wx + r32*wy + t3) = 0
  //   =>  coefficients:
  //         r11:  rz*wx
  //         r21:  0
  //         r31: -rx*wx
  //         r12:  rz*wy
  //         r22:  0
  //         r32: -rx*wy
  //         t1 :  rz
  //         t2 :  0
  //         t3 : -rx
  Eigen::MatrixXd A(2 * n, 9);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const double rx = rays(0, i);
    const double ry = rays(1, i);
    const double rz = rays(2, i);
    const double wx = world_pts_p(0, i);
    const double wy = world_pts_p(1, i);

    // Two rows of [ray]_x, picked per point (dominant-component rule, see
    // selectCrossProductRows) so near-90-degree rays stay well-conditioned.
    // Row k of [ray]_x contributes C[k][j] to r1_j (* wx), r2_j (* wy) and
    // t_j, matching the p = [r1; r2; t] layout above.
    const double C[3][3] = {{0.0, -rz, ry}, {rz, 0.0, -rx}, {-ry, rx, 0.0}};
    const std::pair<int, int> ks = selectCrossProductRows(rx, ry, rz);
    for (int r = 0; r < 2; ++r)
    {
      const int k = (r == 0) ? ks.first : ks.second;
      const Eigen::Index row = 2 * i + r;
      for (int j = 0; j < 3; ++j)
      {
        A(row, j) = C[k][j] * wx;
        A(row, 3 + j) = C[k][j] * wy;
        A(row, 6 + j) = C[k][j];
      }
    }
  }

  Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();

  // sigma.size() == min(2N, 9) == 9 (precondition N >= kMinPoints == 6
  // guarantees 2N >= 12 >= 9), so no defensive guard is needed here.
  const double sigma_largest = sigma(0);
  const double sigma_min_nondegenerate = sigma(7);

  if (!(sigma_largest > 0.0) || !std::isfinite(sigma_largest))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  if (sigma_min_nondegenerate / sigma_largest < kPnpDegenerateSingularRatio)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  Eigen::VectorXd p = svd.matrixV().col(8);

  // Extract r1_raw, r2_raw, t_raw_n from the 9-vector.
  auto extractRt2 =
    [](const Eigen::VectorXd &p_vec, Eigen::Vector3d &r1, Eigen::Vector3d &r2, Eigen::Vector3d &t) {
      r1 << p_vec(0), p_vec(1), p_vec(2);
      r2 << p_vec(3), p_vec(4), p_vec(5);
      t << p_vec(6), p_vec(7), p_vec(8);
    };

  Eigen::Vector3d r1_raw, r2_raw, t_raw_n;
  extractRt2(p, r1_raw, r2_raw, t_raw_n);

  Eigen::Matrix3d R_n;
  double scale = 0.0;
  if (!procrustesPlanar(r1_raw, r2_raw, R_n, scale))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  Eigen::Vector3d t_n = t_raw_n / scale;

  // Sign resolution. Plane-basis points have wz_p ~ 0 so only the first two
  // columns of R_n contribute to camera-frame depth — but the helper
  // works correctly regardless because it multiplies by the full
  // world_pts_p (whose third row is approximately zero).
  Eigen::Index pos_pos = countPositiveDepth(R_n, t_n, world_pts_p, rays);
  // Strict majority: 2*pos < n means MORE than half have negative depth.
  // Exact tie (pos == n/2 for even n) favours the first candidate.
  if (2 * pos_pos < n)
  {
    Eigen::VectorXd p_neg = -p;
    Eigen::Vector3d r1_raw_neg, r2_raw_neg, t_raw_n_neg;
    extractRt2(p_neg, r1_raw_neg, r2_raw_neg, t_raw_n_neg);

    Eigen::Matrix3d R_n_alt;
    double scale_alt = 0.0;
    if (!procrustesPlanar(r1_raw_neg, r2_raw_neg, R_n_alt, scale_alt))
    {
      return StatusCode::NUMERIC_ERROR;
    }
    const Eigen::Vector3d t_n_alt = t_raw_n_neg / scale_alt;
    const Eigen::Index pos_alt = countPositiveDepth(R_n_alt, t_n_alt, world_pts_p, rays);

    if (pos_alt > pos_pos)
    {
      R_n = R_n_alt;
      t_n = t_n_alt;
      pos_pos = pos_alt;
    }
  }

  if (2 * pos_pos < n)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  // The DLT solved p_cam ~ R_n * p_plane + t_n with p_plane = R_plane^T p_n;
  // fold the basis change back so R_n maps normalised-world coordinates.
  R_n = R_n * R_plane.transpose();

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  denormalisePose(R_n, t_n, c_W, scale_W, R, t);

  if (!R.allFinite() || !t.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  R_out = R;
  t_out = t;
  return StatusCode::OK;
}

}  // namespace

StatusCode estimatePoseDLT(
  const camxiom::CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix3Xd> &world_pts,
  const Eigen::Ref<const Eigen::Matrix2Xd> &image_pts, Eigen::Matrix3d &R_out,
  Eigen::Vector3d &t_out
)
{
  // ------------------------------------------------------------------
  // 1. Input validation (no out-param mutation on failure).
  // ------------------------------------------------------------------
  const Eigen::Index n = world_pts.cols();
  if (n < kMinPoints || image_pts.cols() != n)
  {
    return StatusCode::INVALID_INPUT;
  }
  if (!world_pts.allFinite() || !image_pts.allFinite())
  {
    return StatusCode::INVALID_INPUT;
  }

  // ------------------------------------------------------------------
  // 2. Convert each pixel to a unit bearing via the model's pixelToRay64.
  //
  // Stored as columns of a 3 x N matrix. Any non-OK status indicates
  // that the observation is geometrically inconsistent with the model
  // (behind the camera, outside FOV, etc.); we surface that as
  // DEGENERATE_CONFIG rather than silently dropping the point — this is
  // calibration data and a silent skip would mask an upstream bug.
  // ------------------------------------------------------------------
  Eigen::Matrix3Xd rays(3, n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const Pixel2d px{image_pts(0, i), image_pts(1, i)};
    const RayResult64 r = camxiom::pixelToRay64(model, px);
    if (r.status != StatusCode::OK)
    {
      return StatusCode::DEGENERATE_CONFIG;
    }
    const Eigen::Vector3d &d = r.ray.direction;
    const double n_d = d.norm();
    if (!(n_d > 0.0) || !std::isfinite(n_d))
    {
      return StatusCode::DEGENERATE_CONFIG;
    }
    // pixelToRay64 returns a unit-norm direction by contract; the explicit
    // re-normalisation is a defensive no-op against any future drift in
    // that contract.
    rays.col(i) = d / n_d;
  }

  // ------------------------------------------------------------------
  // 3. Hartley normalisation on world points (3D extension of D24).
  //
  //    c_W      = centroid
  //    shifted  = world_pts - c_W (column-wise)
  //    mean_d   = mean column norm of shifted
  //    scale_W  = sqrt(3) / mean_d   (3D analogue: target mean dist = sqrt(3))
  //
  //    world_pts_n = scale_W * shifted
  // ------------------------------------------------------------------
  const Eigen::Vector3d c_W = world_pts.rowwise().mean();
  Eigen::Matrix3Xd shifted(3, n);
  shifted = world_pts.colwise() - c_W;

  double mean_dist = 0.0;
  for (Eigen::Index i = 0; i < n; ++i)
  {
    mean_dist += shifted.col(i).norm();
  }
  mean_dist /= static_cast<double>(n);

  if (!(mean_dist > 0.0) || !std::isfinite(mean_dist))
  {
    // All world points coincide.
    return StatusCode::DEGENERATE_CONFIG;
  }

  const double scale_W = std::sqrt(3.0) / mean_dist;
  if (!std::isfinite(scale_W) || !(scale_W > 0.0))
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  const Eigen::Matrix3Xd world_pts_n = scale_W * shifted;

  // ------------------------------------------------------------------
  // 4. Planarity detection on the (already-shifted) world points.
  //
  // We analyse the eigenvalues of the 3x3 covariance matrix. When the
  // smallest eigenvalue divided by the largest falls below
  // kPlanarRatioThreshold the points lie (numerically) on a plane and
  // we dispatch to the 9-DOF planar path; otherwise we use the 12-DOF
  // path. Operating on `shifted` (or equivalently `world_pts_n` after a
  // global scale) gives an axis-spread metric that is invariant under
  // the Hartley scale.
  // ------------------------------------------------------------------
  const Eigen::Matrix3d cov = (shifted * shifted.transpose()) / static_cast<double>(n);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig_cov(cov);
  if (eig_cov.info() != Eigen::Success)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  const Eigen::Vector3d evals = eig_cov.eigenvalues();  // ascending.
  const bool is_planar = (evals(2) > 0.0) && (evals(0) / evals(2) < kPlanarRatioThreshold);

  // ------------------------------------------------------------------
  // 5. Dispatch.
  // ------------------------------------------------------------------
  if (is_planar)
  {
    // The 9-DOF path solves in the plane's own basis. Planarity detection is
    // rotation-invariant (covariance eigenvalues), so the coplanar set may be
    // arbitrarily tilted in the world frame (e.g. board corners expressed in
    // vehicle coordinates); dropping the raw z coordinate there would
    // silently return a wrong pose. Eigenvalues come out ascending, so
    // column 0 of the eigenvector matrix is the plane normal.
    Eigen::Matrix3d R_plane;
    R_plane.col(0) = eig_cov.eigenvectors().col(2);
    R_plane.col(1) = eig_cov.eigenvectors().col(1);
    R_plane.col(2) = eig_cov.eigenvectors().col(0);
    // Procrustes later forces det(R) = +1, so the basis must be a proper
    // rotation, not a reflection.
    if (R_plane.determinant() < 0.0)
    {
      R_plane.col(2) = -R_plane.col(2);
    }
    return estimatePoseDLTPlanar(world_pts_n, rays, c_W, scale_W, R_plane, R_out, t_out);
  }
  return estimatePoseDLT12DoF(world_pts_n, rays, c_W, scale_W, R_out, t_out);
}

StatusCode estimatePoseRefined(
  const camxiom::CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xd> &world_pts,
  const Eigen::Ref<const Eigen::Matrix2Xd> &image_pts, Eigen::Matrix3d &R_out,
  Eigen::Vector3d &t_out
)
{
  // ------------------------------------------------------------------
  // 1. Linear initialisation (model-agnostic), with the same distortion-off
  //    retry that calibrate() uses: a board in the far image periphery may
  //    have pixels past the forward distortion's fold-over radius, where the
  //    full-model bearing lift fails but the base projection still inverts.
  //    The refinement below always uses the full model.
  // ------------------------------------------------------------------
  const camxiom::CameraModel64 m64 = camxiom::toCameraModel64(model);
  Eigen::Matrix3d R_init;
  Eigen::Vector3d t_init;
  StatusCode dlt = estimatePoseDLT(m64, world_pts, image_pts, R_init, t_init);
  if (dlt != StatusCode::OK)
  {
    camxiom::CameraModel model_nodist = model;
    model_nodist.distortion = camxiom::DistortionModel{};
    const camxiom::CameraModel64 m64_nodist = camxiom::toCameraModel64(model_nodist);
    dlt = estimatePoseDLT(m64_nodist, world_pts, image_pts, R_init, t_init);
  }
  if (dlt != StatusCode::OK)
  {
    // The linear initialisation is a prerequisite. Outputs untouched
    // (estimatePoseDLT leaves them unmodified on a non-OK return).
    return dlt;
  }

  // ------------------------------------------------------------------
  // 2. Build the single-view solver inputs from the DLT seed.
  // ------------------------------------------------------------------
  const Eigen::Index n = world_pts.cols();
  camxiom::optimizer::ObjectPoints obj;
  camxiom::optimizer::ImagePoints img;
  obj.reserve(static_cast<std::size_t>(n));
  img.reserve(static_cast<std::size_t>(n));
  for (Eigen::Index j = 0; j < n; ++j)
  {
    obj.emplace_back(world_pts.col(j));
    img.emplace_back(image_pts.col(j));
  }

  camxiom::optimizer::PnpInitialGuess guess;
  guess.camera_model = model;
  const Eigen::AngleAxisd aa_init(R_init);
  guess.rvecs.push_back(aa_init.axis() * aa_init.angle());
  guess.tvecs.push_back(t_init);

  // ------------------------------------------------------------------
  // 3. Refine the pose only: lock intrinsics, distortion and projection
  //    parameters so just the 6-DOF extrinsics move. This routes to the
  //    solver's direct per-view Gauss-Newton path (no Ceres problem set-up).
  // ------------------------------------------------------------------
  const camxiom::optimizer::PnpFlag pose_only = camxiom::optimizer::PnpFlag::FIX_INTRINSICS |
                                                camxiom::optimizer::PnpFlag::FIX_DISTORTION |
                                                camxiom::optimizer::PnpFlag::FIX_PROJECTION_PARAMS;

  camxiom::optimizer::PnpSolver solver;
  camxiom::optimizer::PnpResult result;
  solver.solve(obj, img, guess, result, camxiom::optimizer::PnpSolverOptions(), pose_only);

  // ------------------------------------------------------------------
  // 4. Use the refined pose when it is finite; otherwise fall back to the DLT
  //    seed (already a valid pose). Either way R_out / t_out hold a pose.
  // ------------------------------------------------------------------
  if (result.success && result.rvecs.size() == 1U && result.tvecs.size() == 1U)
  {
    const Eigen::Vector3d &rvec = result.rvecs[0];
    const Eigen::Vector3d &tvec = result.tvecs[0];
    const double angle = rvec.norm();
    const Eigen::Matrix3d R_ref = (angle > 0.0)
                                    ? Eigen::Matrix3d(Eigen::AngleAxisd(angle, rvec / angle))
                                    : Eigen::Matrix3d::Identity();
    if (R_ref.allFinite() && tvec.allFinite())
    {
      R_out = R_ref;
      t_out = tvec;
      return StatusCode::OK;
    }
  }

  R_out = R_init;
  t_out = t_init;
  return StatusCode::OK;
}

}  // namespace camxiom::init
