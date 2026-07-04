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

#include "camxiom/init/homography.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <cmath>

namespace camxiom::init
{

namespace
{

/// Threshold for declaring the DLT system rank deficient.
///
/// A is a 2N x 9 matrix; we expect its smallest singular value sigma_9
/// to be near zero and the second-smallest sigma_8 to be significantly
/// larger. The ratio sigma_9 / sigma_8 quantifies the gap. If sigma_8
/// is itself negligible (no gap), the homography is not identifiable
/// from these correspondences.
///
/// We pick a conservative cutoff: when sigma_8 / sigma_1 < 1e-12 the
/// problem is treated as degenerate. This is robust to scale because
/// inputs are Hartley-normalised (mean distance from origin = sqrt(2),
/// so the well-conditioned singular values are of order O(1)). The
/// test writer is free to tune this further once empirical evidence
/// is available — the threshold is the only knob.
constexpr double kDegenerateSingularRatio = 1e-12;

/// Hartley isotropic normalisation.
///
/// Builds a 3x3 similarity T such that, in homogeneous coordinates,
/// the transformed points have centroid at the origin and a mean
/// distance from the origin equal to sqrt(2).
///
/// On return:
///   * `T` is the similarity transform (translate, then uniform scale).
///   * `pts_norm` is the normalised 2 x N matrix.
///
/// Returns false if the points are coincident (mean distance == 0),
/// which is itself a degenerate configuration.
bool hartleyNormalise(
  const Eigen::Ref<const Eigen::Matrix2Xd> &pts, Eigen::Matrix2Xd &pts_norm, Eigen::Matrix3d &T
)
{
  const Eigen::Index n = pts.cols();
  const Eigen::Vector2d centroid = pts.rowwise().mean();

  Eigen::Matrix2Xd centred(2, n);
  centred = pts.colwise() - centroid;

  // Mean Euclidean distance from origin (post-centring).
  double mean_dist = 0.0;
  for (Eigen::Index i = 0; i < n; ++i)
  {
    mean_dist += centred.col(i).norm();
  }
  mean_dist /= static_cast<double>(n);

  if (!(mean_dist > 0.0) || !std::isfinite(mean_dist))
  {
    return false;
  }

  const double scale = std::sqrt(2.0) / mean_dist;

  pts_norm = centred * scale;

  T.setZero();
  T(0, 0) = scale;
  T(1, 1) = scale;
  T(0, 2) = -scale * centroid.x();
  T(1, 2) = -scale * centroid.y();
  T(2, 2) = 1.0;

  return true;
}

}  // namespace

StatusCode estimateHomographyDLT(
  const Eigen::Ref<const Eigen::Matrix2Xd> &src, const Eigen::Ref<const Eigen::Matrix2Xd> &dst,
  Eigen::Matrix3d &H_out
)
{
  // ------------------------------------------------------------------
  // 1. Input validation (no out-param mutation on failure).
  // ------------------------------------------------------------------
  const Eigen::Index n = src.cols();
  if (n < 4 || dst.cols() != n)
  {
    return StatusCode::INVALID_INPUT;
  }
  if (!src.allFinite() || !dst.allFinite())
  {
    return StatusCode::INVALID_INPUT;
  }

  // ------------------------------------------------------------------
  // 2. Hartley normalisation.
  // ------------------------------------------------------------------
  Eigen::Matrix2Xd src_norm(2, n);
  Eigen::Matrix2Xd dst_norm(2, n);
  Eigen::Matrix3d T_src = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d T_dst = Eigen::Matrix3d::Identity();

  if (!hartleyNormalise(src, src_norm, T_src) || !hartleyNormalise(dst, dst_norm, T_dst))
  {
    // Coincident src or dst points: cannot identify a homography.
    return StatusCode::DEGENERATE_CONFIG;
  }

  // ------------------------------------------------------------------
  // 3. Build the 2N x 9 DLT design matrix.
  //
  // For each correspondence (x, y) <-> (x', y'), two rows are appended
  // in the standard Hartley-Zisserman form:
  //
  //   [  0   0   0   -x   -y   -1    y'x   y'y   y'  ]
  //   [  x   y   1    0    0    0   -x'x  -x'y  -x' ]
  //
  // Stacking gives A; h = vec_row(H) lies in the right null-space.
  // ------------------------------------------------------------------
  Eigen::MatrixXd A(2 * n, 9);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const double x = src_norm(0, i);
    const double y = src_norm(1, i);
    const double xp = dst_norm(0, i);
    const double yp = dst_norm(1, i);

    A(2 * i, 0) = 0.0;
    A(2 * i, 1) = 0.0;
    A(2 * i, 2) = 0.0;
    A(2 * i, 3) = -x;
    A(2 * i, 4) = -y;
    A(2 * i, 5) = -1.0;
    A(2 * i, 6) = yp * x;
    A(2 * i, 7) = yp * y;
    A(2 * i, 8) = yp;

    A(2 * i + 1, 0) = x;
    A(2 * i + 1, 1) = y;
    A(2 * i + 1, 2) = 1.0;
    A(2 * i + 1, 3) = 0.0;
    A(2 * i + 1, 4) = 0.0;
    A(2 * i + 1, 5) = 0.0;
    A(2 * i + 1, 6) = -xp * x;
    A(2 * i + 1, 7) = -xp * y;
    A(2 * i + 1, 8) = -xp;
  }

  // ------------------------------------------------------------------
  // 4. SVD: h is the right singular vector of A with smallest sigma.
  // ------------------------------------------------------------------
  Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();

  // sigma has min(2N, 9) entries; the ratio test below indexes sigma(7),
  // so we need at least 8 entries (N >= 4).
  if (sigma.size() < 8)
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // Degeneracy check on sigma(7):
  //   For N >= 5 the SVD returns 9 singular values; sigma(7) is the
  //   second-smallest, and sigma(8) is the null-space direction.
  //   For N == 4 the SVD returns 8 singular values; sigma(7) is the
  //   smallest computed, and the null-space direction sits in V.col(8)
  //   without a corresponding singular value entry.
  // In both cases, sigma(7) / sigma(0) below threshold means A is
  // rank-deficient and the homography is not identifiable.
  const double sigma_largest = sigma(0);
  const double sigma_min_nondegenerate = sigma(7);

  if (!(sigma_largest > 0.0) || !std::isfinite(sigma_largest))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  if (sigma_min_nondegenerate / sigma_largest < kDegenerateSingularRatio)
  {
    // Rank of A is effectively < 8 -> homography not identifiable
    // (e.g. all points collinear, or fewer than 4 distinct points).
    return StatusCode::DEGENERATE_CONFIG;
  }

  // h = last column of V (column 8). Reshape row-major to a 3x3.
  const Eigen::VectorXd h = svd.matrixV().col(8);

  Eigen::Matrix3d H_hat;
  H_hat << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);

  // ------------------------------------------------------------------
  // 5. Denormalise.
  // ------------------------------------------------------------------
  // T_src and T_dst are scale+translation similarities; their inverses
  // are cheap closed-forms, but Eigen's inverse() of a 3x3 is fast and
  // numerically fine here.
  Eigen::Matrix3d H = T_dst.inverse() * H_hat * T_src;

  // ------------------------------------------------------------------
  // 6. Final normalisation (||H||_F = 1) and sign convention.
  // ------------------------------------------------------------------
  const double fro_norm = H.norm();
  if (!(fro_norm > 0.0) || !std::isfinite(fro_norm))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  H /= fro_norm;

  // Sign convention: prefer det(H) >= 0 so that, when the underlying
  // scene is orientation-preserving, the recovered map is too. H is
  // only defined up to sign, so flipping is free.
  if (H.determinant() < 0.0)
  {
    H = -H;
  }

  if (!H.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  H_out = H;
  return StatusCode::OK;
}

}  // namespace camxiom::init
