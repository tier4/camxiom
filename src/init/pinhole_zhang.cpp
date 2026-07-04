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

#include "camxiom/init/pinhole_zhang.hpp"

#include "camxiom/init/homography.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <cmath>
#include <cstddef>

namespace camxiom::init
{

namespace
{

/// Minimum number of views required by Zhang's method.
///
/// Each view contributes 2 independent linear constraints on omega (which
/// has 5 unknowns up to scale: B11, B12, B22, B13, B23, B33 with one
/// gauge fixed). N >= 3 provides 6 >= 5 constraints. The DEGENERATE_CONFIG
/// path catches the case where the constraints are not linearly
/// independent (e.g. all views share the same rotation).
constexpr std::size_t kMinViews = 3U;

/// Minimum number of point correspondences per view (homography needs 4).
constexpr Eigen::Index kMinPointsPerView = 4;

/// Threshold for sigma(4) / sigma(0). A small ratio means the
/// second-smallest singular value is also tiny relative to the largest,
/// meaning the null space of V is multi-dimensional (rank of V is less
/// than 5). In that case the IAC is not uniquely identifiable and we
/// return DEGENERATE_CONFIG.
///
/// Mirrors MS1-1's `kDegenerateSingularRatio` pattern.
constexpr double kIacDegenerateSingularRatio = 1e-12;

/// Build the row pair (v_12, v_11 - v_22) contributed by a single
/// homography to the IAC linear system V b = 0.
///
/// Encoding follows Zhang's notation:
///   v_ij(H) = [ h_i1 h_j1,
///               h_i1 h_j2 + h_i2 h_j1,
///               h_i2 h_j2,
///               h_i3 h_j1 + h_i1 h_j3,
///               h_i3 h_j2 + h_i2 h_j3,
///               h_i3 h_j3 ]
/// (1-indexed: h_ij = column i, row j of H).
///
/// `out` must point to a writable 2 x 6 block; row 0 receives v_12, row 1
/// receives v_11 - v_22.
void appendViewConstraints(const Eigen::Matrix3d &H, Eigen::Ref<Eigen::Matrix<double, 2, 6>> out)
{
  // Map Zhang's 1-indexed (column, row) to Eigen's 0-indexed (row, col).
  //   h_i1 = H(0, i-1), h_i2 = H(1, i-1), h_i3 = H(2, i-1)
  const double h11 = H(0, 0);
  const double h21 = H(1, 0);
  const double h31 = H(2, 0);
  const double h12 = H(0, 1);
  const double h22 = H(1, 1);
  const double h32 = H(2, 1);

  // v_12 (constraint: h1^T omega h2 = 0).
  out(0, 0) = h11 * h12;
  out(0, 1) = h11 * h22 + h21 * h12;
  out(0, 2) = h21 * h22;
  out(0, 3) = h31 * h12 + h11 * h32;
  out(0, 4) = h31 * h22 + h21 * h32;
  out(0, 5) = h31 * h32;

  // v_11 - v_22 (constraint: h1^T omega h1 - h2^T omega h2 = 0).
  out(1, 0) = h11 * h11 - h12 * h12;
  out(1, 1) = 2.0 * (h11 * h21 - h12 * h22);
  out(1, 2) = h21 * h21 - h22 * h22;
  out(1, 3) = 2.0 * (h31 * h11 - h32 * h12);
  out(1, 4) = 2.0 * (h31 * h21 - h32 * h22);
  out(1, 5) = h31 * h31 - h32 * h32;
}

}  // namespace

StatusCode estimatePinholeZhang(const std::vector<PlanarObservation> &views, Eigen::Matrix3d &K_out)
{
  // ------------------------------------------------------------------
  // 1. Input validation (no out-param mutation on failure).
  // ------------------------------------------------------------------
  const std::size_t n_views = views.size();
  if (n_views < kMinViews)
  {
    return StatusCode::INVALID_INPUT;
  }

  for (const auto &view : views)
  {
    const Eigen::Index m_board = view.board_pts.cols();
    const Eigen::Index m_image = view.image_pts.cols();
    if (m_board < kMinPointsPerView || m_image != m_board)
    {
      return StatusCode::INVALID_INPUT;
    }
    if (!view.board_pts.allFinite() || !view.image_pts.allFinite())
    {
      return StatusCode::INVALID_INPUT;
    }
  }

  // ------------------------------------------------------------------
  // 2. Per-view homography estimation.
  //
  // A view-level failure stops estimation and propagates the precise
  // failure category from MS1-1.
  // ------------------------------------------------------------------
  std::vector<Eigen::Matrix3d> homographies;
  homographies.reserve(n_views);
  for (const auto &view : views)
  {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    const StatusCode status = estimateHomographyDLT(view.board_pts, view.image_pts, H);
    if (status != StatusCode::OK)
    {
      // Propagate as-is: MS1-1's enum values (INVALID_INPUT,
      // DEGENERATE_CONFIG, NUMERIC_ERROR) line up exactly with the
      // contract this routine advertises.
      return status;
    }
    homographies.push_back(H);
  }

  // ------------------------------------------------------------------
  // 3. Assemble V (2N x 6) from the IAC constraints.
  // ------------------------------------------------------------------
  const Eigen::Index n_rows = static_cast<Eigen::Index>(2U * n_views);
  Eigen::MatrixXd V(n_rows, 6);
  for (std::size_t i = 0; i < n_views; ++i)
  {
    const Eigen::Index row_offset = static_cast<Eigen::Index>(2U * i);
    appendViewConstraints(homographies[i], V.block<2, 6>(row_offset, 0));
  }

  // ------------------------------------------------------------------
  // 4. Solve V b = 0 via SVD; b = right singular vector with smallest sv.
  // ------------------------------------------------------------------
  Eigen::BDCSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
  const Eigen::VectorXd &sigma = svd.singularValues();

  // V has 6 columns -> the SVD returns min(2N, 6) singular values.
  // With N >= 3 this is always 6.
  if (sigma.size() < 6)
  {
    return StatusCode::NUMERIC_ERROR;
  }

  const double sigma_largest = sigma(0);
  if (!(sigma_largest > 0.0) || !std::isfinite(sigma_largest))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // ------------------------------------------------------------------
  // 5. Degeneracy check on sigma(4).
  //
  // sigma(5) is the null-space direction; sigma(4) is the second-
  // smallest. If sigma(4) / sigma(0) is below threshold, V is rank < 5
  // and the IAC null space is multi-dimensional (e.g. all views have
  // the same rotation, or two-axis-only tilt). The estimate is not
  // identifiable.
  // ------------------------------------------------------------------
  if (sigma(4) / sigma_largest < kIacDegenerateSingularRatio)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  // b = V.col(5) = (B11, B12, B22, B13, B23, B33).
  Eigen::VectorXd b = svd.matrixV().col(5);
  if (!b.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // ------------------------------------------------------------------
  // 6. Sign convention: enforce B11 > 0 (since omega(0,0) is a
  //    sum-of-squares of K^-1's first row, it must be positive).
  // ------------------------------------------------------------------
  if (!(b(0) > 0.0))
  {
    if (b(0) < 0.0)
    {
      b = -b;
    }
    else
    {
      // b(0) == 0 (NaN already rejected by the allFinite check above).
      // B11 = 0 would mean K^-1's first row is the zero vector, so K is
      // singular: not a physical pinhole camera.
      return StatusCode::NUMERIC_ERROR;
    }
  }

  const double B11 = b(0);
  const double B12 = b(1);
  const double B22 = b(2);
  const double B13 = b(3);
  const double B23 = b(4);
  const double B33 = b(5);

  // ------------------------------------------------------------------
  // 7. Reconstruct omega = K^-T K^-1 (3x3 symmetric).
  // ------------------------------------------------------------------
  Eigen::Matrix3d omega;
  omega << B11, B12, B13, B12, B22, B23, B13, B23, B33;

  // ------------------------------------------------------------------
  // 8. PSD verification.
  //
  // The SVD always returns *some* b in the right null space; only its
  // re-shaping as a symmetric 3x3 may fail to be PSD when the data is
  // noisy enough to leave the geometric "omega cone" interior. The
  // smallest eigenvalue check is the canonical guard.
  //
  // Two-stage PSD check:
  //   1. SelfAdjointEigenSolver -> smallest eigenvalue > 0.
  //      This is the structural test: noisy IAC reconstructions sometimes
  //      produce an omega with a negative eigenvalue that has nothing to
  //      do with Cholesky precision -- the matrix is simply not PSD.
  //      Catching this here gives a cleaner diagnostic than waiting for
  //      LLT's `info()` to flag it.
  //   2. Eigen::LLT -> info() == Success.
  //      Authoritative final check: handles fp-rounding cases that pass
  //      the eigenvalue threshold but fail the actual factorisation.
  // Cost: SelfAdjointEigenSolver on a fixed-size 3x3 is cheap (well under
  // 1 microsecond), so the double-check is essentially free.
  // ------------------------------------------------------------------
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(omega);
  if (eig.info() != Eigen::Success)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  const Eigen::Vector3d eigvals = eig.eigenvalues();
  // SelfAdjointEigenSolver returns eigenvalues in increasing order.
  if (!(eigvals(0) > 0.0) || !std::isfinite(eigvals(0)))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // ------------------------------------------------------------------
  // 9. Cholesky: omega = L L^T with L lower-triangular.
  //    Then K^-T = L (up to scale), so K = (L^T)^-1.
  // ------------------------------------------------------------------
  Eigen::LLT<Eigen::Matrix3d> llt(omega);
  if (llt.info() != Eigen::Success)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  const Eigen::Matrix3d L = llt.matrixL();

  // K = (L^T)^-1. Use solve on identity for numerical robustness over
  // an explicit inverse.
  Eigen::Matrix3d K =
    L.transpose().triangularView<Eigen::Upper>().solve(Eigen::Matrix3d::Identity());

  // ------------------------------------------------------------------
  // 10. Normalise K(2,2) = 1 and validate physical sanity.
  // ------------------------------------------------------------------
  const double k22 = K(2, 2);
  if (!(k22 > 0.0) || !std::isfinite(k22))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  K /= k22;

  if (!(K(0, 0) > 0.0) || !(K(1, 1) > 0.0))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // ------------------------------------------------------------------
  // 11. Final finite check before mutating the out parameter.
  // ------------------------------------------------------------------
  if (!K.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  K_out = K;
  return StatusCode::OK;
}

}  // namespace camxiom::init
