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

#ifndef CAMXIOM__INIT__HOMOGRAPHY_HPP
#define CAMXIOM__INIT__HOMOGRAPHY_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

namespace camxiom::init
{

/// 2D-to-2D homography estimation via the Direct Linear Transform (DLT).
///
/// Given N >= 4 corresponding 2D points (src, dst), estimate the 3x3
/// homography H such that, for every i, the homogeneous projection of
/// src.col(i) by H equals dst.col(i) up to scale.
///
/// Pipeline (Hartley & Zisserman, MVG 2nd ed., Algorithm 4.2):
///   1. Hartley normalisation on src and dst (centroid -> origin, mean
///      distance from origin -> sqrt(2)).
///   2. Build the 2N x 9 DLT design matrix.
///   3. Solve A h = 0 by SVD (h = right singular vector with smallest
///      singular value).
///   4. Denormalise: H = T_dst^-1 * H_hat * T_src.
///   5. Normalise so that ||H||_F = 1 (the project-wide homography
///      normalisation convention).
///   6. Sign convention: if det(H) < 0 flip the sign so the principal
///      map is orientation preserving when the underlying scene allows.
///
/// The caller is responsible for any pixel-to-ray (or other
/// domain-specific) pre-conversion. This routine is the pure 2D-to-2D
/// primitive.
///
/// Inputs are treated as columns of points (one point per column),
/// matching Eigen's column-major default storage.
///
/// @param src   2 x N source points (N >= 4).
/// @param dst   2 x N destination points (must match src.cols()).
/// @param H_out On StatusCode::OK, set to the estimated homography
///              normalised to ||H||_F = 1. On any non-OK return value
///              H_out is left untouched.
/// @return StatusCode::OK                 success.
///         StatusCode::INVALID_INPUT      N < 4, src.cols() != dst.cols(),
///                                        or any non-finite element.
///         StatusCode::DEGENERATE_CONFIG  rank-deficient configuration: all
///                                        src or dst points collinear /
///                                        coincident, or otherwise fewer than
///                                        4 distinct correspondences.
///         StatusCode::NUMERIC_ERROR      a numerical pathology was detected
///                                        in the SVD or denormalisation step
///                                        (should not occur for finite,
///                                        well-conditioned inputs; emitted
///                                        as a defensive guard).
[[nodiscard]] StatusCode estimateHomographyDLT(
  const Eigen::Ref<const Eigen::Matrix2Xd> &src, const Eigen::Ref<const Eigen::Matrix2Xd> &dst,
  Eigen::Matrix3d &H_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__HOMOGRAPHY_HPP
