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

#ifndef CAMXIOM__INIT__PINHOLE_ZHANG_HPP
#define CAMXIOM__INIT__PINHOLE_ZHANG_HPP

#include "camxiom/init/planar_observation.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init
{

/// Zhang's homography-based pinhole intrinsics initialisation.
///
/// Given N >= 3 planar-target observations, recover the pinhole intrinsics
/// matrix K via the absolute conic / image-of-the-absolute-conic (IAC)
/// linear system. Skew is estimated freely (full 6-DOF IAC); callers that
/// want zero skew should force K(0, 1) = 0 after this routine returns.
///
/// Algorithm (Zhang, "A Flexible New Technique for Camera Calibration",
/// PAMI 2000):
///   1. For each view i, compute H_i = estimateHomographyDLT(board, image).
///   2. Each view contributes two linear constraints on omega = K^-T K^-1
///      (the IAC) via the orthonormality of the first two columns of the
///      rotation, encoded as 6-vectors of column products.
///   3. Stack 2N constraints into V (2N x 6); solve V b = 0 by SVD.
///   4. Reconstruct omega from b, verify positive-definiteness, factor
///      omega = L L^T (Cholesky) and recover K = (L^T)^-1.
///   5. Normalise so K(2, 2) = 1.
///
/// Hartley normalisation is applied transitively through
/// estimateHomographyDLT. The output convention K(2, 2) = 1 makes
/// the result directly usable as an `IntrinsicsModel` (after casting to
/// float, if desired).
///
/// @param views  Vector of planar observations. N = views.size() must be
///               >= 3, each observation must have >= 4 correspondences
///               with matching column counts and finite values.
/// @param K_out  On StatusCode::OK, set to the recovered 3x3 upper-
///               triangular intrinsics matrix normalised so K(2,2) = 1
///               and with K(0,0), K(1,1) > 0. On any non-OK return value
///               K_out is left untouched.
/// @return StatusCode::OK                 success.
///         StatusCode::INVALID_INPUT      N < 3, any view with M < 4,
///                                        column-count mismatch within a
///                                        view, empty matrices, or any
///                                        non-finite element. Also
///                                        propagated from
///                                        estimateHomographyDLT's
///                                        INVALID_INPUT branch.
///         StatusCode::DEGENERATE_CONFIG  Views' orientations are near-
///                                        parallel so the IAC null space
///                                        is not 1-dimensional, or a
///                                        per-view homography itself
///                                        returned DEGENERATE_CONFIG.
///         StatusCode::NUMERIC_ERROR      omega is not PSD, Cholesky
///                                        failed, K(2,2) <= 0, fx or fy
///                                        <= 0, or another numerical
///                                        pathology was detected. Also
///                                        propagated from
///                                        estimateHomographyDLT's
///                                        NUMERIC_ERROR branch.
[[nodiscard]] StatusCode estimatePinholeZhang(
  const std::vector<PlanarObservation> &views, Eigen::Matrix3d &K_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__PINHOLE_ZHANG_HPP
