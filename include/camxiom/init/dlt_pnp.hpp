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

#ifndef CAMXIOM__INIT__DLT_PNP_HPP
#define CAMXIOM__INIT__DLT_PNP_HPP

#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

namespace camxiom::init
{

/// Model-agnostic Direct Linear Transform PnP (DLT-PnP).
///
/// Given N >= 6 correspondences between 3D world points and 2D image
/// observations, plus a fully specified camera model (intrinsics +
/// projection + distortion), recover the camera pose (R, t) such that
/// for every i the camera-frame point P_cam_i = R * world_pts.col(i) + t
/// lies along the bearing camxiom::pixelToRay64(model, image_pts.col(i)).
///
/// This routine is the model-agnostic foundation that the per-model init
/// estimators (KB4 fisheye, MEI omni, Double Sphere, EUCM) build on. By
/// converting each pixel into a unit bearing via the model's
/// pixelToRay64, the downstream linear system depends only on world points
/// and bearings — exactly the structure of the classical "calibrated"
/// PnP problem.
///
/// Handles BOTH coplanar and non-coplanar world points. Planarity is
/// detected automatically from the eigenvalues of the (Hartley-shifted)
/// world-point covariance matrix; coplanar inputs (e.g. flat calibration
/// boards in their own frame, where wz_i = 0 for all i) are routed to a
/// 9-DOF linear system that avoids the rank deficiency of the 12-DOF
/// formulation, while non-coplanar inputs use the 12-DOF formulation.
///
/// Pipeline:
///   1. Convert each pixel to a unit bearing via camxiom::pixelToRay64.
///      Any non-OK pixelToRay64 status (BEHIND_CAMERA, OUT_OF_FOV, ...)
///      is treated as a degenerate configuration of the input data.
///   2. Hartley normalisation on the 3D world points
///      (centroid -> origin, mean distance from origin -> sqrt(3)).
///      Bearings are already unit-norm by construction and are NOT
///      rescaled.
///   3. Planarity detection. Compute the eigenvalues of the (centred)
///      world-point covariance matrix; if the smallest divided by the
///      largest is below an internal ratio threshold, the points are
///      treated as coplanar.
///   4a. Non-coplanar path (12 DOF). Build the 2N x 12 design matrix
///       from ray_i x (R * X_i_n + t_n) = 0 with unknown
///       p = [r11 r12 r13 t1 | r21 r22 r23 t2 | r31 r32 r33 t3], solve
///       A p = 0 via BDCSVD, reshape and project onto SO(3) via
///       Procrustes.
///   4b. Coplanar path (9 DOF). With wz_n = 0 the constraint reduces to
///       ray_i x (r1 * wx_i + r2 * wy_i + t) = 0 (r1, r2: first two
///       columns of R). Build the 2N x 9 design matrix with unknown
///       p = [r1; r2; t], solve A p = 0 via BDCSVD, then orthogonalise
///       via Procrustes on M = [r1_raw, r2_raw, r1_raw x r2_raw]; the
///       third rotation column comes from the cross-product so the
///       average of the first two singular values is used as the scale.
///   5. Sign resolution: the cross-product constraint is invariant under
///      (R, t) -> (-R, -t) in raw form, so the SVD sign is ambiguous.
///      Resolve by requiring the majority of camera-frame points to lie
///      in front of the camera (positive depth: dot(P_cam_i, ray_i) > 0).
///      If the majority is negative we re-do the Procrustes step with
///      negated p (both 12-DOF and 9-DOF paths).
///   6. Denormalise:  t = t_n / scale_W - R * c_W
///      where c_W and scale_W are the world-point Hartley centroid and
///      scale.
///   7. Final validity: assert R is finite and t is finite. Reject
///      otherwise.
///
/// Output convention: the function returns R as Eigen::Matrix3d
/// and t as Eigen::Vector3d. The caller can convert R to a Rodrigues
/// vector for PnpInitialGuess via Eigen::AngleAxisd(R).
///
/// On any non-OK return, R_out and t_out are NOT mutated.
///
/// @param model       Fully specified camxiom CameraModel64.
/// @param world_pts   3 x N world-frame points (N >= 6).
/// @param image_pts   2 x N pixel observations (same N as world_pts).
/// @param R_out       On StatusCode::OK, set to a proper rotation
///                    matrix (det = +1, columns orthonormal).
/// @param t_out       On StatusCode::OK, set to the translation
///                    expressed in the camera frame, in metres
///                    (P_cam = R * P_world + t).
///
/// @return StatusCode::OK                 success.
///         StatusCode::INVALID_INPUT      N < 6, column mismatch, any
///                                        non-finite element, or empty
///                                        inputs.
///         StatusCode::DEGENERATE_CONFIG  one or more pixels have
///                                        pixelToRay64 non-OK status,
///                                        all world points coincide, all
///                                        bearings parallel, the
///                                        design-matrix rank is
///                                        insufficient for the chosen
///                                        path (< 11 for 12-DOF, < 8 for
///                                        the planar 9-DOF path), or sign
///                                        resolution fails (a majority of
///                                        world points lie behind the
///                                        camera under both sign choices
///                                        of the null vector).
///         StatusCode::NUMERIC_ERROR      Procrustes scale near zero, or
///                                        non-finite R/t after all steps.
[[nodiscard]] StatusCode estimatePoseDLT(
  const camxiom::CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix3Xd> &world_pts,
  const Eigen::Ref<const Eigen::Matrix2Xd> &image_pts, Eigen::Matrix3d &R_out,
  Eigen::Vector3d &t_out
);

/// Refined model-agnostic single-view PnP (DLT initialisation + non-linear
/// refinement). The OpenCV-free analogue of cv::solvePnP(SOLVEPNP_ITERATIVE).
///
/// First computes a linear pose with estimatePoseDLT, then refines (R, t) to
/// the geometric reprojection optimum with the camxiom PnP solver, holding the
/// intrinsics, distortion and projection parameters fixed so only the 6-DOF
/// pose moves. Prefer this over the raw estimatePoseDLT whenever pose accuracy
/// feeds a downstream metric (e.g. evaluating a held-out view under a fixed
/// model): the linear DLT pose alone carries a noticeable rotation bias on
/// planar boards, which would inflate any residual measured against it.
///
/// The model is taken as float CameraModel (the type the solver and callers
/// hold); it is widened internally for the DLT bearing lift. Like calibrate(),
/// if the full-model bearing lift fails (a peripheral board past the forward
/// distortion's fold-over radius) the DLT initialisation is retried with
/// distortion disabled; the refinement always uses the full model.
///
/// On any non-OK return, R_out and t_out are NOT mutated.
///
/// @param model       Fully specified camxiom CameraModel; held fixed during
///                    refinement.
/// @param world_pts   3 x N world-frame points (N >= 6).
/// @param image_pts   2 x N pixel observations (same N as world_pts).
/// @param R_out       On StatusCode::OK, the refined rotation (det = +1).
/// @param t_out       On StatusCode::OK, the refined camera-frame translation
///                    (P_cam = R * P_world + t).
///
/// @return StatusCode::OK   success. R_out / t_out hold the refined pose; if
///                          refinement does not converge to a finite result it
///                          falls back to the (valid) DLT pose, still OK.
///         (other)          the estimatePoseDLT status verbatim -- the linear
///                          initialisation is a prerequisite for refinement.
[[nodiscard]] StatusCode estimatePoseRefined(
  const camxiom::CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xd> &world_pts,
  const Eigen::Ref<const Eigen::Matrix2Xd> &image_pts, Eigen::Matrix3d &R_out,
  Eigen::Vector3d &t_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__DLT_PNP_HPP
