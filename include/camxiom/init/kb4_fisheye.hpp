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

#ifndef CAMXIOM__INIT__KB4_FISHEYE_HPP
#define CAMXIOM__INIT__KB4_FISHEYE_HPP

#include "camxiom/init/pinhole_zhang.hpp"  // PlanarObservation (shared init type)
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init
{

/// Output bundle for KB4 fisheye initialisation.
///
/// Holds the recovered pinhole intrinsics K (skew = 0, K(2,2) = 1), the four
/// KB4 distortion coefficients k1..k4 forming the polynomial
///
///   theta_d = theta + k1 * theta^3 + k2 * theta^5
///                   + k3 * theta^7 + k4 * theta^9,
///
/// and the per-view extrinsics (R_i, t_i) such that for every view i and
/// every correspondence j in that view
///
///   P_cam_ij = R_i * (board_pt_ij_lifted_to_Z=0) + t_i.
///
/// The struct mirrors the result-bundle convention shared with the other
/// init routines (pinhole_zhang.hpp). All fields are mutated atomically: a non-OK
/// status from estimateKB4Init() leaves `*this` untouched.
struct KB4InitResult
{
  /// 3x3 upper-triangular pinhole intrinsics. K(2,2) = 1, skew = 0.
  Eigen::Matrix3d K{Eigen::Matrix3d::Identity()};

  /// KB4 polynomial coefficients (k1, k2, k3, k4).
  Eigen::Vector4d D{Eigen::Vector4d::Zero()};

  /// Per-view world-to-camera rotation (P_cam = R * P_world + t).
  std::vector<Eigen::Matrix3d> R_per_view;

  /// Per-view world-to-camera translation (in the same metric as the input
  /// board coordinates).
  std::vector<Eigen::Vector3d> t_per_view;
};

/// Linear-init + best-effort Ceres polish for the 4-parameter Kannala-Brandt
/// fisheye model (KB4) given a set of planar-target observations.
///
/// Algorithm (Phase A, linear, 3 DOF):
///   1. Seed K with cx = image_width/2, cy = image_height/2, fx = fy =
///      half_image_diagonal / 3.0 (equidistant assumption placing the image
///      diagonal corner at ~172 deg incidence, so every pixel lifts to
///      theta < pi under the seed for any aspect ratio — peripheral and
///      >180-deg-FOV observations included). Build a CameraModel64 with
///      projection.type = FISHEYE_THETA (theta_max = pi) and
///      distortion.type = EQUIDISTANT (k_i = 0).
///   2. For each view i, lift the planar board points to Z = 0 and call
///      estimatePoseDLT() with the seed model to recover (R_i, t_i).
///      Any non-OK status here is reported as DEGENERATE_CONFIG.
///   3. Stack all correspondences (across views) into an M_total x 3
///      linear system A a = b which jointly estimates *only* the focal
///      length, k1, and k2. Higher-order coefficients k3 and k4 are
///      fundamentally unobservable from typical calibration-board
///      geometry: for a board spanning theta in roughly [0.03, 0.3] rad
///      the theta^9 column of the design matrix sits about 5e-5 of the
///      theta^1 column, far below the measurement-noise floor (~2.5e-3
///      rad). Including those columns in the LSQ produces large garbage
///      values that pollute the polynomial at every theta. We therefore
///      freeze k3 = k4 = 0 in Phase A and rely on Phase B (Ceres joint
///      refinement) to discover their values starting from zero — that
///      is the correct mathematical answer when the geometry truly does
///      not support those DOFs.
///      Each row reads
///         A_row = [ theta_true,  theta_true^3,  theta_true^5 ]
///         b_row = sqrt((u - cx)^2 + (v - cy)^2)
///      with theta_true derived from the seeded extrinsics. The unknown
///      vector is a = (fx, fx*k1, fx*k2); we recover fx = a(0), k1 =
///      a(1) / fx, k2 = a(2) / fx, and set k3 = k4 = 0.
///      Hartley-style column scaling is applied to A (each column
///      normalised to unit L2 norm before SVD, the solution unscaled
///      afterwards) to keep BDCSVD well-conditioned for the residual
///      column-norm spread of [theta, theta^3, theta^5]. The system is
///      rejected as DEGENERATE_CONFIG if sigma(2) / sigma(0) < 1e-12.
///      fx = fy is assumed (square pixels).
///   4. Update the model's intrinsics with the recovered fx (= fy) and
///      distortion to KB4 with coeffs = (k1, k2, 0, 0). cx, cy remain at
///      the image-centre seed.
///
/// Algorithm (Phase B, best-effort polish):
///   The camxiom::optimizer::PnpSolver supports joint refinement of
///   (K, D, per-view R, per-view t) — it is the same code path used by
///   the downstream consumer (see optimizer/pnp_solver.hpp). Phase B
///   refines fx, fy, cx, cy, k1, k2, and per-view poses via Ceres
///   PnpSolver. k3 and k4 are held fixed at 0.0 (the Phase A output
///   value) because they are not uniquely identifiable from typical
///   fisheye calibration geometries under sub-pixel noise; freeing them
///   creates aliasing local minima that inflate |D| by orders of
///   magnitude while reprojection RMS stays acceptable. The downstream
///   full calibration (calib::calibrate) may refine k3, k4 with additional
///   data. On any failure (solver_failure, non-finite output) we fall back
///   to the Phase A estimate and still return OK, because Phase A is
///   already a usable initial guess for the downstream calibrator.
///
/// All numerics are double precision.
///
/// @param views          N >= 3 planar observations. Each view's
///                       board_pts/image_pts must have >= 4 columns and
///                       matching column counts, with finite entries.
/// @param image_width    Image width in pixels. Must be > 0.
/// @param image_height   Image height in pixels. Must be > 0.
/// @param result_out     On StatusCode::OK, set to a fully-populated
///                       KB4InitResult. On any non-OK return,
///                       `result_out` is NOT mutated.
///
/// @return StatusCode::OK                  success.
///         StatusCode::INVALID_INPUT       N < 3, any view with M < 4 or
///                                         column mismatch, image_width
///                                         or image_height <= 0, or any
///                                         non-finite element in the
///                                         observations.
///         StatusCode::DEGENERATE_CONFIG   propagated from
///                                         estimatePoseDLT() (any view),
///                                         the linear KB4 LSQ is rank
///                                         deficient (sigma(2) / sigma(0)
///                                         < 1e-12), or a design-matrix
///                                         column has zero norm.
///         StatusCode::NUMERIC_ERROR       Phase A produced non-finite K
///                                         or D, or recovered a non-
///                                         physical focal length (fx <=
///                                         0). Phase B failures are
///                                         absorbed; they do NOT
///                                         propagate as NUMERIC_ERROR.
[[nodiscard]] StatusCode estimateKB4Init(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  KB4InitResult &result_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__KB4_FISHEYE_HPP
