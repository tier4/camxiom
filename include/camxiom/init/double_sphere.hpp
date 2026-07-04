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

#ifndef CAMXIOM__INIT__DOUBLE_SPHERE_HPP
#define CAMXIOM__INIT__DOUBLE_SPHERE_HPP

#include "camxiom/init/pinhole_zhang.hpp"  // PlanarObservation (shared init type)
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init
{

/// Output bundle for Double Sphere (Usenko 2018) initialisation.
///
/// Holds the recovered pinhole-like intrinsics K (skew = 0, K(2,2) = 1), the
/// Double Sphere first-sphere offset xi and blending factor alpha, and the
/// per-view extrinsics (R_i, t_i) such that for every view i and every
/// correspondence j in that view
///
///   P_cam_ij = R_i * (board_pt_ij_lifted_to_Z=0) + t_i.
///
/// The struct mirrors the result-bundle convention shared with the other
/// init routines (pinhole_zhang.hpp, kb4_fisheye.hpp, mei_omni.hpp). All fields
/// are mutated atomically: a non-OK status from estimateDSInit() leaves
/// `*this` untouched.
struct DSInitResult
{
  /// 3x3 upper-triangular pinhole-like intrinsics. K(2,2) = 1, skew = 0.
  Eigen::Matrix3d K{Eigen::Matrix3d::Identity()};

  /// Double Sphere first-sphere offset xi (Usenko 2018). Typical real-camera
  /// values lie in [-0.4, -0.05]; the seed used during Phase A is -0.2.
  double xi{-0.2};

  /// Double Sphere blending factor alpha (Usenko 2018). Range (0, 1); the
  /// seed used during Phase A is 0.5 (the central calibrated value reported
  /// across many real fisheye cameras in the Usenko paper).
  double alpha{0.5};

  /// Per-view world-to-camera rotation (P_cam = R * P_world + t).
  std::vector<Eigen::Matrix3d> R_per_view;

  /// Per-view world-to-camera translation (in the same metric as the input
  /// board coordinates).
  std::vector<Eigen::Vector3d> t_per_view;
};

/// Heuristic-seed + Ceres-polish initialisation for the Double Sphere model
/// given a set of planar-target observations.
///
/// Double Sphere forward projection (mirror of src/double_sphere/projection.cpp):
///   d1     = sqrt(X^2 + Y^2 + Z^2)
///   w      = xi * d1 + Z
///   d2     = sqrt(X^2 + Y^2 + w^2)
///   denom  = alpha * d2 + (1 - alpha) * w
///   m_x    = X / denom,    m_y = Y / denom
///   pixel.u = fx * m_x + skew * m_y + cx
///   pixel.v = fy * m_y + cy
/// Distortion is held at NONE for this routine (DS's xi, alpha capture the
/// fisheye behaviour; plane-space radial / tangential distortion is left to
/// the downstream full calibration, calib::calibrate).
///
/// Phase A (heuristic seed): Use default K with fx = fy = image_height / 2,
/// cx = image_width / 2, cy = image_height / 2. Set xi = -0.2 and
/// alpha = 0.5 (Usenko 2018 typical values). Run estimatePoseDLT per view
/// to obtain initial poses. NO IAC-based K refinement is performed — the
/// DS forward equation is non-affine in (X, Y, Z) so a Zhang-style linear
/// init is mathematically invalid (same reasoning as MEI).
///
/// Phase B (Ceres polish with xi and alpha locked): PnpSolver refines
/// (fx, fy, cx, cy, all per-view poses) starting from the Phase A heuristic
/// seed. xi and alpha are held at their seed values via
/// PnpFlag::FIX_PROJECTION_PARAMS: identification of these non-linear
/// projection parameters is deferred to the downstream full calibration
/// with more data and multi-seed restarts.
///
/// If phaseBLooksUsable() rejects the Ceres result we fall back to the
/// Phase A heuristic K + estimatePoseDLT poses and still return OK (Phase
/// A's seed is a usable initial guess for the downstream calibrator).
///
/// All numerics are double precision.
///
/// @param views          N >= 3 planar observations. Each view's
///                       board_pts/image_pts must have >= 4 columns and
///                       matching column counts, with finite entries.
/// @param image_width    Image width in pixels. Must be > 0.
/// @param image_height   Image height in pixels. Must be > 0.
/// @param result_out     On StatusCode::OK, set to a fully-populated
///                       DSInitResult. On any non-OK return,
///                       `result_out` is NOT mutated.
///
/// @return StatusCode::OK                  success.
///         StatusCode::INVALID_INPUT       N < 3, any view with M < 4 or
///                                         column mismatch, image_width
///                                         or image_height <= 0, or any
///                                         non-finite element in the
///                                         observations.
///         StatusCode::DEGENERATE_CONFIG   propagated from
///                                         estimatePoseDLT() (any view
///                                         fails to recover a pose under
///                                         the heuristic-seeded DS model).
///         StatusCode::NUMERIC_ERROR       Final result contains
///                                         non-finite entries.
[[nodiscard]] StatusCode estimateDSInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  DSInitResult &result_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__DOUBLE_SPHERE_HPP
