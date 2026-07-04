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

#ifndef CAMXIOM__INIT__MEI_OMNI_HPP
#define CAMXIOM__INIT__MEI_OMNI_HPP

#include "camxiom/init/pinhole_zhang.hpp"  // PlanarObservation (shared init type)
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init
{

/// Output bundle for MEI (Mei 2007) omnidirectional initialisation.
///
/// Holds the recovered pinhole-like intrinsics K (skew = 0, K(2,2) = 1), the
/// unified-sphere offset xi, and the per-view extrinsics (R_i, t_i) such that
/// for every view i and every correspondence j in that view
///
///   P_cam_ij = R_i * (board_pt_ij_lifted_to_Z=0) + t_i.
///
/// The struct mirrors the result-bundle convention shared with the other
/// init routines (pinhole_zhang.hpp, kb4_fisheye.hpp). All fields are mutated
/// atomically: a non-OK status from estimateMEIInit() leaves `*this`
/// untouched.
struct MEIInitResult
{
  /// 3x3 upper-triangular pinhole-like intrinsics. K(2,2) = 1, skew = 0.
  Eigen::Matrix3d K{Eigen::Matrix3d::Identity()};

  /// Mei's unified-sphere offset xi (Mei 2007). xi = 0 is pinhole, xi = 1
  /// the parabolic mirror special case (the seed used during Phase A).
  double xi{1.0};

  /// Per-view world-to-camera rotation (P_cam = R * P_world + t).
  std::vector<Eigen::Matrix3d> R_per_view;

  /// Per-view world-to-camera translation (in the same metric as the input
  /// board coordinates).
  std::vector<Eigen::Vector3d> t_per_view;
};

/// Heuristic-seed + Ceres-polish initialisation for the MEI omnidirectional
/// model given a set of planar-target observations.
///
/// MEI forward projection (mirror of src/omnidirectional/projection.cpp):
///   r       = ||(X, Y, Z)||
///   denom   = Z + xi * r
///   m_x     = X / denom,    m_y = Y / denom
///   pixel.u = fx * m_x + skew * m_y + cx
///   pixel.v = fy * m_y + cy
/// Distortion is held at NONE for this routine (Mei's xi captures the
/// fisheye behaviour; plane-space radial / tangential distortion is left to
/// the downstream full calibration, calib::calibrate).
///
/// Phase A (heuristic seed): Use default K with fx = fy = image_height / 2,
/// cx = image_width / 2, cy = image_height / 2. Set xi = 1.0 (Mei's canonical
/// parabolic-mirror value). Run estimatePoseDLT per view to obtain initial
/// poses. NO IAC-based K refinement is performed because for MEI's xi != 0
/// the board-to-lifted-plane is not homographic (the denominator contains
/// sqrt(x^2 + y^2 + z^2)). Phase B's Ceres polish handles all K refinement.
///
/// Phase B (Ceres polish with xi locked): PnpSolver refines (fx, fy, cx, cy,
/// all per-view poses) starting from the Phase A heuristic seed. xi is held
/// at 1.0 via PnpFlag::FIX_PROJECTION_PARAMS: xi refinement is deferred to
/// the downstream full calibration with more data and multi-seed restarts.
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
///                       MEIInitResult. On any non-OK return,
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
///                                         the heuristic-seeded MEI model).
///         StatusCode::NUMERIC_ERROR       Final result contains
///                                         non-finite entries.
[[nodiscard]] StatusCode estimateMEIInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  MEIInitResult &result_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__MEI_OMNI_HPP
