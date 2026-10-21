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

#ifndef CAMXIOM__INIT__PINHOLE_OPENCV_HPP
#define CAMXIOM__INIT__PINHOLE_OPENCV_HPP

#include "camxiom/init/planar_observation.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init
{

/// Diagnostics of the two-unknown focal solve, filled (when requested) even on
/// a DEGENERATE_CONFIG return so callers can log why an initialisation was
/// refused. The interesting failure class is views that all share one tilt
/// direction: every homography then contributes nearly proportional equations,
/// the 2x2 normal matrix stays technically rank-2 but is ill-conditioned, and
/// the least-squares solution is garbage that still passes the sign checks
/// (observed in the field as fx=11396 on a ~1600 px camera).
struct PinholeInitDiagnostics
{
  /// Singular values of the 2x2 normal matrix A^T A of the stacked focal
  /// equations, and their ratio. Note this is the NORMAL matrix: its condition
  /// number is the square of the design matrix A's. +infinity when the matrix
  /// is exactly rank-deficient.
  double normal_max_singular_value{0.0};
  double normal_min_singular_value{0.0};
  double normal_condition_number{0.0};
  /// Whether the least-squares solution had both 1/fx^2 and 1/fy^2 positive
  /// (a physical focal length exists).
  bool solution_positive{false};
  /// Estimated focal lengths from the raw least-squares solution, BEFORE any
  /// aspect_ratio override, so a caller can judge the raw fx/fy plausibility.
  /// 0 until solution_positive.
  double fx{0.0};
  double fy{0.0};
};

/// Default cap on PinholeInitDiagnostics::normal_condition_number above which
/// the solve is reported as DEGENERATE_CONFIG. Derived from the fx=11396
/// shared-tilt incident fixture (tests/pinhole_opencv_test.cpp): under 0.3 px
/// corner noise plus mild uncorrected radial distortion, the worst-case focal
/// error grows with the conditioning roughly as
///   cond 6e2 (tilt spread ~15 deg) -> ~2%,   cond 1e4 (~4 deg) -> ~14%,
///   cond 2e4 (~3 deg) -> ~21%,               cond 4e4 (~2 deg) -> ~40%,
///   cond 2e5 (~1 deg) -> fx up to ~7x true (the incident class).
/// 3e4 keeps every accepted solution inside the ~20% band a subsequent joint
/// refinement recovers from, sits 6x below the catastrophic incident class and
/// ~250x above healthy tilt-diverse geometry. A borderline refusal is
/// fail-safe: the caller simply retries once more tilt diversity arrives.
inline constexpr double kDefaultMaxNormalConditionNumber = 3e4;

/// OpenCV-compatible planar pinhole initialisation.
///
/// This follows the `initIntrinsicParams2D` path used by
/// `cv::calibrateCamera` when CALIB_USE_INTRINSIC_GUESS is not set:
///   1. Fix the principal point at ((width - 1) / 2, (height - 1) / 2).
///   2. Estimate one homography per planar view.
///   3. Build the two-equation focal-length system contributed by each
///      homography and solve it in least squares.
///   4. Recover positive fx/fy with OpenCV's sqrt(abs(1/f)) convention.
///
/// This does not estimate principal point or skew from the linear system.
/// That reduced parameterisation is deliberately the same initialisation used
/// by OpenCV before its joint non-linear optimisation of intrinsics,
/// distortion and per-view poses.
///
/// `aspect_ratio == 0` estimates fx and fy independently, matching
/// calibrateCamera without CALIB_FIX_ASPECT_RATIO. A positive value enforces
/// fx / fy == aspect_ratio using OpenCV's averaging rule.
///
/// No focal-length magnitude heuristic is applied: very large finite focal
/// lengths are valid for narrow-angle cameras.
///
/// The linear system's conditioning IS checked: when the 2x2 normal matrix's
/// condition number exceeds `max_normal_condition_number` the solve is
/// reported as DEGENERATE_CONFIG instead of returning a numerically meaningless
/// (but possibly positive) focal pair. Pass +infinity to disable the gate.
/// `diagnostics` (optional) receives the conditioning and the raw solution
/// even when the return is DEGENERATE_CONFIG, so callers can log the refusal.
[[nodiscard]] StatusCode estimatePinholeOpenCv(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  double aspect_ratio, Eigen::Matrix3d &K_out, PinholeInitDiagnostics *diagnostics,
  double max_normal_condition_number
);

/// Convenience overload with no diagnostics output and the default
/// conditioning gate.
[[nodiscard]] StatusCode estimatePinholeOpenCv(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  double aspect_ratio, Eigen::Matrix3d &K_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__PINHOLE_OPENCV_HPP
