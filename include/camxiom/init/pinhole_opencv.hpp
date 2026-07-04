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
[[nodiscard]] StatusCode estimatePinholeOpenCv(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  double aspect_ratio, Eigen::Matrix3d &K_out
);

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__PINHOLE_OPENCV_HPP
