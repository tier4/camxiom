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

#ifndef CAMXIOM__OPENCV__PNP_HPP
#define CAMXIOM__OPENCV__PNP_HPP

/// @file pnp.hpp
/// @brief Lightweight PnP and reprojection utilities built on top of camxiom.
///
/// These functions provide a unified solvePnP / reprojectRmse interface that
/// works with ALL camera models supported by camxiom (pinhole, fisheye,
/// omnidirectional, double sphere, EUCM). Internally they use
/// camxiom::opencv::undistortPoints to normalize observed pixel coordinates,
/// then delegate to cv::solvePnP with an identity camera matrix.
///
/// Only compiled into the camxiom library when OpenCV is found at build time
/// (see the guard rationale in camxiom/opencv.hpp).

#if defined(CAMXIOM_HAS_OPENCV) && __has_include(<opencv2/core.hpp>)

#include "camxiom/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace camxiom::opencv
{

using camxiom::CameraModel;
using camxiom::SolverOptions;

struct SolvePnPConfig
{
  int method{6};            // cv::SOLVEPNP_IPPE = 6  (best for planar targets)
  int fallback_method{0};   // cv::SOLVEPNP_ITERATIVE = 0
  bool use_fallback{true};  // Try fallback on failure
};

/// Solve PnP using camxiom's unified undistortion for any camera model.
/// @return true on success.
[[nodiscard]] bool solvePnP(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points,
  const std::vector<cv::Point2f> &image_points, cv::Vec3d &rvec_out, cv::Vec3d &tvec_out,
  const SolvePnPConfig &config = SolvePnPConfig{},
  const SolverOptions &solver_options = SolverOptions{}
);

/// Compute reprojection RMSE using camxiom's unified projection.
/// Points that fail to project (e.g. outside the model FOV) are excluded;
/// the RMSE is averaged over the successfully projected points only.
/// @return RMSE in pixels, or infinity on failure / when no point projects.
double reprojectRmse(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points,
  const std::vector<cv::Point2f> &image_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec
);

}  // namespace camxiom::opencv

#endif  // CAMXIOM_HAS_OPENCV && __has_include(<opencv2/core.hpp>)

#endif  // CAMXIOM__OPENCV__PNP_HPP
