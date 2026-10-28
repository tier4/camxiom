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

#ifndef CAMXIOM__ROS_HPP
#define CAMXIOM__ROS_HPP

// Optional ROS interoperability layer.
//
// This header and its translation unit (src/compat/ros.cpp) are only compiled
// into the camxiom library when a ROS sensor_msgs package is found at build
// time (the CMake build defines CAMXIOM_HAS_ROS — a PUBLIC definition, so it
// propagates to CMake consumers through the exported targets — and adds the
// source). The guard below therefore requires BOTH the configure-time
// definition and the sensor_msgs headers: __has_include alone would expose
// declarations with no compiled implementation whenever the library was built
// without ROS on a machine that has sensor_msgs installed, turning every use
// into a link error. Without either, the header harmlessly expands to
// nothing. Non-CMake consumers of a ROS-enabled build must define
// CAMXIOM_HAS_ROS themselves.
//
// All conversions funnel through the ROS-free camxiom::CameraInfo POD declared
// in camxiom/compat.hpp, so the heavy K/D/distortion-model parsing stays in the
// ROS-free core and is never duplicated here.

#if defined(CAMXIOM_HAS_ROS) && __has_include(<sensor_msgs/msg/camera_info.hpp>)

#include "camxiom/compat.hpp"
#include "camxiom/types.hpp"

#include <sensor_msgs/msg/camera_info.hpp>

namespace camxiom
{

/// Build a CameraModel from a ROS CameraInfo message (copies k/d/distortion_model
/// into the camxiom::CameraInfo POD and delegates to the core makeCameraModel).
///
/// Import folds binning/ROI into K/P (the image_geometry adjustment), so the
/// model maps the published image's pixels.
[[nodiscard]] CameraModel makeCameraModel(const sensor_msgs::msg::CameraInfo &camera_info);

/// Export a model as a ROS CameraInfo message with the core
/// exportCameraInfo rules: written only when a ROS distortion model holds the
/// same lens, otherwise INVALID_MODEL and the message is left unchanged. On
/// OK, binning_x/binning_y/roi are reset to the spec defaults ("no binning /
/// full image") — the exported K/P describe the model's own pixel frame, and
/// stale metadata would be applied a second time by the next importer.
/// width/height are left to the caller.
[[nodiscard]] StatusCode exportCameraInfo(
  const CameraModel &model, sensor_msgs::msg::CameraInfo &camera_info_out
);

}  // namespace camxiom

#endif  // CAMXIOM_HAS_ROS && __has_include(<sensor_msgs/msg/camera_info.hpp>)

#endif  // CAMXIOM__ROS_HPP
