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

#ifndef CAMXIOM__CAMERA_INFO_YAML_HPP
#define CAMXIOM__CAMERA_INFO_YAML_HPP

#include "camxiom/compat.hpp"
#include "camxiom/types.hpp"

#include <string>

namespace camxiom
{

/// Serialize a CameraInfo to the ROS camera_calibration_parsers YAML layout
/// (image_width / image_height / camera_name / camera_matrix /
/// distortion_model / distortion_coefficients / rectification_matrix /
/// projection_matrix). The output is accepted by
/// camera_calibration_parsers::readCalibration and by camera drivers'
/// camera_info_url — without camxiom depending on ROS: the writer is plain
/// std C++ in the dependency-free core, fed from the same CameraInfo POD the
/// interop layers convert into.
///
/// The layout matches the intrinsic camera calibrator's emitter (fixed-point
/// numbers: matrices at 6 decimals in column-aligned multi-line blocks,
/// distortion coefficients at 12 decimals on one line), so files written here
/// are byte-for-byte the same style as that tool's output. The writer emits
/// what it is given; it does not validate K/D consistency (run
/// validateCameraModel on makeCameraModel(info) for that).
///
/// The output is independent of the process's global locale (numbers always
/// use `.` and no digit grouping). `camera_name` and `distortion_model` are
/// emitted plain when identifier-like and double-quoted (with escaping)
/// otherwise, so any string stays valid YAML.
[[nodiscard]] std::string toCameraInfoYaml(
  const CameraInfo &camera_info, const std::string &camera_name = "camera"
);

/// Write toCameraInfoYaml() to `path` (overwriting). Returns OK on success,
/// INVALID_INPUT when the file cannot be opened or fully written.
[[nodiscard]] StatusCode saveCameraInfoYaml(
  const std::string &path, const CameraInfo &camera_info, const std::string &camera_name = "camera"
);

}  // namespace camxiom

#endif  // CAMXIOM__CAMERA_INFO_YAML_HPP
