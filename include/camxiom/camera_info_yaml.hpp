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
#include "camxiom/types64.hpp"

#include <string>

namespace camxiom
{

/// Serialize a CameraInfo to the ROS camera_calibration_parsers YAML layout
/// (image_width / image_height / camera_name / camera_matrix /
/// distortion_model / distortion_coefficients / rectification_matrix /
/// projection_matrix). The output is accepted by
/// camera_calibration_parsers::readCalibration and by camera drivers'
/// camera_info_url — without camxiom depending on ROS: the writer is plain
/// std C++ in the core, fed from the same CameraInfo POD the
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

/// Numeric formatting for serialization. ROUND_TRIP emits enough significant
/// digits to recover every finite double exactly, including subnormals and -0.
enum class CameraInfoYamlPrecision : std::uint8_t { FIXED = 0U, ROUND_TRIP = 1U };

/// Explicit-precision overloads. The original overloads keep FIXED formatting.
/// The precision comes last and nothing is defaulted, so a call that the
/// original overloads accept, such as `toCameraInfoYaml(info, {})`, never
/// resolves here.
[[nodiscard]] std::string toCameraInfoYaml(
  const CameraInfo &camera_info, const std::string &camera_name, CameraInfoYamlPrecision precision
);
[[nodiscard]] StatusCode saveCameraInfoYaml(
  const std::string &path, const CameraInfo &camera_info, const std::string &camera_name,
  CameraInfoYamlPrecision precision
);

/// Serialize camxiom's own description, one key per field:
///
///     image_width: 1280
///     image_height: 960
///     projection_model: equidistant
///     projection_parameters: []
///     distortion_model: kb4
///     distortion_parameters: [0.02, -0.005, 0.0, 0.0]
///     intrinsic_parameters: [320.0, 320.0, 639.5, 479.5, 0.0]
///
/// Every number is written with the fewest digits that read back as the same
/// double, so a file round-trips exactly. Like toCameraInfoYaml it writes
/// what it is given and does not depend on the global locale.
[[nodiscard]] std::string toCamxiomCameraInfoYaml(const CamxiomCameraInfo &camera_info);

/// Write toCamxiomCameraInfoYaml() to `path` (overwriting). INVALID_INPUT
/// when the file cannot be opened or fully written.
[[nodiscard]] StatusCode saveCamxiomCameraInfoYaml(
  const std::string &path, const CamxiomCameraInfo &camera_info
);

/// Read a ROS CameraInfo YAML (the camera_calibration_parsers layout above:
/// image_width, image_height, camera_matrix, distortion_model and
/// distortion_coefficients; rectification_matrix, projection_matrix,
/// binning_x, binning_y and roi when present; other keys such as camera_name
/// are ignored). Binning and ROI are folded into K, P and the image size
/// exactly as the ROS layer does for a sensor_msgs message, so the result
/// describes the image's own pixels (and the writer above, which has no
/// binning or ROI to write, round-trips it). INVALID_INPUT for a document
/// outside that layout or one that has a projection_model key (the camxiom
/// layout). The YAML readers refuse malformed YAML and, beyond what YAML
/// allows, input past 1 MiB, a second document, a repeated key, collections
/// nested past 16 levels, more than 256 keys in one mapping or 4096 items in
/// one sequence, and a quoted number.
[[nodiscard]] StatusCode parseCameraInfoYaml(const std::string &yaml, CameraInfo &camera_info_out);

/// Read camxiom's layout: exactly the seven keys toCamxiomCameraInfoYaml
/// writes. INVALID_INPUT for a missing, extra or malformed key.
[[nodiscard]] StatusCode parseCamxiomCameraInfoYaml(
  const std::string &yaml, CamxiomCameraInfo &camera_info_out
);

/// Read a camera model from either layout: a document with a
/// projection_model key is camxiom's, any other is a ROS CameraInfo. On OK,
/// model_out is a valid model and image_size_out the image size the file
/// gives. INVALID_INPUT when the document cannot be read (or the file
/// opened), the validation status when it does not describe a valid model.
[[nodiscard]] StatusCode parseCameraModelYaml(
  const std::string &yaml, CameraModel &model_out, ImageSize &image_size_out
);
[[nodiscard]] StatusCode parseCameraModelYaml64(
  const std::string &yaml, CameraModel64 &model_out, ImageSize &image_size_out
);
[[nodiscard]] StatusCode loadCameraModelYaml(
  const std::string &path, CameraModel &model_out, ImageSize &image_size_out
);
[[nodiscard]] StatusCode loadCameraModelYaml64(
  const std::string &path, CameraModel64 &model_out, ImageSize &image_size_out
);

}  // namespace camxiom

#endif  // CAMXIOM__CAMERA_INFO_YAML_HPP
