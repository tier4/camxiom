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

#ifndef CAMXIOM__COMPAT_HPP
#define CAMXIOM__COMPAT_HPP

#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace camxiom
{

/// A ROS sensor_msgs/CameraInfo without the ROS dependency: the same fields,
/// so the K/D/distortion_model reading in makeCameraModel() lives in the
/// ROS-free core and the optional ROS layer (camxiom/ros.hpp) only
/// copies fields.
struct CameraInfo
{
  std::array<double, 9> k{
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};  // intrinsic camera matrix (row-major)
  std::vector<double> d{};                           // distortion coefficients
  std::string distortion_model{};                    // distortion model string
  std::array<double, 9> r{
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};  // rectification matrix (row-major)
  std::array<double, 12> p{
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};  // projection matrix (row-major)
  std::uint32_t width{0U};
  std::uint32_t height{0U};
};

/// camxiom's own camera description: every valid CameraModel, field by field.
/// Unlike CameraInfo it keeps the projection law and its parameters apart
/// from the lens correction, names both the way toString does, and says
/// "none" when there is no correction instead of writing zeros.
struct CamxiomCameraInfo
{
  std::uint32_t image_width{0U};
  std::uint32_t image_height{0U};
  /// toString(ProjectionModelType), e.g. "pinhole", "equidistant", "eucm".
  std::string projection_model{};
  /// The law's own parameters: xi (omnidirectional), xi and alpha (double
  /// sphere), alpha and beta (EUCM); empty for every other law.
  std::vector<double> projection_parameters{};
  /// toString(DistortionModelType), e.g. "none", "radtan5", "kb4".
  std::string distortion_model{};
  /// Exactly the correction's coefficients (4 for radtan4 and kb4, 5 for
  /// radtan5, 8 for rational8 and kb8, 12 for thin_prism12, 14 for tilted14,
  /// none for "none"), in the order the model stores them.
  std::vector<double> distortion_parameters{};
  /// fx, fy, cx, cy, skew.
  std::array<double, 5> intrinsic_parameters{{1.0, 1.0, 0.0, 0.0, 0.0}};
};

/// Build a CameraModel from a raw camera description. The distortion_model
/// name gives the projection law and the lens correction: the ROS names
/// `plumb_bob` / `rational_polynomial` (pinhole with radtan / rational, the
/// correction chosen by the length of d) and `equidistant` (the equidistant
/// law with KB4), and camxiom's own names (`radtan4`, `radtan5`, `rational8`,
/// `thin_prism12`, `tilted14`, `kb4`, `kb8`, `ideal_equidistant`, `equisolid`,
/// `stereographic`, `orthographic`, `omnidirectional` / `double_sphere` /
/// `eucm` with their projection parameters first in d, and older aliases).
/// An empty name means pinhole, with the correction chosen by the length of
/// d. A description that is not a valid model comes back as one that
/// validateCameraModel rejects.
[[nodiscard]] CameraModel makeCameraModel(const CameraInfo &camera_info);

/// Parse K/D directly into double storage, without a float intermediate.
[[nodiscard]] CameraModel64 makeCameraModel64(const CameraInfo &camera_info);

/// Export a model as a ROS CameraInfo, only when the ROS distortion models
/// can hold the same lens:
///
/// | model                                        | distortion_model      | D          |
/// |----------------------------------------------|-----------------------|------------|
/// | PINHOLE + NONE / RADTAN4 / RADTAN5           | `plumb_bob`           | 5 (padded) |
/// | PINHOLE + RATIONAL8                          | `rational_polynomial` | 8          |
/// | PINHOLE + THIN_PRISM12 / TILTED14, extra 0   | `rational_polynomial` | 8          |
/// | EQUIDISTANT + NONE / KB4                     | `equidistant`         | 4 (padded) |
/// | EQUIDISTANT + KB8 with k5..k8 = 0            | `equidistant`         | 4          |
///
/// Zeros are added only where the ROS name fixes the length; no non-zero
/// coefficient is dropped (a RADTAN4 model that carries k3 in a fifth
/// coefficient writes it). Every other model returns INVALID_MODEL and
/// leaves camera_info_out unchanged: a prism or tilt term, KB8 terms past
/// k4, a non-zero skew (ROS defines K without one and its receivers ignore
/// it), and every projection other than pinhole and the equidistant law. An
/// invalid model returns its validation status, also without writing. On
/// OK, k is [fx 0 cx; 0 fy cy; 0 0 1], r the identity and p [K | 0];
/// width/height are left to the caller.
[[nodiscard]] StatusCode exportCameraInfo(const CameraModel &model, CameraInfo &camera_info_out);
[[nodiscard]] StatusCode exportCameraInfo64(
  const CameraModel64 &model, CameraInfo &camera_info_out
);

/// Build a CameraModel from camxiom's own description. The names must be the
/// toString names, projection_parameters must hold exactly the law's
/// parameters and distortion_parameters exactly the correction's
/// coefficients; theta_max is derived from the model (updateThetaMax). A
/// description that does not give a valid model comes back as one that
/// validateCameraModel rejects.
[[nodiscard]] CameraModel makeCameraModel(const CamxiomCameraInfo &camera_info);
[[nodiscard]] CameraModel64 makeCameraModel64(const CamxiomCameraInfo &camera_info);

/// Export any valid model as camxiom's own description; it reads back as the
/// same model (makeCameraModel), theta_max apart, which is derived again on
/// reading, and a five-term RADTAN4, which is written and read back as
/// RADTAN5 (the same lens). A float model's values are written as the shortest decimal that
/// reads back as the same float. An invalid model returns its validation
/// status and leaves camera_info_out unchanged. image_width/image_height are
/// left to the caller.
[[nodiscard]] StatusCode exportCameraInfo(
  const CameraModel &model, CamxiomCameraInfo &camera_info_out
);
[[nodiscard]] StatusCode exportCameraInfo64(
  const CameraModel64 &model, CamxiomCameraInfo &camera_info_out
);

}  // namespace camxiom

#endif  // CAMXIOM__COMPAT_HPP
