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

#ifndef CAMXIOM__COMPAT__INTERNAL_HPP
#define CAMXIOM__COMPAT__INTERNAL_HPP

#include "camxiom/compat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace camxiom
{
namespace
{

inline bool isFiniteScalar(const double value) { return std::isfinite(value); }

inline bool isFiniteArray9(const std::array<double, 9> &array_9)
{
  for (const double value : array_9)
  {
    if (!isFiniteScalar(value))
    {
      return false;
    }
  }
  return true;
}

inline bool isFiniteVector(const std::vector<double> &values)
{
  for (const double value : values)
  {
    if (!isFiniteScalar(value))
    {
      return false;
    }
  }
  return true;
}

inline int expectedDistortionCount(const PinholeCompatProfile profile)
{
  switch (profile)
  {
    case PinholeCompatProfile::CANONICAL:
      return -1;
    case PinholeCompatProfile::OPENCV_CALIB3D_D4:
      return 4;
    case PinholeCompatProfile::OPENCV_CALIB3D_D5:
      return 5;
    case PinholeCompatProfile::OPENCV_CALIB3D_D8:
      return 8;
    case PinholeCompatProfile::OPENCV_CALIB3D_D12:
      return 12;
    case PinholeCompatProfile::OPENCV_CALIB3D_D14:
      return 14;
    case PinholeCompatProfile::ROS_CAMERA_INFO_RAW:
      return -1;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const FisheyeCompatProfile profile)
{
  switch (profile)
  {
    case FisheyeCompatProfile::CANONICAL:
      return -1;
    case FisheyeCompatProfile::OPENCV_FISHEYE_D4:
      return 4;
    case FisheyeCompatProfile::ROS_CAMERA_INFO_EQUIDISTANT:
      return -1;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const OmnidirectionalCompatProfile profile)
{
  switch (profile)
  {
    case OmnidirectionalCompatProfile::CANONICAL:
      return -1;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4:
      return 4;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D5:
      return 5;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D8:
      return 8;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const DoubleSphereCompatProfile profile)
{
  switch (profile)
  {
    case DoubleSphereCompatProfile::CANONICAL:
      return -1;
    case DoubleSphereCompatProfile::BASALT_D0:
      return 0;
    case DoubleSphereCompatProfile::BASALT_D4:
      return 4;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const EucmCompatProfile profile)
{
  switch (profile)
  {
    case EucmCompatProfile::CANONICAL:
      return -1;
    case EucmCompatProfile::KALIBR_D0:
      return 0;
    case EucmCompatProfile::KALIBR_D4:
      return 4;
    default:
      return -2;
  }
}

inline void fillProjectionFromIntrinsics(
  const IntrinsicsModel &intrinsics, std::array<double, 9> &camera_matrix
)
{
  camera_matrix = {
    static_cast<double>(intrinsics.fx),
    static_cast<double>(intrinsics.skew),
    static_cast<double>(intrinsics.cx),
    0.0,
    static_cast<double>(intrinsics.fy),
    static_cast<double>(intrinsics.cy),
    0.0,
    0.0,
    1.0};
}

inline void fillProjectionMatrix3x4(
  const IntrinsicsModel &intrinsics, std::array<double, 12> &projection
)
{
  projection = {
    static_cast<double>(intrinsics.fx),
    static_cast<double>(intrinsics.skew),
    static_cast<double>(intrinsics.cx),
    0.0,
    0.0,
    static_cast<double>(intrinsics.fy),
    static_cast<double>(intrinsics.cy),
    0.0,
    0.0,
    0.0,
    1.0,
    0.0};
}

inline std::string pinholeDistortionModelByCount(const std::size_t count)
{
  return count >= 8U ? "rational_polynomial" : "plumb_bob";
}

inline std::string pinholeDistortionModelByType(const DistortionModelType model_type)
{
  switch (model_type)
  {
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
      return "plumb_bob";
    case DistortionModelType::RATIONAL8:
      return "rational_polynomial";
    case DistortionModelType::THIN_PRISM12:
      return "thin_prism_fisheye";
    case DistortionModelType::TILTED14:
      return "tilted";
    case DistortionModelType::NONE:
      return "plumb_bob";
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return "plumb_bob";
}

inline constexpr std::size_t kFisheyeCanonicalMaxD = 8U;
inline constexpr std::size_t kFisheyeLegacyMaxD = 4U;

inline std::string fisheyeDistortionModelString(const DistortionModelType model_type)
{
  switch (model_type)
  {
    case DistortionModelType::EQUIDISTANT:
      return "equidistant";
    case DistortionModelType::OPENCV_FISHEYE4:
      return "equidistant";
    case DistortionModelType::KB4:
      return "kb4";
    case DistortionModelType::KB8:
      return "kb8";
    case DistortionModelType::EQUISOLID:
      return "equisolid";
    case DistortionModelType::STEREOGRAPHIC:
      return "stereographic";
    case DistortionModelType::ORTHOGRAPHIC:
      return "orthographic";
    case DistortionModelType::NONE:
      return "equidistant";
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return "equidistant";
}

inline StatusCode makeAndValidateProjectionModel(
  const CameraInfo &camera_info, const ProjectionModelType projection_type, CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  model_out = makeCameraModel(camera_info);
  const StatusCode validation_status = validateCameraModel(model_out);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model_out.projection.type != projection_type)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

inline StatusCode validatePinholeModelForExport(const CameraModel &model)
{
  const StatusCode validation_status = validateCameraModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::PINHOLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (model.distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

inline StatusCode validateFisheyeModelForExport(const CameraModel &model)
{
  const StatusCode validation_status = validateCameraModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (model.distortion.space == DistortionSpace::PLANE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

inline StatusCode validateOmniModelForExport(const CameraModel &model)
{
  const StatusCode validation_status = validateCameraModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (model.distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

inline StatusCode validateDoubleSphereModelForExport(const CameraModel &model)
{
  const StatusCode validation_status = validateCameraModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (model.distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

inline StatusCode validateEucmModelForExport(const CameraModel &model)
{
  const StatusCode validation_status = validateCameraModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::EUCM)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (model.distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

}  // namespace
}  // namespace camxiom

#endif  // CAMXIOM__COMPAT__INTERNAL_HPP
