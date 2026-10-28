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

// CameraModel -> ROS CameraInfo POD export. The packing (distortion_model
// name, d, identity r and [K | 0] p) lives in the ROS-free core; the
// ROS layer delegates here and only copies fields into sensor_msgs.

#include "camxiom/compat.hpp"
#include "compat/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace camxiom
{

namespace
{

template <typename T>
bool coefficientsAreZero(
  const DistortionModelT<T> &distortion, const std::size_t first, const std::size_t last
)
{
  for (std::size_t index = first; index < last; ++index)
  {
    if (distortion.coeffs[index] != T(0))
    {
      return false;
    }
  }
  return true;
}

// The ROS distortion_model name and coefficient count that hold the model's
// lens exactly, or an empty name when ROS has none.
template <typename T>
std::pair<const char *, std::size_t> rosDistortionModel(const CameraModelT<T> &model)
{
  const DistortionModelT<T> &distortion = model.distortion;
  if (model.projection.type == ProjectionModelType::PINHOLE)
  {
    switch (distortion.type)
    {
      case DistortionModelType::NONE:
      case DistortionModelType::RADTAN4:
      case DistortionModelType::RADTAN5:
        return {"plumb_bob", 5U};
      case DistortionModelType::RATIONAL8:
        return {"rational_polynomial", 8U};
      case DistortionModelType::THIN_PRISM12:
      case DistortionModelType::TILTED14:
        // The prism and tilt terms follow the eight rational ones.
        if (coefficientsAreZero(distortion, 8U, distortion.count))
        {
          return {"rational_polynomial", 8U};
        }
        break;
      default:
        break;
    }
  }
  else if (model.projection.type == ProjectionModelType::EQUIDISTANT)
  {
    switch (distortion.type)
    {
      case DistortionModelType::NONE:
      case DistortionModelType::KB4:
        return {"equidistant", 4U};
      case DistortionModelType::KB8:
        if (coefficientsAreZero(distortion, 4U, distortion.count))
        {
          return {"equidistant", 4U};
        }
        break;
      default:
        break;
    }
  }
  return {"", 0U};
}

template <typename T>
StatusCode exportRosCameraInfoImpl(const CameraModelT<T> &model, CameraInfo &camera_info_out)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  // ROS defines K with zero off-diagonal terms, and its receivers (e.g.
  // image_geometry) project without skew, so a skewed camera is not a lens
  // ROS can hold.
  const auto [name, count] = rosDistortionModel(model);
  if (name[0] == '\0' || model.intrinsics.skew != T(0))
  {
    return StatusCode::INVALID_MODEL;
  }

  camera_info_out.distortion_model = name;
  camera_info_out.d.assign(count, 0.0);
  const std::size_t copy_count = std::min<std::size_t>(model.distortion.count, count);
  for (std::size_t index = 0; index < copy_count; ++index)
  {
    camera_info_out.d[index] = static_cast<double>(model.distortion.coeffs[index]);
  }
  fillProjectionFromIntrinsics(model.intrinsics, camera_info_out.k);
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);
  return StatusCode::OK;
}

}  // namespace

StatusCode exportCameraInfo(const CameraModel &model, CameraInfo &camera_info_out)
{
  return exportRosCameraInfoImpl(model, camera_info_out);
}

StatusCode exportCameraInfo64(const CameraModel64 &model, CameraInfo &camera_info_out)
{
  return exportRosCameraInfoImpl(model, camera_info_out);
}

}  // namespace camxiom
