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

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "model/internal.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace camxiom
{

namespace
{
// The ROS distortion models: the name gives the projection law and the
// correction family, the number of coefficients the correction.
std::pair<ProjectionModelType, DistortionModelType> rosModel(
  const std::string &name, const std::size_t count
)
{
  if (name == "plumb_bob")
  {
    switch (count)
    {
      case 0U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::NONE};
      case 4U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::RADTAN4};
      case 5U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::RADTAN5};
      default:
        break;
    }
  }
  else if (name == "rational_polynomial")
  {
    switch (count)
    {
      case 8U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::RATIONAL8};
      case 12U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::THIN_PRISM12};
      case 14U:
        return {ProjectionModelType::PINHOLE, DistortionModelType::TILTED14};
      default:
        break;
    }
  }
  else if (name == "equidistant" && count == 4U)
  {
    return {ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4};
  }
  return {ProjectionModelType::UNKNOWN, DistortionModelType::UNKNOWN};
}

template <typename T>
CameraModelT<T> makeCameraModelImpl(const CameraInfo &camera_info)
{
  CameraModelT<T> model;
  model.intrinsics.fx = static_cast<T>(camera_info.k[0]);
  model.intrinsics.skew = static_cast<T>(camera_info.k[1]);
  model.intrinsics.cx = static_cast<T>(camera_info.k[2]);
  model.intrinsics.fy = static_cast<T>(camera_info.k[4]);
  model.intrinsics.cy = static_cast<T>(camera_info.k[5]);

  const auto [law, correction] = rosModel(camera_info.distortion_model, camera_info.d.size());
  if (law == ProjectionModelType::UNKNOWN)
  {
    // projection.type stays UNKNOWN, which validation rejects.
    return model;
  }

  model.distortion.type = correction;
  model.distortion.count = static_cast<std::uint8_t>(camera_info.d.size());
  for (std::size_t index = 0; index < camera_info.d.size(); ++index)
  {
    model.distortion.coeffs[index] = static_cast<T>(camera_info.d[index]);
  }
  detail::rebuildDistortionAuxState(model.distortion);
  model.projection.type = law;

  if constexpr (std::is_same_v<T, float>) updateThetaMax(model);
  else
    updateThetaMax64(model);
  return model;
}

}  // namespace

CameraModel makeCameraModel(const CameraInfo &info) { return makeCameraModelImpl<float>(info); }

CameraModel64 makeCameraModel64(const CameraInfo &info) { return makeCameraModelImpl<double>(info); }

}  // namespace camxiom
