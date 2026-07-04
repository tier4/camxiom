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

#include <algorithm>
#include <limits>
#include <string>

namespace camxiom
{

CameraModel makeCameraModel(const CameraInfo &camera_info)
{
  CameraModel model;
  const float invalid_parameter = std::numeric_limits<float>::quiet_NaN();
  model.intrinsics.fx = static_cast<float>(camera_info.k[0]);
  model.intrinsics.skew = static_cast<float>(camera_info.k[1]);
  model.intrinsics.cx = static_cast<float>(camera_info.k[2]);
  model.intrinsics.fy = static_cast<float>(camera_info.k[4]);
  model.intrinsics.cy = static_cast<float>(camera_info.k[5]);
  model.projection.xi = 0.0f;
  model.projection.alpha = 0.0f;
  detail::setIdentity3x3(model.distortion.tilt_matrix);
  detail::setIdentity3x3(model.distortion.inv_tilt_matrix);

  DistortionModelType parsed_type = parseDistortionModelType(camera_info.distortion_model);
  const bool is_omni = parsed_type == DistortionModelType::OMNIDIRECTIONAL;

  const std::string model_lower = detail::toLowerCopy(camera_info.distortion_model);
  const bool is_ds = (model_lower == "double_sphere" || model_lower == "ds");
  const bool is_eucm = (model_lower == "eucm");

  if (parsed_type == DistortionModelType::UNKNOWN && !is_ds && !is_eucm)
  {
    if (camera_info.distortion_model.empty())
    {
      parsed_type = camera_info.d.empty()
                      ? DistortionModelType::NONE
                      : detail::classifyPlaneDistortionByCount(camera_info.d.size());
    }
  }

  std::size_t distortion_offset = 0U;
  if (is_omni)
  {
    if (!camera_info.d.empty())
    {
      model.projection.xi = static_cast<float>(camera_info.d[0]);
      distortion_offset = 1U;
    }
    const std::size_t remaining =
      camera_info.d.size() > distortion_offset ? (camera_info.d.size() - distortion_offset) : 0U;
    parsed_type = detail::classifyPlaneDistortionByCount(remaining);
  }

  if (is_ds)
  {
    if (!camera_info.d.empty())
    {
      model.projection.xi = static_cast<float>(camera_info.d[0]);
    }
    if (camera_info.d.size() >= 2U)
    {
      model.projection.alpha = static_cast<float>(camera_info.d[1]);
      distortion_offset = 2U;
    }
    else
    {
      // double_sphere requires xi + alpha. Keep projection type but force invalid model.
      model.projection.alpha = invalid_parameter;
      distortion_offset = camera_info.d.size();
    }
    const std::size_t remaining =
      camera_info.d.size() > distortion_offset ? (camera_info.d.size() - distortion_offset) : 0U;
    parsed_type = detail::classifyPlaneDistortionByCount(remaining);
  }

  if (is_eucm)
  {
    if (!camera_info.d.empty())
    {
      model.projection.alpha = static_cast<float>(camera_info.d[0]);
    }
    if (camera_info.d.size() >= 2U)
    {
      model.projection.beta = static_cast<float>(camera_info.d[1]);
      distortion_offset = 2U;
    }
    else
    {
      // EUCM requires alpha + beta. Keep projection type but force invalid model.
      model.projection.beta = invalid_parameter;
      distortion_offset = camera_info.d.size();
    }
    const std::size_t remaining =
      camera_info.d.size() > distortion_offset ? (camera_info.d.size() - distortion_offset) : 0U;
    parsed_type = detail::classifyPlaneDistortionByCount(remaining);
  }

  if (!is_omni && !is_ds && !is_eucm && (parsed_type == DistortionModelType::RADTAN5 || parsed_type == DistortionModelType::RATIONAL8 || parsed_type == DistortionModelType::THIN_PRISM12 || parsed_type == DistortionModelType::TILTED14))
  {
    parsed_type = detail::classifyPlaneDistortionByCount(camera_info.d.size());
  }

  model.distortion.type = parsed_type;
  model.distortion.space = detail::distortionSpaceFromType(parsed_type);

  const std::size_t effective_d_count =
    camera_info.d.size() > distortion_offset ? (camera_info.d.size() - distortion_offset) : 0U;
  if (effective_d_count > model.distortion.coeffs.size())
  {
    model.projection.type = ProjectionModelType::UNKNOWN;
    return model;
  }

  const std::size_t coefficient_count =
    std::min<std::size_t>(effective_d_count, model.distortion.coeffs.size());
  model.distortion.count = static_cast<std::uint8_t>(coefficient_count);
  for (std::size_t index = 0; index < coefficient_count; ++index)
  {
    model.distortion.coeffs[index] = static_cast<float>(camera_info.d[index + distortion_offset]);
  }

  detail::rebuildDistortionAuxState(model.distortion);

  if (is_ds)
  {
    model.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  }
  else if (is_eucm)
  {
    model.projection.type = ProjectionModelType::EUCM;
  }
  else if (is_omni)
  {
    model.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  }
  else
  {
    model.projection.type = chooseProjectionModelType(model.distortion.type);
  }

  // Single source for the per-family theta_max policy — this used to be an
  // inline copy of updateThetaMax's switch, the exact two-sided-copy
  // pattern behind the D47 pi-boundary bug.
  updateThetaMax(model);

  return model;
}

const char *toString(const PinholeCompatProfile profile)
{
  switch (profile)
  {
    case PinholeCompatProfile::CANONICAL:
      return "canonical";
    case PinholeCompatProfile::OPENCV_CALIB3D_D4:
      return "opencv_calib3d_d4";
    case PinholeCompatProfile::OPENCV_CALIB3D_D5:
      return "opencv_calib3d_d5";
    case PinholeCompatProfile::OPENCV_CALIB3D_D8:
      return "opencv_calib3d_d8";
    case PinholeCompatProfile::OPENCV_CALIB3D_D12:
      return "opencv_calib3d_d12";
    case PinholeCompatProfile::OPENCV_CALIB3D_D14:
      return "opencv_calib3d_d14";
    case PinholeCompatProfile::ROS_CAMERA_INFO_RAW:
      return "ros_camera_info_raw";
    default:
      return "unknown";
  }
}

const char *toString(const FisheyeCompatProfile profile)
{
  switch (profile)
  {
    case FisheyeCompatProfile::CANONICAL:
      return "canonical";
    case FisheyeCompatProfile::OPENCV_FISHEYE_D4:
      return "opencv_fisheye_d4";
    case FisheyeCompatProfile::ROS_CAMERA_INFO_EQUIDISTANT:
      return "ros_camera_info_equidistant";
    default:
      return "unknown";
  }
}

const char *toString(const OmnidirectionalCompatProfile profile)
{
  switch (profile)
  {
    case OmnidirectionalCompatProfile::CANONICAL:
      return "canonical";
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4:
      return "opencv_omnidir_d4";
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D5:
      return "opencv_omnidir_d5";
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D8:
      return "opencv_omnidir_d8";
    default:
      return "unknown";
  }
}

const char *toString(const DoubleSphereCompatProfile profile)
{
  switch (profile)
  {
    case DoubleSphereCompatProfile::CANONICAL:
      return "canonical";
    case DoubleSphereCompatProfile::BASALT_D0:
      return "basalt_d0";
    case DoubleSphereCompatProfile::BASALT_D4:
      return "basalt_d4";
    default:
      return "unknown";
  }
}

const char *toString(const EucmCompatProfile profile)
{
  switch (profile)
  {
    case EucmCompatProfile::CANONICAL:
      return "canonical";
    case EucmCompatProfile::KALIBR_D0:
      return "kalibr_d0";
    case EucmCompatProfile::KALIBR_D4:
      return "kalibr_d4";
    default:
      return "unknown";
  }
}

}  // namespace camxiom
