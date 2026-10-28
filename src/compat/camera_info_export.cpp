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

// CameraModel -> CameraInfo POD export (the canonical CameraInfo packing).
//
// Moved here from the optional ROS layer (src/compat/ros.cpp) so the packing —
// per-family distortion_model string, d-vector layout (xi/alpha prefixes for
// omnidirectional / double sphere / EUCM), identity r and [K | 0] p — lives in
// the ROS-free core. The ROS layer now delegates to these functions and
// only copies fields into sensor_msgs.

#include "camxiom/compat.hpp"
#include "compat/internal.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace camxiom
{

template <typename T>
StatusCode exportPinholeCameraInfoImpl(
  const CameraModelT<T> &model, const PinholeCompatProfile profile, CameraInfo &camera_info_out
)
{
  PinholeExternalModel external_model;
  const StatusCode status = [&]() {
    if constexpr (std::is_same_v<T, float>) return exportPinholeModel(model, profile, external_model);
    else return exportPinholeModel64(model, profile, external_model);
  }();
  if (status != StatusCode::OK)
  {
    return status;
  }

  camera_info_out.k = external_model.K;
  camera_info_out.d = external_model.D;
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);

  if (profile == PinholeCompatProfile::CANONICAL || profile == PinholeCompatProfile::ROS_CAMERA_INFO_RAW)
  {
    camera_info_out.distortion_model = pinholeDistortionModelByType(model.distortion.type);
  }
  else
  {
    camera_info_out.distortion_model = pinholeDistortionModelByCount(camera_info_out.d.size());
  }
  return StatusCode::OK;
}

template <typename T>
StatusCode exportFisheyeCameraInfoImpl(
  const CameraModelT<T> &model, const FisheyeCompatProfile profile, CameraInfo &camera_info_out
)
{
  FisheyeExternalModel external_model;
  const StatusCode status = [&]() {
    if constexpr (std::is_same_v<T, float>) return exportFisheyeModel(model, profile, external_model);
    else return exportFisheyeModel64(model, profile, external_model);
  }();
  if (status != StatusCode::OK)
  {
    return status;
  }

  if (profile == FisheyeCompatProfile::CANONICAL)
  {
    camera_info_out.distortion_model =
      fisheyeDistortionModelString(external_model.projection_type, external_model.distortion_type);
  }
  else
  {
    camera_info_out.distortion_model = "equidistant";
  }
  camera_info_out.k = external_model.K;
  camera_info_out.d = external_model.D;
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);
  return StatusCode::OK;
}

template <typename T>
StatusCode exportOmnidirectionalCameraInfoImpl(
  const CameraModelT<T> &model, const OmnidirectionalCompatProfile profile, CameraInfo &camera_info_out
)
{
  OmnidirectionalExternalModel external_model;
  const StatusCode status = [&]() {
    if constexpr (std::is_same_v<T, float>) return exportOmnidirectionalModel(model, profile, external_model);
    else return exportOmnidirectionalModel64(model, profile, external_model);
  }();
  if (status != StatusCode::OK)
  {
    return status;
  }

  camera_info_out.distortion_model = "omnidirectional";
  camera_info_out.k = external_model.K;
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);

  if (profile == OmnidirectionalCompatProfile::CANONICAL)
  {
    camera_info_out.d.assign(1U + external_model.D.size(), 0.0);
    camera_info_out.d[0] = external_model.xi;
    for (std::size_t index = 0; index < external_model.D.size(); ++index)
    {
      camera_info_out.d[index + 1U] = external_model.D[index];
    }
  }
  else
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
StatusCode exportDoubleSphereCameraInfoImpl(
  const CameraModelT<T> &model, const DoubleSphereCompatProfile profile, CameraInfo &camera_info_out
)
{
  DoubleSphereExternalModel external_model;
  const StatusCode status = [&]() {
    if constexpr (std::is_same_v<T, float>) return exportDoubleSphereModel(model, profile, external_model);
    else return exportDoubleSphereModel64(model, profile, external_model);
  }();
  if (status != StatusCode::OK)
  {
    return status;
  }

  camera_info_out.distortion_model = "double_sphere";
  camera_info_out.k = external_model.K;
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);

  camera_info_out.d.clear();
  camera_info_out.d.reserve(2U + external_model.D.size());
  camera_info_out.d.push_back(external_model.xi);
  camera_info_out.d.push_back(external_model.alpha);
  camera_info_out.d.insert(
    camera_info_out.d.end(), external_model.D.begin(), external_model.D.end()
  );
  return StatusCode::OK;
}

template <typename T>
StatusCode exportEucmCameraInfoImpl(
  const CameraModelT<T> &model, const EucmCompatProfile profile, CameraInfo &camera_info_out
)
{
  EucmExternalModel external_model;
  const StatusCode status = [&]() {
    if constexpr (std::is_same_v<T, float>) return exportEucmModel(model, profile, external_model);
    else return exportEucmModel64(model, profile, external_model);
  }();
  if (status != StatusCode::OK)
  {
    return status;
  }

  camera_info_out.distortion_model = "eucm";
  camera_info_out.k = external_model.K;
  camera_info_out.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  fillProjectionMatrix3x4(model.intrinsics, camera_info_out.p);

  camera_info_out.d.clear();
  camera_info_out.d.reserve(2U + external_model.D.size());
  camera_info_out.d.push_back(external_model.alpha);
  camera_info_out.d.push_back(external_model.beta);
  camera_info_out.d.insert(
    camera_info_out.d.end(), external_model.D.begin(), external_model.D.end()
  );
  return StatusCode::OK;
}

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

StatusCode exportPinholeCameraInfo(
  const CameraModel &model, PinholeCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportPinholeCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportPinholeCameraInfo64(
  const CameraModel64 &model, PinholeCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportPinholeCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportFisheyeCameraInfo(
  const CameraModel &model, FisheyeCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportFisheyeCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportFisheyeCameraInfo64(
  const CameraModel64 &model, FisheyeCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportFisheyeCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportOmnidirectionalCameraInfo(
  const CameraModel &model, OmnidirectionalCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportOmnidirectionalCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportOmnidirectionalCameraInfo64(
  const CameraModel64 &model, OmnidirectionalCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportOmnidirectionalCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportDoubleSphereCameraInfo(
  const CameraModel &model, DoubleSphereCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportDoubleSphereCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportDoubleSphereCameraInfo64(
  const CameraModel64 &model, DoubleSphereCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportDoubleSphereCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportEucmCameraInfo(
  const CameraModel &model, EucmCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportEucmCameraInfoImpl(model, profile, camera_info_out);
}

StatusCode exportEucmCameraInfo64(
  const CameraModel64 &model, EucmCompatProfile profile, CameraInfo &camera_info_out)
{
  return exportEucmCameraInfoImpl(model, profile, camera_info_out);
}

}  // namespace camxiom
