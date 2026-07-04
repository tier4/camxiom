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
// the dependency-free core. The ROS layer now delegates to these functions and
// only copies fields into sensor_msgs.

#include "camxiom/compat.hpp"
#include "compat/internal.hpp"

namespace camxiom
{

StatusCode exportPinholeCameraInfo(
  const CameraModel &model, const PinholeCompatProfile profile, CameraInfo &camera_info_out
)
{
  PinholeExternalModel external_model;
  const StatusCode status = exportPinholeModel(model, profile, external_model);
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

StatusCode exportFisheyeCameraInfo(
  const CameraModel &model, const FisheyeCompatProfile profile, CameraInfo &camera_info_out
)
{
  FisheyeExternalModel external_model;
  const StatusCode status = exportFisheyeModel(model, profile, external_model);
  if (status != StatusCode::OK)
  {
    return status;
  }

  if (profile == FisheyeCompatProfile::CANONICAL)
  {
    camera_info_out.distortion_model = fisheyeDistortionModelString(external_model.distortion_type);
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

StatusCode exportOmnidirectionalCameraInfo(
  const CameraModel &model, const OmnidirectionalCompatProfile profile, CameraInfo &camera_info_out
)
{
  OmnidirectionalExternalModel external_model;
  const StatusCode status = exportOmnidirectionalModel(model, profile, external_model);
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

StatusCode exportDoubleSphereCameraInfo(
  const CameraModel &model, const DoubleSphereCompatProfile profile, CameraInfo &camera_info_out
)
{
  DoubleSphereExternalModel external_model;
  const StatusCode status = exportDoubleSphereModel(model, profile, external_model);
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

StatusCode exportEucmCameraInfo(
  const CameraModel &model, const EucmCompatProfile profile, CameraInfo &camera_info_out
)
{
  EucmExternalModel external_model;
  const StatusCode status = exportEucmModel(model, profile, external_model);
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

}  // namespace camxiom
