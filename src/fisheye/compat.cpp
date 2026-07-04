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

#include "compat/internal.hpp"

namespace camxiom
{

StatusCode importFisheyeModel(
  const FisheyeExternalModel &external_model, const FisheyeCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(external_model.K) || !isFiniteVector(external_model.D))
  {
    return StatusCode::INVALID_INPUT;
  }
  const std::size_t max_d =
    (profile == FisheyeCompatProfile::CANONICAL) ? kFisheyeCanonicalMaxD : kFisheyeLegacyMaxD;
  if (external_model.D.size() > max_d)
  {
    return StatusCode::INVALID_INPUT;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }
  if (expected_count >= 0 && external_model.D.size() != static_cast<std::size_t>(expected_count))
  {
    return StatusCode::INVALID_INPUT;
  }

  CameraInfo camera_info;
  if (profile == FisheyeCompatProfile::CANONICAL && external_model.distortion_type != DistortionModelType::UNKNOWN)
  {
    camera_info.distortion_model = fisheyeDistortionModelString(external_model.distortion_type);
  }
  else
  {
    camera_info.distortion_model = "equidistant";
  }
  camera_info.k = external_model.K;
  camera_info.d = external_model.D;

  if (profile == FisheyeCompatProfile::CANONICAL &&
      (external_model.distortion_type == DistortionModelType::EQUIDISTANT ||
       external_model.distortion_type == DistortionModelType::NONE))
  {
    if (!external_model.D.empty())
    {
      return StatusCode::INVALID_INPUT;
    }
    camera_info.distortion_model = "equidistant";
    camera_info.d = {0.0, 0.0, 0.0, 0.0};
    const StatusCode status =
      makeAndValidateProjectionModel(camera_info, ProjectionModelType::FISHEYE_THETA, model_out);
    if (status != StatusCode::OK)
    {
      return status;
    }
    model_out.distortion.type = external_model.distortion_type;
    model_out.distortion.space = (external_model.distortion_type == DistortionModelType::NONE)
                                   ? DistortionSpace::NONE
                                   : DistortionSpace::ANGLE;
    model_out.distortion.count = 0U;
    return validateCameraModel(model_out);
  }

  return makeAndValidateProjectionModel(camera_info, ProjectionModelType::FISHEYE_THETA, model_out);
}

StatusCode exportFisheyeModel(
  const CameraModel &model, const FisheyeCompatProfile profile,
  FisheyeExternalModel &external_model_out
)
{
  const StatusCode model_status = validateFisheyeModelForExport(model);
  if (model_status != StatusCode::OK)
  {
    return model_status;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }

  fillProjectionFromIntrinsics(model.intrinsics, external_model_out.K);
  external_model_out.distortion_type = model.distortion.type;
  std::size_t export_count = model.distortion.count;
  if (expected_count >= 0)
  {
    export_count = static_cast<std::size_t>(expected_count);
  }
  const std::size_t max_d =
    (profile == FisheyeCompatProfile::CANONICAL) ? kFisheyeCanonicalMaxD : kFisheyeLegacyMaxD;
  if (export_count > max_d)
  {
    return StatusCode::INVALID_MODEL;
  }

  external_model_out.D.assign(export_count, 0.0);
  const std::size_t copy_count = std::min<std::size_t>(model.distortion.count, export_count);
  for (std::size_t index = 0; index < copy_count; ++index)
  {
    external_model_out.D[index] = static_cast<double>(model.distortion.coeffs[index]);
  }
  return StatusCode::OK;
}

}  // namespace camxiom
