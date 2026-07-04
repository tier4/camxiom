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

StatusCode importEucmModel(
  const EucmExternalModel &external_model, const EucmCompatProfile profile, CameraModel &model_out
)
{
  if (!isFiniteArray9(external_model.K) || !isFiniteVector(external_model.D) ||
      !isFiniteScalar(external_model.alpha) || !isFiniteScalar(external_model.beta))
  {
    return StatusCode::INVALID_INPUT;
  }
  if (external_model.D.size() > 14U)
  {
    return StatusCode::INVALID_INPUT;
  }
  if (external_model.beta <= 0.0)
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
  camera_info.distortion_model = "eucm";
  camera_info.k = external_model.K;
  camera_info.d.clear();
  camera_info.d.reserve(2U + external_model.D.size());
  camera_info.d.push_back(external_model.alpha);
  camera_info.d.push_back(external_model.beta);
  camera_info.d.insert(camera_info.d.end(), external_model.D.begin(), external_model.D.end());
  return makeAndValidateProjectionModel(camera_info, ProjectionModelType::EUCM, model_out);
}

StatusCode exportEucmModel(
  const CameraModel &model, const EucmCompatProfile profile, EucmExternalModel &external_model_out
)
{
  const StatusCode model_status = validateEucmModelForExport(model);
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
  external_model_out.alpha = static_cast<double>(model.projection.alpha);
  external_model_out.beta = static_cast<double>(model.projection.beta);

  std::size_t export_count = model.distortion.count;
  if (expected_count >= 0)
  {
    export_count = static_cast<std::size_t>(expected_count);
  }
  if (export_count > 14U)
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
