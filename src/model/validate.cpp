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

#include "camxiom/model.hpp"
#include "detail/internal.hpp"
#include "model/validate_impl.hpp"

namespace camxiom
{

namespace detail
{

StatusCode validateCameraModelQuery(const CameraModel &model)
{
  return detail_impl::validateCameraModelQueryImpl<float>(model);
}

}  // namespace detail

StatusCode validateCameraModel(const CameraModel &model)
{
  return detail_impl::validateCameraModelImpl<float>(model);
}

CameraModel makeDistortionFree(const CameraModel &model)
{
  CameraModel result = model;
  result.distortion.type = DistortionModelType::NONE;
  result.distortion.space = DistortionSpace::NONE;
  result.distortion.coeffs.fill(0.0f);
  result.distortion.count = 0U;
  result.distortion.is_rational = false;
  result.distortion.has_thin_prism = false;
  result.distortion.has_tilt = false;
  return result;
}

}  // namespace camxiom
