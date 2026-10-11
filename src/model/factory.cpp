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
#include "model/internal.hpp"

namespace camxiom
{

void updateThetaMax(CameraModel &model)
{
  if (model.projection.type == ProjectionModelType::OMNIDIRECTIONAL ||
      model.projection.type == ProjectionModelType::DOUBLE_SPHERE ||
      model.projection.type == ProjectionModelType::EUCM)
  {
    model.projection.theta_max = detail::kPi;
  }
  else if (model.projection.type == ProjectionModelType::FISHEYE_THETA)
  {
    model.projection.theta_max = detail::defaultFisheyeThetaMax(model.distortion);
  }
  else
  {
    model.projection.theta_max = detail::kHalfPi - 1e-4f;
  }
}

void updateThetaMax64(CameraModel64 &model)
{
  using P = ProjectionModelType;
  if (model.projection.type == P::OMNIDIRECTIONAL ||
      model.projection.type == P::DOUBLE_SPHERE || model.projection.type == P::EUCM)
    model.projection.theta_max = constants::kPi;
  else if (model.projection.type == P::FISHEYE_THETA)
  {
    if (detail::isPolynomialFisheyeDistortion(model.distortion.type))
      model.projection.theta_max = detail::estimateSafeThetaMaxForPolynomialFisheyeT(model.distortion);
    else if (model.distortion.type == DistortionModelType::ORTHOGRAPHIC)
      model.projection.theta_max = constants::kHalfPi - 1e-4;
    else
      model.projection.theta_max = constants::kPi - 1e-4;
  }
  else
    model.projection.theta_max = constants::kHalfPi - 1e-4;
}

}  // namespace camxiom
