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

}  // namespace camxiom
