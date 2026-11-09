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

#ifndef CAMXIOM__CALIB__SUPPORTED_LAWS_HPP
#define CAMXIOM__CALIB__SUPPORTED_LAWS_HPP

#include "camxiom/types.hpp"

namespace camxiom::detail
{

/// True for the projection laws outside the calibration layer -- intrinsic
/// calibration, model conversion and pose estimation (DLT and the PnP
/// solver), whose cost functions are written for the other laws. The
/// lat-long projections are outside it by scope, not by geometry: their ray
/// Jacobian exists and a pose from their rays is well defined. UNKNOWN is
/// not a law; each entry point's own validation rejects it.
///
/// Every entry point of the layer asks this one question. The switch has no
/// default on purpose: a new law does not compile here (-Werror=switch) until
/// it is placed on one side, instead of passing a guard that lists the laws
/// to refuse.
constexpr bool isOutsideCalibrationLayer(const ProjectionModelType type) noexcept
{
  switch (type)
  {
    case ProjectionModelType::EQUIRECTANGULAR:
    case ProjectionModelType::CYLINDRICAL:
      return true;
    case ProjectionModelType::PINHOLE:
    case ProjectionModelType::EQUIDISTANT:
    case ProjectionModelType::EQUISOLID:
    case ProjectionModelType::STEREOGRAPHIC:
    case ProjectionModelType::ORTHOGRAPHIC:
    case ProjectionModelType::OMNIDIRECTIONAL:
    case ProjectionModelType::DOUBLE_SPHERE:
    case ProjectionModelType::EUCM:
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return false;
}

}  // namespace camxiom::detail

#endif  // CAMXIOM__CALIB__SUPPORTED_LAWS_HPP
