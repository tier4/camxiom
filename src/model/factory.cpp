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
#include "model/validate_impl.hpp"

namespace camxiom
{
namespace
{

FisheyeDomain64 deriveFisheyeDomain64(const DistortionModel64 &distortion)
{
  FisheyeDomain64 result;
  result.status = StatusCode::OK;
  result.limit = FisheyeDomainLimit::ANGULAR_LIMIT;
  result.theta_limit = constants::kPi;
  result.safe_theta_max = constants::kPi - 1e-4;
  if (detail::isPolynomialFisheyeDistortion(distortion.type))
  {
    const auto scanned = detail_impl::scanPolynomialFisheyeDomain(distortion);
    result.safe_theta_max = scanned.safe_theta_max;
    if (scanned.limit_status != StatusCode::OK)
    {
      result.theta_limit = scanned.theta_limit;
      result.limit = scanned.limit_status == StatusCode::DOMAIN_ERROR
                       ? FisheyeDomainLimit::MONOTONICITY_LIMIT
                       : FisheyeDomainLimit::NUMERIC_LIMIT;
    }
  }
  else if (distortion.type == DistortionModelType::ORTHOGRAPHIC)
  {
    result.theta_limit = constants::kHalfPi;
    result.safe_theta_max = constants::kHalfPi - 1e-4;
  }
  return result;
}

}  // namespace

FisheyeDomain64 getFisheyeDomain64(const CameraModel64 &model)
{
  FisheyeDomain64 result;
  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    result.status = StatusCode::INVALID_MODEL;
    return result;
  }

  const auto derived = deriveFisheyeDomain64(model.distortion);
  auto candidate = model;
  candidate.projection.theta_max = derived.safe_theta_max;
  // The scan already checked monotonicity. Validate the remaining model
  // invariants with the derived cap, never with the caller's declared FOV.
  result.status = detail_impl::validateCameraModelQueryImpl(candidate);
  return result.ok() ? derived : result;
}

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
    model.projection.theta_max = deriveFisheyeDomain64(model.distortion).safe_theta_max;
  }
  else
    model.projection.theta_max = constants::kHalfPi - 1e-4;
}

}  // namespace camxiom
