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

#include <cmath>

namespace camxiom
{
namespace
{

FisheyeDomain64 deriveFisheyeDomain64(const CameraModel64 &model)
{
  const DistortionModel64 &distortion = model.distortion;
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
  else if (model.projection.type == ProjectionModelType::ORTHOGRAPHIC)
  {
    result.theta_limit = constants::kHalfPi;
    result.safe_theta_max = constants::kHalfPi - 1e-4;
  }
  return result;
}

// A lat-long model whose image spans the normalized coordinates
// [x_min, x_max] x [y_min, y_max] edge to edge: pixel -0.5 maps to the low
// end and W - 0.5 (H - 0.5) to the high end.
StatusCode makeLatLongModel(
  const ProjectionModelType type, const ImageSize image_size, const double x_min,
  const double x_max, const double y_min, const double y_max, CameraModel &model_out
)
{
  model_out = CameraModel{};
  const double fx = static_cast<double>(image_size.width) / (x_max - x_min);
  const double fy = static_cast<double>(image_size.height) / (y_max - y_min);

  CameraModel model;
  model.projection.type = type;
  model.projection.theta_max = constants::kPiF;
  model.intrinsics.fx = static_cast<float>(fx);
  model.intrinsics.fy = static_cast<float>(fy);
  model.intrinsics.cx = static_cast<float>(-0.5 - x_min * fx);
  model.intrinsics.cy = static_cast<float>(-0.5 - y_min * fy);
  if (detail_impl::validateCameraModelImpl<float>(model) != StatusCode::OK)
  {
    return StatusCode::INVALID_INPUT;
  }
  model_out = model;
  return StatusCode::OK;
}

bool validLongitudeRange(const double lon_min, const double lon_max)
{
  return std::isfinite(lon_min) && std::isfinite(lon_max) && lon_min >= -constants::kPi &&
         lon_min < lon_max && lon_max <= constants::kPi;
}

}  // namespace

StatusCode makeEquirectangularModel(const ImageSize image_size, CameraModel &model_out)
{
  return makeEquirectangularModel(
    image_size, -constants::kPi, constants::kPi, -constants::kHalfPi, constants::kHalfPi, model_out
  );
}

StatusCode makeEquirectangularModel(
  const ImageSize image_size, const double lon_min, const double lon_max, const double lat_min,
  const double lat_max, CameraModel &model_out
)
{
  const bool valid_latitudes = std::isfinite(lat_min) && std::isfinite(lat_max) &&
                               lat_min >= -constants::kHalfPi && lat_min < lat_max &&
                               lat_max <= constants::kHalfPi;
  if (!image_size.isValid() || !validLongitudeRange(lon_min, lon_max) || !valid_latitudes)
  {
    model_out = CameraModel{};
    return StatusCode::INVALID_INPUT;
  }
  return makeLatLongModel(
    ProjectionModelType::EQUIRECTANGULAR, image_size, lon_min, lon_max, lat_min, lat_max, model_out
  );
}

StatusCode makeCylindricalModel(
  const ImageSize image_size, const double lon_min, const double lon_max, const double lat_min,
  const double lat_max, CameraModel &model_out
)
{
  const bool valid_latitudes = std::isfinite(lat_min) && std::isfinite(lat_max) &&
                               lat_min > -constants::kHalfPi && lat_min < lat_max &&
                               lat_max < constants::kHalfPi;
  if (!image_size.isValid() || !validLongitudeRange(lon_min, lon_max) || !valid_latitudes)
  {
    model_out = CameraModel{};
    return StatusCode::INVALID_INPUT;
  }
  return makeLatLongModel(
    ProjectionModelType::CYLINDRICAL, image_size, lon_min, lon_max, std::tan(lat_min),
    std::tan(lat_max), model_out
  );
}

FisheyeDomain64 getFisheyeDomain64(const CameraModel64 &model)
{
  FisheyeDomain64 result;
  if (!isFisheyeProjection(model.projection.type))
  {
    result.status = StatusCode::INVALID_MODEL;
    return result;
  }

  const auto derived = deriveFisheyeDomain64(model);
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
      model.projection.type == ProjectionModelType::EUCM ||
      isLatLongProjection(model.projection.type))
  {
    model.projection.theta_max = detail::kPi;
  }
  else if (isFisheyeProjection(model.projection.type))
  {
    model.projection.theta_max = detail::defaultFisheyeThetaMax(model);
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
      model.projection.type == P::DOUBLE_SPHERE || model.projection.type == P::EUCM ||
      isLatLongProjection(model.projection.type))
    model.projection.theta_max = constants::kPi;
  else if (isFisheyeProjection(model.projection.type))
  {
    model.projection.theta_max = deriveFisheyeDomain64(model).safe_theta_max;
  }
  else
    model.projection.theta_max = constants::kHalfPi - 1e-4;
}

}  // namespace camxiom
