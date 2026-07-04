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

#include "camxiom/projection.hpp"
#include "detail/internal.hpp"
#include "detail/projection_models.hpp"

namespace camxiom
{

PixelResult rayToPixel(const CameraModel &model, const Eigen::Vector3f &ray_direction)
{
  // Query-tier guard (see detail/internal.hpp): identical accept/reject
  // across the scalar / batch / SIMD / Jacobian query paths. The public
  // validateCameraModel() additionally certifies the fisheye monotone cap.
  const StatusCode validation_status = detail::validateCameraModelQuery(model);
  if (validation_status != StatusCode::OK)
  {
    return detail::invalidPixelResult(validation_status);
  }

  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::rayToPixel(model, ray_direction);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::rayToPixel(model, ray_direction);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::rayToPixel(model, ray_direction);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::rayToPixel(model, ray_direction);
    case ProjectionModelType::EUCM:
      return eucm::rayToPixel(model, ray_direction);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return detail::invalidPixelResult(StatusCode::INVALID_MODEL);
}

PixelResult rayToPixel(
  const CameraModel &model, const float x_direction, const float y_direction,
  const float z_direction
)
{
  return rayToPixel(model, Eigen::Vector3f(x_direction, y_direction, z_direction));
}

PixelResult rayToPixel(const CameraModel &model, const Ray3 &ray)
{
  return rayToPixel(model, ray.direction);
}

RayResult pixelToRay(
  const CameraModel &model, const Pixel2 &pixel, const SolverOptions &solver_options
)
{
  // Query-tier guard (see detail/internal.hpp): identical accept/reject
  // across the scalar / batch / SIMD / Jacobian query paths. The public
  // validateCameraModel() additionally certifies the fisheye monotone cap.
  const StatusCode validation_status = detail::validateCameraModelQuery(model);
  if (validation_status != StatusCode::OK)
  {
    return detail::invalidRayResult(validation_status);
  }

  if (!detail::isFinite2(pixel.u, pixel.v))
  {
    return detail::invalidRayResult(StatusCode::INVALID_INPUT);
  }

  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::pixelToRay(model, pixel, solver_options);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::pixelToRay(model, pixel, solver_options);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::pixelToRay(model, pixel, solver_options);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::pixelToRay(model, pixel, solver_options);
    case ProjectionModelType::EUCM:
      return eucm::pixelToRay(model, pixel, solver_options);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return detail::invalidRayResult(StatusCode::INVALID_MODEL);
}

RayResult pixelToRay(
  const CameraModel &model, const float u, const float v, const SolverOptions &solver_options
)
{
  return pixelToRay(model, Pixel2{u, v}, solver_options);
}

bool isRayProjectable(const CameraModel &model, const Eigen::Vector3f &ray_direction)
{
  return rayToPixel(model, ray_direction).status == StatusCode::OK;
}

bool isRayProjectable(
  const CameraModel &model, const Eigen::Vector3f &ray_direction, const int image_width,
  const int image_height
)
{
  const PixelResult px = rayToPixel(model, ray_direction);
  return px.status == StatusCode::OK && px.pixel.u >= 0.0f && px.pixel.v >= 0.0f &&
         px.pixel.u < static_cast<float>(image_width) &&
         px.pixel.v < static_cast<float>(image_height);
}

}  // namespace camxiom
