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

#include "camxiom/model.hpp"  // validateCameraModel64
#include "camxiom/projection64.hpp"
#include "detail/projection_models.hpp"
#include "projection64/internal.hpp"

namespace camxiom
{

PixelResult64 rayToPixel64(const CameraModel64 &model, const Eigen::Vector3d &ray_direction)
{
  // Query-tier guard: see the float dispatch (src/dispatch.cpp) — identical
  // accept/reject across every query path; the public validateCameraModel64()
  // additionally certifies the fisheye monotone cap.
  const StatusCode v = detail64::validateCameraModelQuery64(model);
  if (v != StatusCode::OK) return detail64::invalidPixelResult(v);

  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::rayToPixel64(model, ray_direction);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::rayToPixel64(model, ray_direction);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::rayToPixel64(model, ray_direction);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::rayToPixel64(model, ray_direction);
    case ProjectionModelType::EUCM:
      return eucm::rayToPixel64(model, ray_direction);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return detail64::invalidPixelResult(StatusCode::INVALID_MODEL);
}

RayResult64 pixelToRay64(
  const CameraModel64 &model, const Pixel2d &pixel, const SolverOptions64 &opts
)
{
  // Query-tier guard: see the float dispatch (src/dispatch.cpp) — identical
  // accept/reject across every query path; the public validateCameraModel64()
  // additionally certifies the fisheye monotone cap.
  const StatusCode v = detail64::validateCameraModelQuery64(model);
  if (v != StatusCode::OK) return detail64::invalidRayResult(v);

  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::pixelToRay64(model, pixel, opts);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::pixelToRay64(model, pixel, opts);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::pixelToRay64(model, pixel, opts);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::pixelToRay64(model, pixel, opts);
    case ProjectionModelType::EUCM:
      return eucm::pixelToRay64(model, pixel, opts);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return detail64::invalidRayResult(StatusCode::INVALID_MODEL);
}

}  // namespace camxiom
