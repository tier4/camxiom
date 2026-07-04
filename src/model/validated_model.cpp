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

#include "camxiom/validated_model.hpp"

#include "camxiom/model.hpp"
#include "detail/internal.hpp"
#include "detail/projection_models.hpp"

namespace camxiom
{
namespace
{

using ForwardFn = PixelResult (*)(const CameraModel &, const Eigen::Vector3f &);
using InverseFn = RayResult (*)(const CameraModel &, const Pixel2 &, const SolverOptions &);

// Map ProjectionModelType -> the internal per-model forward/inverse entry
// points. This is the same resolution the generic dispatch switch
// (src/dispatch.cpp) and the batch layer (src/batch/float.cpp) perform; here it
// runs once at construction so the wrapped model never re-dispatches.
ForwardFn resolveForwardFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::rayToPixel;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::rayToPixel;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::rayToPixel;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::rayToPixel;
    case ProjectionModelType::EUCM:
      return &eucm::rayToPixel;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

InverseFn resolveInverseFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::pixelToRay;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::pixelToRay;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::pixelToRay;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::pixelToRay;
    case ProjectionModelType::EUCM:
      return &eucm::pixelToRay;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

}  // namespace

std::optional<ValidatedCameraModel> ValidatedCameraModel::tryMake(const CameraModel &model)
{
  if (validateCameraModel(model) != StatusCode::OK)
  {
    return std::nullopt;
  }

  const ForwardFn forward = resolveForwardFn(model.projection.type);
  const InverseFn inverse = resolveInverseFn(model.projection.type);
  // validateCameraModel already rejects UNKNOWN / unsupported projection types,
  // so both resolvers succeed here; the guard keeps the invariant explicit.
  if (forward == nullptr || inverse == nullptr)
  {
    return std::nullopt;
  }

  return ValidatedCameraModel(model, forward, inverse);
}

PixelResult ValidatedCameraModel::rayToPixel(const Eigen::Vector3f &ray_direction) const
{
  return forward_(model_, ray_direction);
}

PixelResult ValidatedCameraModel::rayToPixel(
  const float x_direction, const float y_direction, const float z_direction
) const
{
  return forward_(model_, Eigen::Vector3f(x_direction, y_direction, z_direction));
}

PixelResult ValidatedCameraModel::rayToPixel(const Ray3 &ray) const
{
  return forward_(model_, ray.direction);
}

RayResult ValidatedCameraModel::pixelToRay(const Pixel2 &pixel, const SolverOptions &solver_options)
  const
{
  // Mirror the generic pixelToRay input guard (src/dispatch.cpp) so behaviour is
  // identical bar the skipped model re-validation.
  if (!detail::isFinite2(pixel.u, pixel.v))
  {
    return detail::invalidRayResult(StatusCode::INVALID_INPUT);
  }
  return inverse_(model_, pixel, solver_options);
}

RayResult ValidatedCameraModel::pixelToRay(
  const float u, const float v, const SolverOptions &solver_options
) const
{
  return pixelToRay(Pixel2{u, v}, solver_options);
}

}  // namespace camxiom
