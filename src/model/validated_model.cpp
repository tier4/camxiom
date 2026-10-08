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
#include "detail/projection_common.hpp"
#include "detail/projection_models.hpp"

namespace camxiom
{
namespace
{

using ForwardFn = PixelResult (*)(const detail_impl::PreparedModel &, const Eigen::Vector3f &);
using InverseFn =
  RayResult (*)(const detail_impl::PreparedModel &, const Pixel2 &, const SolverOptions &);

// Map ProjectionModelType -> the internal per-model forward/inverse entry
// points. This is the same resolution the generic dispatch switch
// (src/dispatch.cpp) and the batch layer (src/batch/float.cpp) perform; here it
// runs once at construction so the wrapped model never re-dispatches.
ForwardFn resolveForwardFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::rayToPixelPrepared;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::rayToPixelPrepared;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::rayToPixelPrepared;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::rayToPixelPrepared;
    case ProjectionModelType::EUCM:
      return &eucm::rayToPixelPrepared;
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
      return &pinhole::pixelToRayPrepared;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::pixelToRayPrepared;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::pixelToRayPrepared;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::pixelToRayPrepared;
    case ProjectionModelType::EUCM:
      return &eucm::pixelToRayPrepared;
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

  // The model is validated and immutable from here on, so its derived
  // constants are too: derive them once instead of on every projected point.
  return ValidatedCameraModel(
    detail_impl::PreparedModel{model, detail_impl::prepareProjection(model.projection)}, forward,
    inverse
  );
}

PixelResult ValidatedCameraModel::rayToPixel(const Eigen::Vector3f &ray_direction) const
{
  return forward_(prepared_, ray_direction);
}

PixelResult ValidatedCameraModel::rayToPixel(
  const float x_direction, const float y_direction, const float z_direction
) const
{
  return forward_(prepared_, Eigen::Vector3f(x_direction, y_direction, z_direction));
}

PixelResult ValidatedCameraModel::rayToPixel(const Ray3 &ray) const
{
  return forward_(prepared_, ray.direction);
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
  return inverse_(prepared_, pixel, solver_options);
}

RayResult ValidatedCameraModel::pixelToRay(
  const float u, const float v, const SolverOptions &solver_options
) const
{
  return pixelToRay(Pixel2{u, v}, solver_options);
}

}  // namespace camxiom
