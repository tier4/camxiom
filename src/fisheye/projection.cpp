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

#include "fisheye/projection_impl.hpp"

// Float (camxiom::fisheye) projection API. The math now lives once in the
// scalar-templated core (fisheye/projection_impl.hpp); these are the <float>
// instantiations. The double counterparts live in fisheye/projection64.cpp.

namespace camxiom::fisheye
{

PixelResult rayToPixel(const CameraModel &model, const Eigen::Vector3f &ray_direction)
{
  return impl::rayToPixel<float>(model, ray_direction);
}

RayResult pixelToRay(
  const CameraModel &model, const Pixel2 &pixel, const SolverOptions &solver_options
)
{
  return impl::pixelToRay<float>(model, pixel, solver_options);
}

// Fisheye reads nothing from the prepared state: atan2 has already produced
// theta, so the cap test is an exact theta <= theta_max comparison that needs
// no cosine, and there is no double-sphere bijectivity bound. The overloads
// exist so callers holding a fixed model bind one uniform signature across all
// five models.
PixelResult rayToPixelPrepared(
  const detail_impl::PreparedModel &prepared, const Eigen::Vector3f &ray_direction
)
{
  return impl::rayToPixel<float>(prepared.model, ray_direction);
}

RayResult pixelToRayPrepared(
  const detail_impl::PreparedModel &prepared, const Pixel2 &pixel,
  const SolverOptions &solver_options
)
{
  return impl::pixelToRay<float>(prepared.model, pixel, solver_options);
}

}  // namespace camxiom::fisheye
