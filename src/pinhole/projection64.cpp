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

#include "camxiom/projection64.hpp"

#include "pinhole/projection_impl.hpp"

// Double (camxiom::pinhole ::*64) projection API. Thin <double> instantiations
// of the scalar-templated core in pinhole/projection_impl.hpp; the math is
// shared with the float path in pinhole/projection.cpp.

namespace camxiom::pinhole
{

PixelResult64 rayToPixel64(const CameraModel64 &model, const Eigen::Vector3d &ray_direction)
{
  return impl::rayToPixel<double>(model, ray_direction);
}

RayResult64 pixelToRay64(
  const CameraModel64 &model, const Pixel2d &pixel, const SolverOptions64 &solver_options
)
{
  return impl::pixelToRay<double>(model, pixel, solver_options);
}

}  // namespace camxiom::pinhole
