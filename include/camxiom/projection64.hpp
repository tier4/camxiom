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

#ifndef CAMXIOM__PROJECTION64_HPP
#define CAMXIOM__PROJECTION64_HPP

#include "camxiom/types64.hpp"

#include <Eigen/Core>

namespace camxiom
{

// ---------------------------------------------------------------------------
// Double-precision projection API
// ---------------------------------------------------------------------------

// validateCameraModel64 moved to camxiom/model.hpp (beside the float
// validator).

PixelResult64 rayToPixel64(const CameraModel64 &model, const Eigen::Vector3d &ray_direction);

RayResult64 pixelToRay64(
  const CameraModel64 &model, const Pixel2d &pixel,
  const SolverOptions64 &solver_options = SolverOptions64{}
);

// The per-model hot-path entry points (camxiom::pinhole::rayToPixel64, ...) are
// an internal implementation detail of the generic dispatch / batch / SIMD
// layers and are no longer part of the public API (#3). For bulk projection
// use the batch API (camxiom/batch64.hpp); it dispatches to the SIMD kernels.

}  // namespace camxiom

#endif  // CAMXIOM__PROJECTION64_HPP
