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

#ifndef CAMXIOM__PROJECTION_HPP
#define CAMXIOM__PROJECTION_HPP

#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <array>

namespace camxiom
{

// ---------------------------------------------------------------------------
// Generic dispatch API — validates model and dispatches by ProjectionModelType
// ---------------------------------------------------------------------------

PixelResult rayToPixel(const CameraModel &model, const Eigen::Vector3f &ray_direction);
PixelResult rayToPixel(
  const CameraModel &model, float x_direction, float y_direction, float z_direction
);
/// Convenience overload for Ray3. Projection is central: only ray.direction
/// is used and ray.origin is IGNORED (assumed to be the camera center).
PixelResult rayToPixel(const CameraModel &model, const Ray3 &ray);

RayResult pixelToRay(
  const CameraModel &model, const Pixel2 &pixel,
  const SolverOptions &solver_options = SolverOptions{}
);
RayResult pixelToRay(
  const CameraModel &model, float u, float v, const SolverOptions &solver_options = SolverOptions{}
);

/// True iff the ray direction lies inside the model's valid projection
/// domain, i.e. rayToPixel(model, ray_direction) would return OK (not
/// BEHIND_CAMERA / OUT_OF_FOV / an error). It performs the projection, so it
/// is a readability helper, not a cheaper path. No image-bounds check.
[[nodiscard]] bool isRayProjectable(const CameraModel &model, const Eigen::Vector3f &ray_direction);

/// As above, and additionally requires the projected pixel to fall inside
/// [0, image_width) x [0, image_height) — the same in-bounds rule the remap
/// builders apply.
[[nodiscard]] bool isRayProjectable(
  const CameraModel &model, const Eigen::Vector3f &ray_direction, int image_width, int image_height
);

// The per-model hot-path entry points (camxiom::pinhole::rayToPixel, ...) are
// an internal implementation detail of the generic dispatch / batch / SIMD
// layers and are no longer part of the public API (#3). For bulk projection
// use the batch API (camxiom/batch.hpp); it dispatches to the SIMD kernels.

}  // namespace camxiom

#endif  // CAMXIOM__PROJECTION_HPP
