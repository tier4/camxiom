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

#ifndef CAMXIOM__DEFAULT_SEED_HPP
#define CAMXIOM__DEFAULT_SEED_HPP

#include "camxiom/types.hpp"

namespace camxiom
{

/// Build a data-independent ("magic") seed CameraModel for the requested
/// projection model.
///
/// This is the canonical fallback seed used by the application layer when no
/// data-driven initial guess is available (camxiom is strategy-free; this
/// factory performs zero data-dependent computation and requires zero
/// samples).
///
/// The returned model uses only the image dimensions to place the principal
/// point and a fixed focal heuristic; all distortion coefficients and the
/// projection-specific parameters take their canonical default values:
///
///   - common to all models: cx = w/2, cy = h/2, skew = 0
///   - PINHOLE          : fx = fy = h/2; no projection params; no distortion
///   - FISHEYE_THETA    : fx = fy = h/pi; theta_max = pi;
///                        KB4 angle distortion, k1..k4 = 0
///   - OMNIDIRECTIONAL  : fx = fy = h/2; xi = 1.0; no distortion
///   - DOUBLE_SPHERE    : fx = fy = h/2; xi = -0.2, alpha = 0.5; no distortion
///   - EUCM             : fx = fy = h/2; alpha = 0.5, beta = 1.0; no distortion
///
/// For every valid (model_type, image_width, image_height) with
/// image_width > 0 and image_height > 0, the returned model satisfies
/// validateCameraModel(model) == StatusCode::OK.
///
/// Invalid input handling: if image_width <= 0, image_height <= 0, or
/// model_type is not one of the five valid projection types (i.e. UNKNOWN
/// or any out-of-range value), a default-constructed CameraModel{} is
/// returned. Its projection.type is ProjectionModelType::UNKNOWN, which
/// validateCameraModel rejects (StatusCode::INVALID_MODEL), giving callers
/// a detectable sentinel without throwing or clamping.
///
/// \param model_type    Requested projection model.
/// \param image_width   Image width in pixels (must be > 0).
/// \param image_height  Image height in pixels (must be > 0).
/// \return A seed CameraModel, or a default-constructed sentinel on
///         invalid input.
[[nodiscard]] CameraModel getDefaultSeed(
  ProjectionModelType model_type, int image_width, int image_height
);

}  // namespace camxiom

#endif  // CAMXIOM__DEFAULT_SEED_HPP
