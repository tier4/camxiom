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

#ifndef CAMXIOM__JACOBIAN__FULL_JACOBIAN64_INTERNAL_HPP
#define CAMXIOM__JACOBIAN__FULL_JACOBIAN64_INTERNAL_HPP

// Internal (not installed) unchecked entry to the full-parameter Jacobian.
//
// Every consumer of rayToPixelWithFullJacobian64 runs a "fixed model x N
// points" loop (analytical batch cost, GN pose refinement, the C3
// reduced-normal rebuild); the public entry re-validates the camera model on
// EVERY call, which for a 50-view x 100-point x 50-iteration calibration is
// ~250k redundant validations. Same principle as ValidatedCameraModel (C1):
// validate once, then use the unchecked entry inside the loop.

#include "camxiom/jacobian_with_distortion_deriv64.hpp"

#include <Eigen/Core>

namespace camxiom::detail
{

/// Identical to camxiom::rayToPixelWithFullJacobian64 EXCEPT that the
/// per-call validateCameraModel64 is skipped.
///
/// Caller contract: validate the model ONCE (validateCameraModel64 == OK)
/// before a fixed-model point loop, then call this per point. All per-ray
/// guards (finite input, behind-camera, distortion domain, finite output)
/// remain inside, so per-point statuses are identical to the public entry.
FullProjectionJacobian64 rayToPixelWithFullJacobian64Unchecked(
  const CameraModel64 &model, const Eigen::Vector3d &ray_direction
);

}  // namespace camxiom::detail

#endif  // CAMXIOM__JACOBIAN__FULL_JACOBIAN64_INTERNAL_HPP
