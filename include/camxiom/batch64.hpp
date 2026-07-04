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

#ifndef CAMXIOM__BATCH64_HPP
#define CAMXIOM__BATCH64_HPP

#include "camxiom/types64.hpp"

#include <Eigen/Core>

namespace camxiom
{

/// Batch forward projection (double): rays → pixels.
/// Returns -1 when Eigen input/output shapes are invalid or model validation fails.
[[nodiscard]] int rayToPixelBatch64(
  const CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix3Xd> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xd> pixels_out, StatusCode *statuses_out = nullptr
);

/// Batch forward projection (double): raw pointer variant.
/// Returns -1 when raw inputs are invalid (e.g. negative count or nullptr required pointers).
[[nodiscard]] int rayToPixelBatch64(
  const CameraModel64 &model, const double *rays_xyz, int count, double *u_out, double *v_out,
  StatusCode *statuses_out = nullptr
);

/// Batch inverse projection (double): pixels → rays.
/// Returns -1 when Eigen input/output shapes are invalid or model validation fails.
[[nodiscard]] int pixelToRayBatch64(
  const CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix2Xd> &pixels,
  Eigen::Ref<Eigen::Matrix3Xd> directions_out, StatusCode *statuses_out = nullptr,
  const SolverOptions64 &solver_options = SolverOptions64{}
);

/// Batch inverse projection (double): raw pointer variant.
/// Returns -1 when raw inputs are invalid (e.g. negative count or nullptr required pointers).
[[nodiscard]] int pixelToRayBatch64(
  const CameraModel64 &model, const double *u_in, const double *v_in, int count, double *dirs_xyz,
  StatusCode *statuses_out = nullptr, const SolverOptions64 &solver_options = SolverOptions64{}
);

}  // namespace camxiom

#endif  // CAMXIOM__BATCH64_HPP
