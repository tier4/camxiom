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

#ifndef CAMXIOM__JACOBIAN_BATCH_HPP
#define CAMXIOM__JACOBIAN_BATCH_HPP

#include "camxiom/jacobian.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

namespace camxiom
{

/// Batch forward projection with Jacobian: rays → pixels + 2x3 Jacobians.
///
/// @param model           Camera model (validated once).
/// @param ray_directions  Input ray directions, size = 3 x count (col-major).
/// @param pixels_out      Output pixel coordinates, size = 2 x count (col-major).
/// @param jacobians_out   Output Jacobians, array of count 2x3 matrices.
/// @param statuses_out    Output status codes per point, size = count. May be nullptr.
/// @return Number of successfully projected points.
///         Returns -1 when Eigen input/output shapes are invalid or model validation fails.
[[nodiscard]] int rayToPixelWithJacobianBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xf> pixels_out, Eigen::Matrix<float, 2, 3> *jacobians_out,
  StatusCode *statuses_out = nullptr
);

/// Batch forward projection with Jacobian: raw pointer variant.
///
/// @param model       Camera model (validated once).
/// @param rays_xyz    Input ray directions as interleaved [x0,y0,z0, ...], size = 3*count.
/// @param count       Number of rays.
/// @param u_out       Output u-coordinates, size = count.
/// @param v_out       Output v-coordinates, size = count.
/// @param jacobians_out  Output Jacobians, array of count 2x3 matrices (row-major float[6]).
///                       Layout per point: [du/dX, du/dY, du/dZ, dv/dX, dv/dY, dv/dZ].
///                       May be nullptr to skip Jacobian output.
/// @param statuses_out   Output status codes, size = count. May be nullptr.
/// @return Number of successfully projected points.
///         Returns -1 when raw inputs are invalid (e.g. negative count or nullptr required
///         pointers).
[[nodiscard]] int rayToPixelWithJacobianBatch(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  float *jacobians_out, StatusCode *statuses_out = nullptr
);

}  // namespace camxiom

#endif  // CAMXIOM__JACOBIAN_BATCH_HPP
