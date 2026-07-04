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

#ifndef CAMXIOM__BATCH_HPP
#define CAMXIOM__BATCH_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

namespace camxiom
{

/// Batch forward projection: rays → pixels.
///
/// @param model           Camera model (validated once).
/// @param ray_directions  Input ray directions, size = 3 x count (col-major).
/// @param pixels_out      Output pixel coordinates, size = 2 x count (col-major).
///                        Row 0 = u, Row 1 = v.
/// @param statuses_out    Output status codes per point, size = count. May be nullptr.
/// @return Number of successfully projected points.
///         Returns -1 when Eigen input/output shapes are invalid or model validation fails.
[[nodiscard]] int rayToPixelBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xf> pixels_out, StatusCode *statuses_out = nullptr
);

/// Batch forward projection: raw pointer variant.
///
/// @param model     Camera model (validated once).
/// @param rays_xyz  Input ray directions as interleaved [x0,y0,z0, x1,y1,z1, ...], size = 3*count.
/// @param count     Number of rays.
/// @param u_out     Output u-coordinates, size = count.
/// @param v_out     Output v-coordinates, size = count.
/// @param statuses_out  Output status codes, size = count. May be nullptr.
/// @return Number of successfully projected points.
///         Returns -1 when raw inputs are invalid (e.g. negative count or nullptr required
///         pointers).
[[nodiscard]] int rayToPixelBatch(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out = nullptr
);

/// Batch inverse projection: pixels → rays.
///
/// @param model           Camera model (validated once).
/// @param pixels          Input pixel coordinates, size = 2 x count (col-major).
///                        Row 0 = u, Row 1 = v.
/// @param directions_out  Output ray directions, size = 3 x count (col-major). Unit vectors.
/// @param statuses_out    Output status codes per point, size = count. May be nullptr.
/// @param solver_options  Solver options for inverse projection.
/// @return Number of successfully unprojected points.
///         Returns -1 when Eigen input/output shapes are invalid or model validation fails.
[[nodiscard]] int pixelToRayBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix2Xf> &pixels,
  Eigen::Ref<Eigen::Matrix3Xf> directions_out, StatusCode *statuses_out = nullptr,
  const SolverOptions &solver_options = SolverOptions{}
);

/// Batch inverse projection: raw pointer variant.
///
/// @param model     Camera model (validated once).
/// @param u_in      Input u-coordinates, size = count.
/// @param v_in      Input v-coordinates, size = count.
/// @param count     Number of pixels.
/// @param dirs_xyz  Output directions as interleaved [x0,y0,z0, x1,y1,z1, ...], size = 3*count.
/// @param statuses_out  Output status codes, size = count. May be nullptr.
/// @param solver_options  Solver options for inverse projection.
/// @return Number of successfully unprojected points.
///         Returns -1 when raw inputs are invalid (e.g. negative count or nullptr required
///         pointers).
[[nodiscard]] int pixelToRayBatch(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out = nullptr, const SolverOptions &solver_options = SolverOptions{}
);

}  // namespace camxiom

#endif  // CAMXIOM__BATCH_HPP
