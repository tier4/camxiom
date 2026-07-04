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

#ifndef CAMXIOM__OPTIMIZER__PNP_COST_ANALYTICAL_BATCH_HPP
#define CAMXIOM__OPTIMIZER__PNP_COST_ANALYTICAL_BATCH_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <ceres/ceres.h>

#include <vector>

namespace camxiom::optimizer
{

/// Create a CostFunction that evaluates ALL points in a single view at once.
///
/// Instead of one CostFunction per point (2 residuals each), this produces
/// one CostFunction with 2*N residuals for N correspondences.
/// This drastically reduces Ceres per-block overhead and enables
/// amortised Rodrigues + camera model construction.
///
/// Parameter block layout is identical to the per-point version:
///   [0] rvec (3)   [1] tvec (3)   [2] focal_lengths (2)
///   [3] principal_points (2)       [4] dist_coeffs (effective_dist_count)
///   [5] projection_params (3)  — [xi, alpha, beta]
ceres::CostFunction *createAnalyticalBatchCost(
  const std::vector<Eigen::Vector3d> &object_points,
  const std::vector<Eigen::Vector2d> &image_points, const camxiom::CameraModel &camera_model,
  int effective_dist_count
);

}  // namespace camxiom::optimizer

#endif  // CAMXIOM__OPTIMIZER__PNP_COST_ANALYTICAL_BATCH_HPP
