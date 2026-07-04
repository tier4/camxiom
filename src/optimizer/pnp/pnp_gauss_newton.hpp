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

#ifndef CAMXIOM__OPTIMIZER__PNP_GAUSS_NEWTON_HPP
#define CAMXIOM__OPTIMIZER__PNP_GAUSS_NEWTON_HPP

#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::optimizer
{

/// Lightweight Gauss-Newton solver for 6-DOF pose estimation with FIXED
/// intrinsics and distortion.  Each view is solved independently, making
/// this dramatically faster than Ceres for the common extrinsics-only case.
///
/// Supports Huber robust loss.
struct GaussNewtonOptions
{
  int max_iterations{50};
  double function_tolerance{1e-10};
  double parameter_tolerance{1e-10};
  double gradient_tolerance{1e-10};
  double huber_delta{1.0};
};

struct GaussNewtonViewResult
{
  bool converged{false};
  int iterations{0};
  double final_cost{0.0};
};

/// Solve a single view: optimise (rvec, tvec) given fixed camera model.
///
/// @param model64       Camera model (intrinsics + distortion, fixed).
/// @param object_points 3-D target points.
/// @param image_points  Corresponding 2-D observations.
/// @param rvec          [in/out] rotation (angle-axis).
/// @param tvec          [in/out] translation.
/// @param opts          Solver options.
/// @return Per-view result (converged, iterations, cost).
GaussNewtonViewResult solveViewGaussNewton(
  const camxiom::CameraModel64 &model64, const std::vector<Eigen::Vector3d> &object_points,
  const std::vector<Eigen::Vector2d> &image_points, Eigen::Vector3d &rvec, Eigen::Vector3d &tvec,
  const GaussNewtonOptions &opts = GaussNewtonOptions()
);

}  // namespace camxiom::optimizer

#endif  // CAMXIOM__OPTIMIZER__PNP_GAUSS_NEWTON_HPP
