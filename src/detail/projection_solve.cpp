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

#include "detail/projection_common.hpp"
#include "distortion/angle_impl.hpp"
#include "distortion/plane_impl.hpp"

// Single out-of-line home for the heavy iterative inverse solvers shared by the
// per-model pixelToRay cores (#1 step 3). Compiling the Newton/LM (plane) and
// bracketed-Newton (theta) bodies once here — instead of inlining them into
// every model's projection translation unit — preserves the pre-unification
// float inverse hot-path cost. See src/detail/projection_common.hpp.

namespace camxiom::detail_impl
{

StatusCode undistortPlaneSolve(
  const DistortionModelT<float> &model, const NormPointT<float> &observed, const int max_iterations,
  const float residual_tolerance, const float step_tolerance, const bool skip_verify,
  NormPointT<float> &undistorted
)
{
  return undistortPlaneModel<float>(
    model, observed, max_iterations, residual_tolerance, step_tolerance, skip_verify, undistorted
  );
}

StatusCode undistortPlaneSolve(
  const DistortionModelT<double> &model, const NormPointT<double> &observed,
  const int max_iterations, const double residual_tolerance, const double step_tolerance,
  const bool skip_verify, NormPointT<double> &undistorted
)
{
  return undistortPlaneModel<double>(
    model, observed, max_iterations, residual_tolerance, step_tolerance, skip_verify, undistorted
  );
}

StatusCode undistortThetaSolve(
  const DistortionModelT<float> &model, const ProjectionModelT<float> &projection,
  const float radius_d, const int max_iterations, const float residual_tolerance,
  const float step_tolerance, float &theta_out
)
{
  return undistortThetaHybrid<float>(
    model, projection, radius_d, max_iterations, residual_tolerance, step_tolerance, theta_out
  );
}

StatusCode undistortThetaSolve(
  const DistortionModelT<double> &model, const ProjectionModelT<double> &projection,
  const double radius_d, const int max_iterations, const double residual_tolerance,
  const double step_tolerance, double &theta_out
)
{
  return undistortThetaHybrid<double>(
    model, projection, radius_d, max_iterations, residual_tolerance, step_tolerance, theta_out
  );
}

}  // namespace camxiom::detail_impl
