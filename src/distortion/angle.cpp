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

#include "camxiom/types.hpp"
#include "detail/internal.hpp"
#include "distortion/angle_impl.hpp"

// Float (camxiom::detail) angle-distortion API. The math lives once in the
// scalar-templated detail_impl core (distortion/angle_impl.hpp); these are the
// <float> instantiations. The double counterparts forward to the same core
// from src/projection64/internal.hpp.

namespace camxiom::detail
{

StatusCode distortTheta(const DistortionModel &model, const float theta, float &theta_d)
{
  return detail_impl::distortTheta<float>(model, theta, theta_d);
}

StatusCode distortThetaDerivative(
  const DistortionModel &model, const float theta, float &derivative_out
)
{
  return detail_impl::distortThetaDerivative<float>(model, theta, derivative_out);
}

StatusCode undistortThetaHybrid(
  const DistortionModel &model, const ProjectionModel &projection, const float radius_d,
  const SolverOptions &solver_options, float &theta_out
)
{
  return detail_impl::undistortThetaHybrid<float>(
    model, projection, radius_d, solver_options.max_iterations, solver_options.residual_tolerance,
    solver_options.step_tolerance, theta_out
  );
}

}  // namespace camxiom::detail
