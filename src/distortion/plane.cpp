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
#include "distortion/plane_impl.hpp"

// Float (camxiom::detail) plane-distortion API. The math now lives once in the
// scalar-templated detail_impl core (distortion/plane_impl.hpp); these are the
// <float> instantiations exposed under the historical detail:: names.
// The double counterparts live in src/projection64/internal.hpp.

namespace camxiom::detail
{

StatusCode distortPlaneModel(
  const DistortionModel &model, const NormPoint &in_xy, NormPoint &out_xy
)
{
  return detail_impl::distortPlaneModel<float>(model, in_xy, out_xy);
}

StatusCode distortPlaneModelWithJacobian(
  const DistortionModel &model, const NormPoint &in_xy, NormPoint &out_xy,
  DistortionJacobian2x2 &jacobian
)
{
  return detail_impl::distortPlaneModelWithJacobian<float>(
    model, in_xy, out_xy, jacobian.j00, jacobian.j01, jacobian.j10, jacobian.j11
  );
}

StatusCode undistortPlaneModelNewton(
  const DistortionModel &model, const NormPoint &observed_xy, const SolverOptions &solver_options,
  NormPoint &undistorted_xy
)
{
  return detail_impl::undistortPlaneModel<float>(
    model, observed_xy, solver_options.max_iterations, solver_options.residual_tolerance,
    solver_options.step_tolerance, solver_options.skip_verify, undistorted_xy
  );
}

}  // namespace camxiom::detail
