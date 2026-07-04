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

#include "camxiom/jacobian.hpp"
#include "jacobian/jacobian_impl.hpp"

// Float (camxiom::*) projection-Jacobian API. The analytic Jacobians now live
// once in the scalar-templated core (jacobian/jacobian_impl.hpp); these are the
// <float> instantiations. The double counterparts live in jacobian/double.cpp.

namespace camxiom
{

// The per-model public wrappers were removed together with their
// declarations in camxiom/jacobian.hpp (#3 parity): the generic dispatch
// below reaches the per-model math directly through
// <model>::impl::rayToPixelWithJacobian<T> in jacobian/jacobian_impl.hpp.

ProjectionJacobian rayToPixelWithJacobian(
  const CameraModel &model, const Eigen::Vector3f &ray_direction
)
{
  return detail_impl::rayToPixelWithJacobianDispatch<float>(model, ray_direction);
}

}  // namespace camxiom
