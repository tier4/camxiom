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

#include "camxiom/jacobian64.hpp"
#include "jacobian/jacobian_impl.hpp"

// Double (camxiom::* ::*64) projection-Jacobian API. Thin <double> instantiations
// of the scalar-templated core in jacobian/jacobian_impl.hpp; the math (and the
// distortion Jacobian, which this file previously re-implemented locally) is now
// shared with the float path in jacobian/float.cpp.

namespace camxiom
{

// The per-model public wrappers were removed together with their
// declarations in camxiom/jacobian64.hpp (#3 parity): the generic dispatch
// below reaches the per-model math directly through
// <model>::impl::rayToPixelWithJacobian<T> in jacobian/jacobian_impl.hpp.

ProjectionJacobian64 rayToPixelWithJacobian64(
  const CameraModel64 &model, const Eigen::Vector3d &ray_direction
)
{
  return detail_impl::rayToPixelWithJacobianDispatch<double>(model, ray_direction);
}

}  // namespace camxiom
