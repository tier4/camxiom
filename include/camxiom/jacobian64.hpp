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

#ifndef CAMXIOM__JACOBIAN64_HPP
#define CAMXIOM__JACOBIAN64_HPP

#include "camxiom/jacobian.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

namespace camxiom
{

/// 2x3 Jacobian of the projection in double precision: ∂(u,v)/∂(X,Y,Z).
///
/// Alias of the scalar-templated ProjectionJacobianT<double> (see
/// camxiom/jacobian.hpp); the float32 counterpart is `ProjectionJacobian`.
using ProjectionJacobian64 = ProjectionJacobianT<double>;

/// Compute forward projection AND its 2x3 Jacobian simultaneously (double precision).
///
/// Supported for PINHOLE, FISHEYE_THETA, OMNIDIRECTIONAL, DOUBLE_SPHERE, and EUCM models.
ProjectionJacobian64 rayToPixelWithJacobian64(
  const CameraModel64 &model, const Eigen::Vector3d &ray_direction
);

// The per-model entry points (camxiom::pinhole::rayToPixelWithJacobian64,
// ...) are an internal implementation detail of the generic dispatch,
// mirroring the projection API (#3): use the generic function above, or the
// batch API (camxiom/jacobian_batch64.hpp) for bulk evaluation.

}  // namespace camxiom

#endif  // CAMXIOM__JACOBIAN64_HPP
