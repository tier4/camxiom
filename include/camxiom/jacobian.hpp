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

#ifndef CAMXIOM__JACOBIAN_HPP
#define CAMXIOM__JACOBIAN_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

namespace camxiom
{

/// 2x3 Jacobian of the projection: ∂(u,v)/∂(X,Y,Z).
///
/// Written once as a scalar-templated struct (#1 step 5c) and aliased to the
/// float32 runtime type here; the double counterpart `ProjectionJacobian64`
/// (camxiom/jacobian64.hpp) is the <double> alias. POD layout is unchanged.
template <typename T>
struct [[nodiscard]] ProjectionJacobianT
{
  StatusCode status{StatusCode::INVALID_INPUT};
  Pixel2T<T> pixel{};
  Eigen::Matrix<T, 2, 3> J{Eigen::Matrix<T, 2, 3>::Zero()};

  constexpr bool ok() const { return status == StatusCode::OK; }
  explicit operator bool() const { return ok(); }
};

using ProjectionJacobian = ProjectionJacobianT<float>;

/// Compute forward projection AND its 2x3 Jacobian simultaneously.
///
/// The Jacobian J is ∂(u,v)/∂(X,Y,Z) evaluated at ray_direction.
/// This is the composition of:
///   - ∂(x_n,y_n)/∂(X,Y,Z)       (perspective projection)
///   - ∂(x_d,y_d)/∂(x_n,y_n)     (distortion)
///   - ∂(u,v)/∂(x_d,y_d)         (intrinsics)
///
/// Supported for PINHOLE, FISHEYE_THETA, OMNIDIRECTIONAL, DOUBLE_SPHERE, and EUCM models.
ProjectionJacobian rayToPixelWithJacobian(
  const CameraModel &model, const Eigen::Vector3f &ray_direction
);

// The per-model entry points (camxiom::pinhole::rayToPixelWithJacobian, ...)
// are an internal implementation detail of the generic dispatch, mirroring
// the projection API (#3): use the generic rayToPixelWithJacobian above, or
// the batch API (camxiom/jacobian_batch.hpp) for bulk evaluation.

}  // namespace camxiom

#endif  // CAMXIOM__JACOBIAN_HPP
