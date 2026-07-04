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

#ifndef CAMXIOM__JACOBIAN_WITH_DISTORTION_DERIV64_HPP
#define CAMXIOM__JACOBIAN_WITH_DISTORTION_DERIV64_HPP

#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <array>

namespace camxiom
{

/// Extended Jacobian result for optimization use.
///
/// In addition to the standard 2x3 ∂(u,v)/∂(X,Y,Z), this struct provides:
///   - Distorted normalized coordinates (xd, yd) before intrinsics application,
///     needed for ∂(u,v)/∂(fx,fy) = [[xd,0],[0,yd]].
///   - Per-coefficient distortion Jacobian ∂(xd,yd)/∂dist[i],
///     so that ∂(u,v)/∂dist[i] = [[fx*dxd_ddist[i]],[fy*dyd_ddist[i]]].
struct [[nodiscard]] FullProjectionJacobian64
{
  StatusCode status{StatusCode::INVALID_INPUT};
  Pixel2d pixel{};
  double xd{0.0};
  double yd{0.0};
  Eigen::Matrix<double, 2, 3> J_point{Eigen::Matrix<double, 2, 3>::Zero()};
  std::array<double, 14> dxd_ddist{};
  std::array<double, 14> dyd_ddist{};
  int dist_count{0};
  std::array<double, 3> dxd_dproj{};  // ∂xd/∂[xi, alpha, beta]
  std::array<double, 3> dyd_dproj{};  // ∂yd/∂[xi, alpha, beta]
  int proj_param_count{0};

  constexpr bool ok() const { return status == StatusCode::OK; }
  explicit operator bool() const { return ok(); }
};

/// Compute forward projection with full Jacobian information (double precision).
///
/// Supported for PINHOLE, FISHEYE_THETA, OMNIDIRECTIONAL, DOUBLE_SPHERE, and EUCM.
FullProjectionJacobian64 rayToPixelWithFullJacobian64(
  const CameraModel64 &model, const Eigen::Vector3d &ray_direction
);

}  // namespace camxiom

#endif  // CAMXIOM__JACOBIAN_WITH_DISTORTION_DERIV64_HPP
