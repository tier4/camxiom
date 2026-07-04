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

#ifndef CAMXIOM__OPTIMIZER__PNP__ANGLE_AXIS_JACOBIAN_HPP
#define CAMXIOM__OPTIMIZER__PNP__ANGLE_AXIS_JACOBIAN_HPP

// Single source for the Rodrigues (angle-axis) point-rotation derivative
// d(R(omega) * p) / d(omega).
//
// This ~35-line formula previously existed as three hand-maintained copies
// (pnp_gauss_newton.cpp, pnp_cost_analytical_batch.cpp,
// calib/intrinsics.cpp). The calibration uncertainty diagnostics (C3/C5)
// explicitly rely on rebuilding the SAME normal equations the solver used,
// so any drift between those copies would silently break the premise the
// Schur identity tests verify. Internal header (not installed): shared by
// camxiom_calib TUs only.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

namespace camxiom::optimizer::detail
{

/// 3x3 cross-product (skew-symmetric) matrix of v.
inline Eigen::Matrix3d skew3(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

/// d(R(omega) * p) / d(omega) (3x3), with the small-angle limit -skew(p).
inline Eigen::Matrix3d angleAxisPointJacobian(
  const Eigen::Vector3d &omega, const Eigen::Vector3d &p
)
{
  const double theta2 = omega.squaredNorm();
  const Eigen::Vector3d w_cross_p = omega.cross(p);

  if (theta2 < 1e-20)
  {
    return -skew3(p);
  }

  const double theta = std::sqrt(theta2);
  const double inv_t = 1.0 / theta;
  const double inv_t2 = inv_t * inv_t;
  const double st = std::sin(theta);
  const double ct = std::cos(theta);

  const double A = st * inv_t;
  const double B = (1.0 - ct) * inv_t2;
  const double C = (theta * ct - st) * inv_t * inv_t2;
  const double D = (theta * st - 2.0 * (1.0 - ct)) * inv_t2 * inv_t2;

  const Eigen::Vector3d ww_cross_p = omega.cross(w_cross_p);

  return -A * skew3(p) - B * (skew3(w_cross_p) + skew3(omega) * skew3(p)) +
         (C * w_cross_p + D * ww_cross_p) * omega.transpose();
}

}  // namespace camxiom::optimizer::detail

#endif  // CAMXIOM__OPTIMIZER__PNP__ANGLE_AXIS_JACOBIAN_HPP
