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

#ifndef CAMXIOM__INIT__INIT_DETAIL_HPP
#define CAMXIOM__INIT__INIT_DETAIL_HPP

// Small geometry helpers shared by the init estimators. These were verbatim
// per-file copies in mei_omni / double_sphere / eucm / kb4_fisheye; hoisted
// here (internal header, not installed) so they cannot drift.

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace camxiom::init::detail
{

/// Convert a Z=0 planar board (2 x M) into a 3 x M world-point matrix.
inline Eigen::Matrix3Xd liftBoardToZ0(const Eigen::Matrix2Xd &board_pts)
{
  const Eigen::Index m = board_pts.cols();
  Eigen::Matrix3Xd world(3, m);
  world.topRows<2>() = board_pts;
  world.row(2).setZero();
  return world;
}

/// Convert a rotation matrix to a Rodrigues vector (axis * angle).
inline Eigen::Vector3d rotationMatrixToAngleAxis(const Eigen::Matrix3d &R)
{
  const Eigen::AngleAxisd aa(R);
  return aa.axis() * aa.angle();
}

/// Convert a Rodrigues vector back to a rotation matrix.
inline Eigen::Matrix3d angleAxisToRotationMatrix(const Eigen::Vector3d &rvec)
{
  const double angle = rvec.norm();
  if (!(angle > 0.0))
  {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::AngleAxisd aa(angle, rvec / angle);
  return aa.toRotationMatrix();
}

}  // namespace camxiom::init::detail

#endif  // CAMXIOM__INIT__INIT_DETAIL_HPP
