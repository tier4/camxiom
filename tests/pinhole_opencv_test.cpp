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

#include "camxiom/init/pinhole_opencv.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "support/calib_test_fixtures.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using camxiom::StatusCode;
using camxiom::init::estimatePinholeOpenCv;
using camxiom::init::PlanarObservation;

namespace
{

// Shared synthetic-geometry helpers (tests/support/calib_test_fixtures.hpp).
using camxiom::test::makeCheckerboard;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;

bool buildView(
  const Eigen::Matrix3d &K, const Eigen::Matrix3d &rotation, const Eigen::Vector3d &translation,
  const Eigen::Matrix2Xd &board, PlanarObservation &view
)
{
  view.board_pts = board;
  view.image_pts.resize(2, board.cols());
  for (Eigen::Index i = 0; i < board.cols(); ++i)
  {
    const Eigen::Vector3d point(board(0, i), board(1, i), 0.0);
    const Eigen::Vector3d camera_point = rotation * point + translation;
    if (camera_point.z() <= 0.0)
    {
      return false;
    }
    const Eigen::Vector3d pixel = K * camera_point;
    view.image_pts(0, i) = pixel.x() / pixel.z();
    view.image_pts(1, i) = pixel.y() / pixel.z();
  }
  return true;
}

std::vector<PlanarObservation> makeViews(const Eigen::Matrix3d &K, const Eigen::Matrix2Xd &board)
{
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const double deg20 = 20.0 * camxiom::constants::kPi / 180.0;
  std::vector<PlanarObservation> views(5);
  if (!buildView(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.2, -0.15, 1.0), board, views[0]) || !buildView(K, rotMatX(deg25), Eigen::Vector3d(-0.2, -0.15, 1.2), board, views[1]) || !buildView(K, rotMatX(-deg25), Eigen::Vector3d(-0.2, -0.15, 1.0), board, views[2]) || !buildView(K, rotMatY(deg25), Eigen::Vector3d(-0.2, -0.15, 1.0), board, views[3]) || !buildView(K, rotMatX(deg20) * rotMatY(-deg20), Eigen::Vector3d(-0.2, -0.15, 1.1), board, views[4]))
  {
    return {};
  }
  return views;
}

Eigen::Matrix3d cameraMatrix(double fx, double fy, double cx, double cy)
{
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = fx;
  K(1, 1) = fy;
  K(0, 2) = cx;
  K(1, 2) = cy;
  return K;
}

}  // namespace

TEST(PinholeOpenCv, RecoversFocalWithPrincipalPointFixedAtImageCenter)
{
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  const std::vector<PlanarObservation> views =
    makeViews(cameraMatrix(500.0, 500.0, 320.0, 240.0), board);
  ASSERT_FALSE(views.empty());

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Zero();
  const StatusCode status = estimatePinholeOpenCv(views, 640, 480, 0.0, K_out);

  ASSERT_EQ(status, StatusCode::OK);
  // OpenCV fixes the initial principal point at ((w - 1) / 2, (h - 1) / 2).
  EXPECT_NEAR(K_out(0, 0), 500.0, 0.5);
  EXPECT_NEAR(K_out(1, 1), 500.0, 0.5);
  EXPECT_DOUBLE_EQ(K_out(0, 2), 319.5);
  EXPECT_DOUBLE_EQ(K_out(1, 2), 239.5);
  EXPECT_DOUBLE_EQ(K_out(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(K_out(2, 2), 1.0);
}

TEST(PinholeOpenCv, SupportsUltraNarrowFiniteFocalLengths)
{
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  const std::vector<PlanarObservation> views =
    makeViews(cameraMatrix(9000.0, 8900.0, 1919.5, 1079.5), board);
  ASSERT_FALSE(views.empty());

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Zero();
  const StatusCode status = estimatePinholeOpenCv(views, 3840, 2160, 0.0, K_out);

  ASSERT_EQ(status, StatusCode::OK);
  EXPECT_NEAR(K_out(0, 0), 9000.0, 0.5);
  EXPECT_NEAR(K_out(1, 1), 8900.0, 0.5);
  EXPECT_DOUBLE_EQ(K_out(0, 2), 1919.5);
  EXPECT_DOUBLE_EQ(K_out(1, 2), 1079.5);
}

TEST(PinholeOpenCv, RejectsFrontParallelDegenerateViews)
{
  const Eigen::Matrix3d K = cameraMatrix(500.0, 500.0, 320.0, 240.0);
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  std::vector<PlanarObservation> views(3);
  ASSERT_TRUE(
    buildView(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.2, -0.15, 0.8), board, views[0])
  );
  ASSERT_TRUE(
    buildView(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.1, -0.10, 1.0), board, views[1])
  );
  ASSERT_TRUE(
    buildView(K, Eigen::Matrix3d::Identity(), Eigen::Vector3d(-0.3, -0.20, 1.2), board, views[2])
  );

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;
  const StatusCode status = estimatePinholeOpenCv(views, 640, 480, 0.0, K_out);

  EXPECT_EQ(status, StatusCode::DEGENERATE_CONFIG);
  EXPECT_TRUE(K_out.isApprox(sentinel));
}
