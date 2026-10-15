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
#include <limits>
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

// --- fx=11396 incident regression (icc_online_20260713_183620.samplelog) ----
// A ~60 deg hFOV camera (f=1620 on 1920x1200) observed eight boards whose
// normals all pointed the same way (~11.5 deg about Y with ~1 deg jitter, zero
// tilt diversity). Every homography then contributes nearly proportional focal
// equations: the 2x2 normal system stays rank-2 but is ill-conditioned
// (cond ~2e5 here), so realistic corner noise + uncorrected distortion get
// amplified into a positive but wildly wrong focal pair (fx=11396/fy=5007 in
// the field). Measured worst-case focal error vs conditioning, at 0.3 px noise
// + mild k1 (the basis for kDefaultMaxNormalConditionNumber = 3e4):
//   cond ~6e2 (tilt spread ~15 deg): ~2%     cond ~1e4 (~4 deg): ~14%
//   cond ~2e4 (~3 deg):             ~21%     cond ~4e4 (~2 deg): ~40%
//   cond ~2e5 (~1 deg):             up to ~7x true (this incident class)
// The gate must refuse this geometry instead of seeding refinement with it.
namespace
{

std::vector<PlanarObservation> makeSharedTiltIncidentViews(const Eigen::Matrix3d &K)
{
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  const double kDegToRad = camxiom::constants::kPi / 180.0;
  // Same tilt direction for every view; only the board position and ~1 deg of
  // jitter vary, mimicking a hand-held board shown near-frontally.
  const double jitter_deg[8] = {0.0, 1.0, -0.75, 0.5, -0.5, 0.75, -1.0, 0.25};
  const Eigen::Vector3d translations[8] = {
    {-0.20, -0.15, 1.00}, {-0.15, -0.12, 1.05}, {-0.25, -0.18, 0.95}, {-0.18, -0.10, 1.10},
    {-0.22, -0.16, 1.00}, {-0.12, -0.14, 1.08}, {-0.24, -0.11, 0.98}, {-0.19, -0.17, 1.03},
  };
  std::vector<PlanarObservation> views(8);
  for (std::size_t i = 0; i < views.size(); ++i)
  {
    if (!buildView(
          K, rotMatY((11.5 + jitter_deg[i]) * kDegToRad), translations[i], board, views[i]
        ))
    {
      return {};
    }
  }
  return views;
}

}  // namespace

TEST(PinholeOpenCv, RejectsSharedTiltViewsAsIllConditioned)
{
  const std::vector<PlanarObservation> views =
    makeSharedTiltIncidentViews(cameraMatrix(1620.0, 1620.0, 959.5, 599.5));
  ASSERT_FALSE(views.empty());

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;
  camxiom::init::PinholeInitDiagnostics diagnostics;
  const StatusCode status = estimatePinholeOpenCv(
    views, 1920, 1200, 0.0, K_out, &diagnostics,
    camxiom::init::kDefaultMaxNormalConditionNumber
  );

  EXPECT_EQ(status, StatusCode::DEGENERATE_CONFIG);
  EXPECT_TRUE(K_out.isApprox(sentinel));
  // Margin contract for the default gate (3e4): the incident class sits above
  // 1e5, healthy tilt-diverse geometry below 1e3 (next test). If either
  // expectation fails, re-derive the default from the sweep in the suite
  // comment rather than loosening one side.
  EXPECT_GT(diagnostics.normal_condition_number, 1e5);
}

// Demonstrates the failure the gate exists for: the same geometry with a
// deterministic ~0.3 px corner perturbation and mild uncorrected radial
// distortion still passes the sign checks when the conditioning gate is
// disabled, but the focal it returns is fiction. (Noise-free corners would
// solve exactly even when ill-conditioned, hence the perturbation.)
TEST(PinholeOpenCv, SharedTiltConditioningAmplifiesNoiseIntoGarbageFocals)
{
  const Eigen::Matrix3d K = cameraMatrix(1620.0, 1620.0, 959.5, 599.5);
  std::vector<PlanarObservation> views = makeSharedTiltIncidentViews(K);
  ASSERT_FALSE(views.empty());
  // Deterministic (platform-independent) stand-in for corner noise + mild k1
  // radial model error, the conditions of the recorded incident.
  for (std::size_t v = 0; v < views.size(); ++v)
  {
    PlanarObservation &view = views[v];
    for (Eigen::Index i = 0; i < view.image_pts.cols(); ++i)
    {
      const double phase = 0.7 * static_cast<double>(i) + 2.1 * static_cast<double>(v);
      const double dx = view.image_pts(0, i) - K(0, 2);
      const double dy = view.image_pts(1, i) - K(1, 2);
      const double r2 = (dx * dx + dy * dy) / (K(0, 0) * K(0, 0));
      // k1 = -0.10 applied on the normalised radius, plus 0.3 px dither.
      view.image_pts(0, i) += -0.10 * r2 * dx + 0.3 * std::sin(phase);
      view.image_pts(1, i) += -0.10 * r2 * dy + 0.3 * std::cos(phase);
    }
  }

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Zero();
  camxiom::init::PinholeInitDiagnostics diagnostics;
  const StatusCode ungated_status = estimatePinholeOpenCv(
    views, 1920, 1200, 0.0, K_out, &diagnostics, std::numeric_limits<double>::infinity()
  );

  // Ungated, the solve must NOT be trustworthy: either the perturbation flips
  // a sign (DEGENERATE/NUMERIC) or it "succeeds" with a focal far from 1620.
  if (ungated_status == StatusCode::OK)
  {
    EXPECT_GT(std::abs(K_out(0, 0) - 1620.0) / 1620.0, 0.5)
      << "ill-conditioned solve unexpectedly accurate: fx=" << K_out(0, 0);
  }

  // The default gate refuses the same input up front.
  Eigen::Matrix3d K_gated = Eigen::Matrix3d::Zero();
  const StatusCode gated_status = estimatePinholeOpenCv(views, 1920, 1200, 0.0, K_gated);
  EXPECT_EQ(gated_status, StatusCode::DEGENERATE_CONFIG);
}

// The healthy side of the same gate: a tilt-diverse view set on the same
// camera stays orders of magnitude below the default threshold and recovers
// the true focal pair, so the gate cannot misfire on legitimate geometry.
TEST(PinholeOpenCv, TiltDiverseViewsStayWellConditioned)
{
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  const std::vector<PlanarObservation> views =
    makeViews(cameraMatrix(1620.0, 1620.0, 959.5, 599.5), board);
  ASSERT_FALSE(views.empty());

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Zero();
  camxiom::init::PinholeInitDiagnostics diagnostics;
  const StatusCode status = estimatePinholeOpenCv(
    views, 1920, 1200, 0.0, K_out, &diagnostics,
    camxiom::init::kDefaultMaxNormalConditionNumber
  );

  ASSERT_EQ(status, StatusCode::OK);
  EXPECT_NEAR(K_out(0, 0), 1620.0, 2.0);
  EXPECT_NEAR(K_out(1, 1), 1620.0, 2.0);
  EXPECT_TRUE(diagnostics.solution_positive);
  EXPECT_NEAR(diagnostics.fx, 1620.0, 2.0);
  EXPECT_NEAR(diagnostics.fy, 1620.0, 2.0);
  // See RejectsSharedTiltViewsAsIllConditioned for the margin contract.
  EXPECT_LT(diagnostics.normal_condition_number, 1e3);
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
