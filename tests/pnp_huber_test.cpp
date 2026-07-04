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

// Regression test for per-point Huber semantics on the ANALYTICAL PnP path.
//
// Ceres applies a LossFunction to the squared norm of an entire residual
// block. The analytical cost used to be one batched block per view (2*N
// residuals), so Huber down-weighted whole views instead of individual
// points — and on a single view it degenerated to plain L2 (uniformly
// scaling every residual does not move the minimiser), leaving the solve
// with no outlier robustness at all. The fix adds one analytical block per
// point whenever huber_loss_delta > 0.
//
// The test plants one gross outlier in an otherwise perfect single-view
// pose-only problem and requires the robust solve to (a) reproject the
// inliers accurately in absolute terms and (b) beat the plain-L2 solve by a
// clear factor. Before the fix (a) and (b) both fail because the robust
// solution coincides with the L2 one.

#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "support/calib_test_fixtures.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace camxiom;
using namespace camxiom::optimizer;
using camxiom::test::makeCheckerboard3D;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;

namespace
{

CameraModel makePinholeModel()
{
  CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

Eigen::Vector2d projectPinhole(
  const CameraModel &m, const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans,
  const Eigen::Vector3d &world_pt
)
{
  const Eigen::Vector3d p = rot * world_pt + trans;
  return {
    m.intrinsics.fx * p.x() / p.z() + m.intrinsics.cx,
    m.intrinsics.fy * p.y() / p.z() + m.intrinsics.cy};
}

double inlierRmse(
  const CameraModel &m, const Eigen::Vector3d &rvec, const Eigen::Vector3d &tvec,
  const ObjectPoints &object_points, const ImagePoints &clean_image_points,
  std::size_t outlier_index
)
{
  const double angle = rvec.norm();
  const Eigen::Matrix3d rot = (angle > 0.0)
                                ? Eigen::AngleAxisd(angle, rvec / angle).toRotationMatrix()
                                : Eigen::Matrix3d::Identity();
  double sum_sq = 0.0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    if (i == outlier_index)
    {
      continue;
    }
    const Eigen::Vector2d px = projectPinhole(m, rot, tvec, object_points[i]);
    sum_sq += (px - clean_image_points[i]).squaredNorm();
    ++n;
  }
  return std::sqrt(sum_sq / static_cast<double>(n));
}

}  // namespace

TEST(PnpHuber, GaussNewtonHuberSuppressesSingleOutlier)
{
  // NOTE: the pose-only flag combination (FIX_INTRINSICS | FIX_DISTORTION |
  // FIX_PROJECTION_PARAMS) routes the solve to the Gauss-Newton path, NOT
  // the Ceres ANALYTICAL cost — this test pins the GN per-point huberWeight.
  // The ANALYTICAL block granularity is covered by
  // AnalyticalHuberSuppressesSingleOutlier below.
  const CameraModel model = makePinholeModel();

  // Ground-truth pose: mildly tilted board 0.8 m in front of the camera.
  const Eigen::Matrix3d rot_gt = rotMatX(0.25) * rotMatY(-0.15);
  const Eigen::Vector3d tvec_gt(0.02, -0.05, 0.8);
  const Eigen::AngleAxisd aa_gt(rot_gt);
  const Eigen::Vector3d rvec_gt = aa_gt.axis() * aa_gt.angle();

  const ObjectPoints object_points = makeCheckerboard3D(6, 9, 0.05);
  ImagePoints image_points;
  image_points.reserve(object_points.size());
  for (const auto &wp : object_points)
  {
    image_points.push_back(projectPinhole(model, rot_gt, tvec_gt, wp));
  }
  const ImagePoints clean_image_points = image_points;

  // One gross outlier (e.g. a corner mismatched by the detector).
  const std::size_t outlier_index = 10;
  image_points[outlier_index] += Eigen::Vector2d(30.0, -20.0);

  PnpInitialGuess guess;
  guess.camera_model = model;
  guess.rvecs = {rvec_gt};
  guess.tvecs = {tvec_gt};

  const PnpFlag flags =
    PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;

  PnpSolverOptions huber_options;
  huber_options.huber_loss_delta = 1.0;

  PnpSolverOptions l2_options = huber_options;
  l2_options.huber_loss_delta = 0.0;

  PnpSolver solver_huber;
  PnpResult huber_result;
  ASSERT_TRUE(
    solver_huber.solve({object_points}, {image_points}, guess, huber_result, huber_options, flags)
  );
  ASSERT_TRUE(huber_result.success);

  PnpSolver solver_l2;
  PnpResult l2_result;
  ASSERT_TRUE(solver_l2.solve({object_points}, {image_points}, guess, l2_result, l2_options, flags)
  );
  ASSERT_TRUE(l2_result.success);

  const double huber_rmse = inlierRmse(
    model, huber_result.rvecs[0], huber_result.tvecs[0], object_points, clean_image_points,
    outlier_index
  );
  const double l2_rmse = inlierRmse(
    model, l2_result.rvecs[0], l2_result.tvecs[0], object_points, clean_image_points, outlier_index
  );

  // Per-point Huber caps the outlier influence at delta/|r| (~1/36 here), so
  // the inliers must reproject nearly perfectly...
  EXPECT_LT(huber_rmse, 0.15) << "robust solve still dragged by the outlier";
  // ...while plain L2 provably absorbs the outlier into the pose.
  EXPECT_GT(l2_rmse, huber_rmse * 3.0) << "huber=" << huber_rmse << " l2=" << l2_rmse;
}

TEST(PnpHuber, AnalyticalHuberSuppressesSingleOutlier)
{
  // Two views with free intrinsics so the solve genuinely takes the Ceres
  // ANALYTICAL path (pose-only flags would route to Gauss-Newton instead).
  // Ceres applies a LossFunction per residual block: with the batched
  // view-level block Huber cannot suppress an individual outlier point, so
  // this test fails unless robust ANALYTICAL solves use per-point blocks.
  const CameraModel model = makePinholeModel();

  const Eigen::Matrix3d rot0 = rotMatX(0.25) * rotMatY(-0.15);
  const Eigen::Vector3d tvec0(0.02, -0.05, 0.8);
  const Eigen::Matrix3d rot1 = rotMatX(-0.2) * rotMatY(0.25);
  const Eigen::Vector3d tvec1(-0.05, 0.03, 0.9);

  const ObjectPoints object_points = makeCheckerboard3D(6, 9, 0.05);
  ImagePoints img0;
  ImagePoints img1;
  for (const auto &wp : object_points)
  {
    img0.push_back(projectPinhole(model, rot0, tvec0, wp));
    img1.push_back(projectPinhole(model, rot1, tvec1, wp));
  }
  const ImagePoints clean_img0 = img0;

  const std::size_t outlier_index = 10;
  img0[outlier_index] += Eigen::Vector2d(30.0, -20.0);

  const Eigen::AngleAxisd aa0(rot0);
  const Eigen::AngleAxisd aa1(rot1);
  PnpInitialGuess guess;
  guess.camera_model = model;
  guess.rvecs = {aa0.axis() * aa0.angle(), aa1.axis() * aa1.angle()};
  guess.tvecs = {tvec0, tvec1};

  // Intrinsics free (identifiable from two tilted views); distortion and
  // projection params fixed.
  const PnpFlag flags = PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;

  PnpSolverOptions huber_options;
  huber_options.cost_type = PnpCostType::ANALYTICAL;
  huber_options.huber_loss_delta = 1.0;

  PnpSolverOptions l2_options = huber_options;
  l2_options.huber_loss_delta = 0.0;

  PnpSolver solver_huber;
  PnpResult huber_result;
  ASSERT_TRUE(solver_huber.solve(
    {object_points, object_points}, {img0, img1}, guess, huber_result, huber_options, flags
  ));
  ASSERT_TRUE(huber_result.success);

  PnpSolver solver_l2;
  PnpResult l2_result;
  ASSERT_TRUE(solver_l2.solve(
    {object_points, object_points}, {img0, img1}, guess, l2_result, l2_options, flags
  ));
  ASSERT_TRUE(l2_result.success);

  // The Ceres path must report the number of actually projectable points
  // (it used to hard-code valid_count = total).
  EXPECT_EQ(huber_result.valid_count, 2U * object_points.size());
  EXPECT_EQ(huber_result.total_count, 2U * object_points.size());

  // Evaluate with each solve's own recovered intrinsics.
  const double huber_rmse = inlierRmse(
    huber_result.camera_model, huber_result.rvecs[0], huber_result.tvecs[0], object_points,
    clean_img0, outlier_index
  );
  const double l2_rmse = inlierRmse(
    l2_result.camera_model, l2_result.rvecs[0], l2_result.tvecs[0], object_points, clean_img0,
    outlier_index
  );

  EXPECT_LT(huber_rmse, 0.15) << "robust solve still dragged by the outlier";
  EXPECT_GT(l2_rmse, huber_rmse * 3.0) << "huber=" << huber_rmse << " l2=" << l2_rmse;
}

TEST(PnpHuber, GaussNewtonRecoversWhenSeedLeavesFov)
{
  // A seed pose that pushes part of the board past theta_max used to be
  // actively repelled: the invalid-projection penalty always pulled z_cam
  // down towards 0.1, but a point that failed *in front* of the camera
  // (theta beyond theta_max) needs the opposite — lowering z pushes it
  // further out of the FOV, and the 1e3 penalty scale can overwhelm the
  // valid points. With the constant (zero-gradient) front-side penalty the
  // valid majority drives the pose back into the FOV.
  CameraModel model;
  model.intrinsics.fx = 280.0f;
  model.intrinsics.fy = 280.0f;
  model.intrinsics.cx = 320.0f;
  model.intrinsics.cy = 240.0f;
  model.projection.type = ProjectionModelType::FISHEYE_THETA;
  model.projection.theta_max = 1.2f;
  model.distortion.type = DistortionModelType::EQUIDISTANT;
  model.distortion.space = DistortionSpace::ANGLE;
  model.distortion.count = 0U;
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  const CameraModel64 m64 = toCameraModel64(model);

  const ObjectPoints object_points = makeCheckerboard3D(6, 9, 0.05);
  Eigen::Vector3d board_center(0.0, 0.0, 0.0);
  for (const auto &p : object_points)
  {
    board_center += p;
  }
  board_center /= static_cast<double>(object_points.size());

  // Ground truth: board centred at 0.8 rad incidence, normal facing the
  // camera, 0.7 m away — every corner inside theta_max with some margin.
  const double theta_c = 0.8;
  const Eigen::Vector3d dir(std::sin(theta_c), 0.0, std::cos(theta_c));
  const Eigen::Vector3d ez = -dir;
  const Eigen::Vector3d ex = Eigen::Vector3d::UnitY().cross(ez).normalized();
  const Eigen::Vector3d ey = ez.cross(ex);
  Eigen::Matrix3d rot_gt;
  rot_gt.col(0) = ex;
  rot_gt.col(1) = ey;
  rot_gt.col(2) = ez;
  const Eigen::Vector3d tvec_gt = 0.7 * dir - rot_gt * board_center;

  ImagePoints image_points;
  for (const auto &p : object_points)
  {
    const PixelResult64 pr = camxiom::rayToPixel64(m64, rot_gt * p + tvec_gt);
    ASSERT_EQ(pr.status, StatusCode::OK) << "fixture: GT view must be in FOV";
    image_points.emplace_back(pr.pixel.u, pr.pixel.v);
  }

  // Seed: rotate the camera so part of the board leaves the FOV (but the
  // valid majority remains, as the GN success gate requires).
  const Eigen::Matrix3d wobble = rotMatY(0.15);
  const Eigen::Matrix3d rot_seed = wobble * rot_gt;
  const Eigen::Vector3d tvec_seed = wobble * tvec_gt;
  int out_of_fov = 0;
  for (const auto &p : object_points)
  {
    if (camxiom::rayToPixel64(m64, rot_seed * p + tvec_seed).status != StatusCode::OK)
    {
      ++out_of_fov;
    }
  }
  ASSERT_GT(out_of_fov, 0) << "fixture: seed must push corners out of FOV";
  ASSERT_LT(out_of_fov, static_cast<int>(object_points.size()) / 2);

  const Eigen::AngleAxisd aa_seed(rot_seed);
  PnpInitialGuess guess;
  guess.camera_model = model;
  guess.rvecs = {aa_seed.axis() * aa_seed.angle()};
  guess.tvecs = {tvec_seed};

  const PnpFlag flags =
    PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;

  PnpSolver solver;
  PnpResult result;
  ASSERT_TRUE(
    solver.solve({object_points}, {image_points}, guess, result, PnpSolverOptions{}, flags)
  );
  ASSERT_TRUE(result.success);

  const Eigen::AngleAxisd aa_gt(rot_gt);
  const Eigen::Vector3d rvec_gt = aa_gt.axis() * aa_gt.angle();
  EXPECT_LT((result.tvecs[0] - tvec_gt).norm(), 1e-4) << "t = " << result.tvecs[0].transpose();
  EXPECT_LT((result.rvecs[0] - rvec_gt).norm(), 1e-4) << "r = " << result.rvecs[0].transpose();
}

TEST(PnpSolverSummary, LastSummaryResetsOnValidationFailure)
{
  // A reused solver must not report the PREVIOUS solve's summary through
  // lastSummary() after a call that failed input validation (those paths
  // return false before any optimization runs).
  const CameraModel model = makePinholeModel();
  const Eigen::Matrix3d rot_gt = rotMatX(0.2) * rotMatY(-0.1);
  const Eigen::Vector3d tvec_gt(0.02, -0.05, 0.8);
  const Eigen::AngleAxisd aa_gt(rot_gt);

  const ObjectPoints object_points = makeCheckerboard3D(6, 9, 0.05);
  ImagePoints image_points;
  image_points.reserve(object_points.size());
  for (const auto &wp : object_points)
  {
    image_points.push_back(projectPinhole(model, rot_gt, tvec_gt, wp));
  }

  PnpInitialGuess guess;
  guess.camera_model = model;
  guess.rvecs = {aa_gt.axis() * aa_gt.angle()};
  guess.tvecs = {tvec_gt};
  const PnpFlag flags =
    PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;

  PnpSolver solver;
  PnpResult result;
  ASSERT_TRUE(
    solver.solve({object_points}, {image_points}, guess, result, PnpSolverOptions{}, flags)
  );
  ASSERT_TRUE(solver.lastSummary().converged);
  ASSERT_TRUE(solver.lastSummary().solution_usable);

  // Empty input: rejected before optimizing. The summary must read as the
  // defaults, not the successful call's values.
  PnpResult failed;
  EXPECT_FALSE(
    solver.solve(ObjectPointSets{}, ImagePointSets{}, guess, failed, PnpSolverOptions{}, flags)
  );
  EXPECT_FALSE(solver.lastSummary().converged);
  EXPECT_FALSE(solver.lastSummary().solution_usable);
  EXPECT_EQ(solver.lastSummary().num_successful_steps, 0);
  EXPECT_EQ(solver.lastSummary().num_unsuccessful_steps, 0);
  EXPECT_DOUBLE_EQ(solver.lastSummary().final_cost, 0.0);
}
