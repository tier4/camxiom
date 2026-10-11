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

// Tests for the ROS-free / OpenCV-free compat interchange layer
// (src/compat/factory.cpp + src/<model>/compat.cpp).
//
// makeCameraModel() is the single canonical K/D/distortion-model-string parser
// that every adapter (ROS, OpenCV) funnels through, and the per-model
// import/export*Model functions are the profile-aware conversions used by the
// interop layers. Until now none of this had a direct test (LOADMAP T3).
//
// Strategy: for every model family, round-trip an external description through
// import -> CameraModel -> export and assert it comes back (K + coefficients +
// model-specific scalars), plus characterize the makeCameraModel() string
// parsing and the deterministic error paths. Core test (no Ceres/ROS/OpenCV).

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace camxiom;
using namespace camxiom::optimizer;

namespace
{
CameraModel64 makeModel64(const std::string &family, const std::vector<double> &coeffs)
{
  CameraInfo info;
  info.k = {
    513.1234567890123,
    0.1234567890123,
    319.9876543210987,
    0.0,
    509.9876543210987,
    239.1234567890123,
    0.0,
    0.0,
    1.0};
  info.distortion_model = family;
  info.d = coeffs;
  auto model = makeCameraModel64(info);
  if (model.projection.type != ProjectionModelType::PINHOLE)
    model.projection.theta_max = 0.912345678901234;
  return model;
}

void makeObservations(const CameraModel64 &model, ObjectPoints &objects, ImagePoints &images)
{
  for (int row = -4; row <= 4; ++row)
    for (int col = -5; col <= 5; ++col)
    {
      const Eigen::Vector3d ray(0.12 * col, 0.12 * row, 1.0 + 0.03 * ((col + row + 9) % 3));
      const auto pixel = rayToPixel64(model, ray);
      ASSERT_EQ(pixel.status, StatusCode::OK);
      objects.push_back(ray);
      images.emplace_back(pixel.pixel.u, pixel.pixel.v);
    }
}
}  // namespace

TEST(PnpPrecision, FitsAndReusesDoubleModelsInBothCeresPaths)
{
  const std::vector<CameraModel64> models = {
    makeModel64(
      "plumb_bob", {0.013456789012345, -0.002345678901234, 0.000123456789012, -0.000234567890123,
                    0.000345678901234}
    ),
    makeModel64(
      "equidistant", {0.013456789012345, -0.002345678901234, 0.000123456789012, -0.000023456789012}
    ),
    makeModel64(
      "omnidirectional", {0.513456789012345, 0.013456789012345, -0.002345678901234,
                          0.000123456789012, -0.000234567890123}
    ),
    makeModel64("double_sphere", {0.213456789012345, 0.613456789012345}),
    makeModel64("eucm", {0.413456789012345, 1.013456789012345})};
  for (const auto &truth : models)
    for (const auto cost : {PnpCostType::ANALYTICAL, PnpCostType::AUTO_DIFF})
    {
      SCOPED_TRACE(static_cast<int>(truth.projection.type));
      SCOPED_TRACE(static_cast<int>(cost));
      ASSERT_EQ(validateCameraModel64(truth), StatusCode::OK);
      ObjectPoints objects;
      ImagePoints images;
      makeObservations(truth, objects, images);
      PnpInitialGuess64 guess;
      guess.camera_model = truth;
      guess.camera_model.intrinsics.fx += 3.0;
      guess.camera_model.intrinsics.fy -= 2.0;
      guess.rvecs = {Eigen::Vector3d::Zero()};
      guess.tvecs = {Eigen::Vector3d::Zero()};
      PnpSolverOptions options;
      options.cost_type = cost;
      options.huber_loss_delta = 0.0;
      options.solver_options.function_tolerance = 1e-14;
      options.solver_options.parameter_tolerance = 1e-14;
      options.solver_options.gradient_tolerance = 1e-14;
      const auto flags = PnpFlag::FIX_EXTRINSICS | PnpFlag::FIX_PRINCIPAL_POINTS |
                         PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;
      PnpSolver solver;
      PnpResult64 result;
      ASSERT_TRUE(solver.solve(objects, images, guess, result, options, flags));
      EXPECT_NEAR(result.camera_model.intrinsics.fx, truth.intrinsics.fx, 1e-9);
      EXPECT_NEAR(result.camera_model.intrinsics.fy, truth.intrinsics.fy, 1e-9);
      EXPECT_EQ(result.camera_model.intrinsics.cx, truth.intrinsics.cx);
      EXPECT_EQ(result.camera_model.intrinsics.skew, truth.intrinsics.skew);
      EXPECT_EQ(result.camera_model.distortion.coeffs, truth.distortion.coeffs);
      EXPECT_EQ(result.camera_model.projection.xi, truth.projection.xi);
      EXPECT_EQ(result.camera_model.projection.alpha, truth.projection.alpha);
      EXPECT_EQ(result.camera_model.projection.beta, truth.projection.beta);
      EXPECT_EQ(result.camera_model.projection.theta_max, truth.projection.theta_max);
      EXPECT_LT(result.rmse, 1e-9);
      EXPECT_EQ(result.valid_count, objects.size());
      EXPECT_GT(
        std::abs(
          static_cast<double>(static_cast<float>(truth.intrinsics.fx)) -
          result.camera_model.intrinsics.fx
        ),
        1e-6
      );
      guess.camera_model = result.camera_model;
      const auto reused =
        solver.solve(ObjectPointSets{objects}, ImagePointSets{images}, guess, options, flags);
      ASSERT_TRUE(reused);
      EXPECT_LT(reused.rmse, 1e-9);
    }
}

TEST(PnpPrecision, PoseOnlyPathPreservesDoubleAndFailureResetsResult)
{
  const auto model = makeModel64("plumb_bob", {});
  ObjectPoints objects;
  ImagePoints images;
  makeObservations(model, objects, images);
  PnpInitialGuess64 guess;
  guess.camera_model = model;
  guess.rvecs = {Eigen::Vector3d(0.002, -0.001, 0.003)};
  guess.tvecs = {Eigen::Vector3d(0.001, -0.002, 0.004)};
  PnpResult64 result;
  PnpSolver solver;
  const auto flags =
    PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;
  ASSERT_TRUE(solver.solve(objects, images, guess, result, PnpSolverOptions{}, flags));
  EXPECT_EQ(result.camera_model.intrinsics.fx, model.intrinsics.fx);
  EXPECT_EQ(result.camera_model.intrinsics.cx, model.intrinsics.cx);
  EXPECT_LT(result.rmse, 1e-7);
  guess.camera_model.intrinsics.fx = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(solver.solve(objects, images, guess, result));
  EXPECT_FALSE(result.success);
  EXPECT_TRUE(result.rvecs.empty());
  EXPECT_EQ(result.valid_count, 0U);
  EXPECT_FALSE(solver.lastSummary().solution_usable);
}

TEST(PnpPrecision, FloatAndDoubleSolvesPreserveExplicitDomainCaps)
{
  for (const auto &family :
       {std::string("ideal_equidistant"), std::string("omnidirectional"),
        std::string("double_sphere"), std::string("eucm")})
  {
    const auto coeffs = family == "double_sphere"     ? std::vector<double>{0.2, 0.6}
                        : family == "eucm"            ? std::vector<double>{0.4, 1.0}
                        : family == "omnidirectional" ? std::vector<double>{0.5}
                                                      : std::vector<double>{};
    const auto model64 = makeModel64(family, coeffs);
    const auto model = fromCameraModel64(model64);
    ObjectPoints objects;
    ImagePoints images;
    makeObservations(toCameraModel64(model), objects, images);
    PnpInitialGuess guess;
    guess.camera_model = model;
    guess.rvecs = {Eigen::Vector3d::Zero()};
    guess.tvecs = {Eigen::Vector3d::Zero()};
    PnpResult result;
    PnpSolver solver;
    const auto flags =
      PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;
    ASSERT_TRUE(solver.solve(objects, images, guess, result, PnpSolverOptions{}, flags));
    EXPECT_EQ(result.camera_model.projection.theta_max, model.projection.theta_max);
    const double theta = model.projection.theta_max + 0.01;
    const Eigen::Vector3d outside(std::sin(theta), 0.0, std::cos(theta));
    EXPECT_EQ(
      rayToPixel64(toCameraModel64(result.camera_model), outside).status, StatusCode::OUT_OF_FOV
    );
    objects.clear();
    images.clear();
    makeObservations(model64, objects, images);
    PnpInitialGuess64 guess64;
    guess64.camera_model = model64;
    guess64.rvecs = guess.rvecs;
    guess64.tvecs = guess.tvecs;
    PnpResult64 result64;
    ASSERT_TRUE(solver.solve(objects, images, guess64, result64, PnpSolverOptions{}, flags));
    EXPECT_EQ(result64.camera_model.projection.theta_max, model64.projection.theta_max);
    EXPECT_EQ(rayToPixel64(result64.camera_model, outside).status, StatusCode::OUT_OF_FOV);
  }
}
