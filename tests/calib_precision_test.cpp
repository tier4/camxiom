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

#include "camxiom/calib/convert.hpp"
#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "support/calib_test_fixtures.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace camxiom;
using namespace camxiom::calib;

namespace
{
// A 2 px prior pulls the noise-free pinhole fit about 16-18 px towards a
// reference 20 px away; 5 px leaves margin while still failing if the prior
// never reached the solve.
constexpr double kPriorSigmaPx = 2.0;
constexpr double kPriorShiftPx = 5.0;

CameraModel64 precisionModel()
{
  CameraInfo info;
  info.k = {
    621.1234567890123,
    0.1234567890123,
    319.9876543210987,
    0.0,
    609.9876543210987,
    239.1234567890123,
    0.0,
    0.0,
    1.0};
  info.distortion_model = "plumb_bob";
  return makeCameraModel64(info);
}

std::vector<CalibrationView> precisionViews(const CameraModel64 &model)
{
  const auto points = test::makeCheckerboard3D(9, 7, 0.04);
  std::vector<CalibrationView> views;
  for (int i = 0; i < 7; ++i)
  {
    const Eigen::Matrix3d rotation =
      test::rotMatX(0.1 * (i - 3)) * test::rotMatY(0.15 * (i % 3 - 1));
    const Eigen::Vector3d translation(-0.08 + 0.02 * i, -0.05, 0.7 + 0.05 * i);
    CalibrationView view;
    view.world_points = points;
    for (const auto &point : points)
    {
      const auto pixel = rayToPixel64(model, rotation * point + translation);
      EXPECT_EQ(pixel.status, StatusCode::OK);
      view.image_points.emplace_back(pixel.pixel.u, pixel.pixel.v);
    }
    views.push_back(std::move(view));
  }
  return views;
}

CalibrationOptions precisionOptions()
{
  CalibrationOptions options;
  options.image_width = 640;
  options.image_height = 480;
  options.max_iterations = 500;
  options.function_tolerance = 1e-14;
  options.parameter_tolerance = 1e-14;
  options.huber_loss_delta = 0.0;
  options.compute_per_point_residuals = true;
  options.estimate_uncertainty = true;
  return options;
}
}  // namespace

TEST(CalibPrecision, RetainsFittedParametersDiagnosticsAndWarmStart)
{
  const auto truth = precisionModel();
  const auto views = precisionViews(truth);
  auto seed = truth;
  seed.intrinsics.fx += 1.0;
  seed.intrinsics.fy -= 2.0;
  const auto flags = PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;
  const auto result = calibrate64(views, seed, flags, precisionOptions());
  ASSERT_EQ(result.status, StatusCode::OK);
  EXPECT_NEAR(result.camera_model.intrinsics.fx, truth.intrinsics.fx, 1e-8);
  EXPECT_NEAR(result.camera_model.intrinsics.fy, truth.intrinsics.fy, 1e-8);
  EXPECT_NEAR(result.camera_model.intrinsics.cx, truth.intrinsics.cx, 1e-8);
  EXPECT_NEAR(result.camera_model.intrinsics.cy, truth.intrinsics.cy, 1e-8);
  EXPECT_EQ(result.camera_model.intrinsics.skew, truth.intrinsics.skew);
  EXPECT_LT(result.rms_reprojection_error_px, 1e-9);
  EXPECT_EQ(result.reprojection_status, StatusCode::OK);
  EXPECT_EQ(result.valid_count, result.total_count);
  EXPECT_EQ(result.total_count, views.size() * views.front().world_points.size());
  ASSERT_EQ(result.per_view_valid_counts.size(), views.size());
  for (std::size_t i = 0; i < views.size(); ++i)
    EXPECT_EQ(result.per_view_valid_counts[i], views[i].world_points.size());
  EXPECT_TRUE(result.uncertainty_available);
  EXPECT_TRUE(result.observability_available);
  EXPECT_EQ(result.uncertainty_labels.size(), 4U);
  ASSERT_EQ(result.per_point_residuals.size(), views.size());
  std::vector<double> errors;
  double max_error = 0.0;
  const double rms = computeReprojErrors64(
    views, result.camera_model, result.per_view_rotations, result.per_view_translations, errors,
    max_error
  );
  EXPECT_EQ(rms, result.rms_reprojection_error_px);
  EXPECT_EQ(max_error, result.max_reprojection_error_px);
  const auto residuals = computePerPointResiduals64(
    views, result.camera_model, result.per_view_rotations, result.per_view_translations
  );
  for (std::size_t i = 0; i < views.size(); ++i)
    for (std::size_t j = 0; j < views[i].world_points.size(); ++j)
      EXPECT_EQ((residuals[i][j] - result.per_point_residuals[i][j]).squaredNorm(), 0.0);

  CamxiomCameraInfo info;
  ASSERT_EQ(exportCameraInfo64(result.camera_model, info), StatusCode::OK);
  const auto reloaded = makeCameraModel64(info);
  EXPECT_EQ(reloaded.intrinsics.fx, result.camera_model.intrinsics.fx);
  EXPECT_EQ(reloaded.intrinsics.skew, result.camera_model.intrinsics.skew);
  const auto repeated = calibrate64(views, reloaded, flags, precisionOptions());
  ASSERT_EQ(repeated.status, StatusCode::OK);
  EXPECT_LT(repeated.rms_reprojection_error_px, 1e-9);
  EXPECT_NEAR(repeated.camera_model.intrinsics.fx, truth.intrinsics.fx, 1e-8);
}

TEST(CalibPrecision, FailureIsAtomicAndDiagnosticsHandleMissingPoses)
{
  const auto model = precisionModel();
  auto result = calibrate64({}, model, PnpFlag::NONE, precisionOptions());
  EXPECT_EQ(result.status, StatusCode::INVALID_INPUT);
  EXPECT_TRUE(result.per_view_rotations.empty());
  EXPECT_FALSE(result.uncertainty_available);
  auto invalid = model;
  invalid.intrinsics.fx = std::numeric_limits<double>::quiet_NaN();
  result = calibrate64(precisionViews(model), invalid, PnpFlag::NONE, precisionOptions());
  EXPECT_EQ(result.status, StatusCode::INVALID_MODEL);
  EXPECT_TRUE(result.per_view_translations.empty());
  std::vector<double> errors{1.0};
  double max_error = 1.0;
  const double kInf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(computeReprojErrors64(precisionViews(model), model, {}, {}, errors, max_error), kInf);
  EXPECT_TRUE(errors.empty());
  EXPECT_EQ(max_error, kInf);
  EXPECT_TRUE(computePerPointResiduals64(precisionViews(model), model, {}, {}).empty());
}

TEST(CalibPrecision, PrincipalPointPriorMatchesFloatPath)
{
  // The prior and its diagnostics run through the shared calibration core, so
  // the double entry must reject the same settings and land where the float
  // entry lands when both start from the same (float-representable) seed.
  const auto truth = precisionModel();
  const auto views = precisionViews(truth);
  const auto flags = PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS;
  const CameraModel seed = fromCameraModel64(truth);
  const CameraModel64 seed64 = toCameraModel64(seed);

  auto options = precisionOptions();
  options.apply_principal_point_prior = true;
  options.principal_point_prior_reference_x = truth.intrinsics.cx + 20.0;
  options.principal_point_prior_reference_y = truth.intrinsics.cy - 20.0;
  options.principal_point_prior_sigma_px = 0.0;
  EXPECT_EQ(calibrate64(views, seed64, flags, options).status, StatusCode::INVALID_INPUT);

  const auto free64 = calibrate64(views, seed64, flags, precisionOptions());
  ASSERT_EQ(free64.status, StatusCode::OK);

  options.principal_point_prior_sigma_px = kPriorSigmaPx;
  const auto with = calibrate(views, seed, flags, options);
  const auto with64 = calibrate64(views, seed64, flags, options);
  ASSERT_EQ(with.status, StatusCode::OK);
  ASSERT_EQ(with64.status, StatusCode::OK);

  // The prior moved the double solution towards the reference ...
  EXPECT_GT(with64.camera_model.intrinsics.cx - free64.camera_model.intrinsics.cx, kPriorShiftPx);
  EXPECT_LT(with64.camera_model.intrinsics.cy - free64.camera_model.intrinsics.cy, -kPriorShiftPx);
  // ... to the float solution up to the float write-back.
  EXPECT_NEAR(with64.camera_model.intrinsics.cx, with.camera_model.intrinsics.cx, 1e-3);
  EXPECT_NEAR(with64.camera_model.intrinsics.cy, with.camera_model.intrinsics.cy, 1e-3);

  ASSERT_TRUE(with64.uncertainty_available);
  ASSERT_EQ(with64.uncertainty_labels, with.uncertainty_labels);
  for (std::size_t i = 0; i < with64.parameter_std.size(); ++i)
    EXPECT_NEAR(with64.parameter_std[i], with.parameter_std[i], 1e-3 * with.parameter_std[i]);
}

TEST(CalibPrecision, ConversionPreservesDoubleSourceAndDestination)
{
  const auto source = precisionModel();
  auto seed = source;
  seed.intrinsics.fx += 8.0;
  seed.intrinsics.fy -= 6.0;
  seed.intrinsics.cx -= 0.3;
  const auto result = convertCameraModel64(source, 640, 480, seed);
  ASSERT_EQ(result.status, StatusCode::OK);
  EXPECT_EQ(result.representable_point_count, result.used_point_count);
  EXPECT_LT(result.rms_fit_error_px, 1e-8);
  EXPECT_NEAR(result.camera_model.intrinsics.fx, source.intrinsics.fx, 1e-7);
  EXPECT_NEAR(result.camera_model.intrinsics.fy, source.intrinsics.fy, 1e-7);
  EXPECT_NEAR(result.camera_model.intrinsics.cx, source.intrinsics.cx, 1e-7);
  EXPECT_NEAR(result.camera_model.intrinsics.cy, source.intrinsics.cy, 1e-7);
  EXPECT_GT(
    std::abs(
      result.camera_model.intrinsics.fx -
      static_cast<double>(static_cast<float>(result.camera_model.intrinsics.fx))
    ),
    1e-6
  );
  const auto repeated = convertCameraModel64(result.camera_model, 640, 480, result.camera_model);
  EXPECT_EQ(repeated.status, StatusCode::OK);
  EXPECT_LT(repeated.rms_fit_error_px, 1e-8);
  EXPECT_EQ(convertCameraModel64(source, 0, 480, seed).status, StatusCode::INVALID_INPUT);
}

TEST(CalibPrecision, IterationCapKeepsBestEffortDoubleResult)
{
  const auto truth = precisionModel();
  auto seed = truth;
  seed.intrinsics.fx += 80.0;
  auto options = precisionOptions();
  options.max_iterations = 1;
  const auto result = calibrate64(
    precisionViews(truth), seed, PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS, options
  );
  EXPECT_EQ(result.status, StatusCode::NON_CONVERGED);
  EXPECT_EQ(result.per_view_rotations.size(), 7U);
  EXPECT_TRUE(std::isfinite(result.rms_reprojection_error_px));
  EXPECT_EQ(result.camera_model.intrinsics.skew, truth.intrinsics.skew);
}
