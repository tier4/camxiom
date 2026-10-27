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

#include <camxiom/internal/constants.hpp>
#include <camxiom/jacobian64.hpp>
#include <camxiom/model.hpp>
#include <camxiom/projection64.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace
{
using namespace camxiom;

CameraModel64 fisheye(
  DistortionModelType type = DistortionModelType::KB4,
  ProjectionModelType law = ProjectionModelType::EQUIDISTANT
)
{
  CameraModel64 model;
  model.intrinsics.fx = 300.123456789123;
  model.intrinsics.fy = 305.987654321987;
  model.intrinsics.cx = 320.0;
  model.intrinsics.cy = 240.0;
  model.projection.type = law;
  model.projection.theta_max = 1.0;
  model.distortion.type = type;
  if (type == DistortionModelType::KB8)
    model.distortion.count = 8;
  else if (type == DistortionModelType::KB4)
    model.distortion.count = 4;
  return model;
}

Eigen::Vector3d rayAt(double theta) { return {std::sin(theta), 0.0, std::cos(theta)}; }

void expectFailure(const CameraModel64 &model, StatusCode status = StatusCode::INVALID_MODEL)
{
  const auto domain = getFisheyeDomain64(model);
  EXPECT_EQ(domain.status, status);
  EXPECT_FALSE(domain.ok());
  EXPECT_EQ(domain.safe_theta_max, 0.0);
  EXPECT_EQ(domain.theta_limit, 0.0);
  EXPECT_EQ(domain.limit, FisheyeDomainLimit::UNKNOWN);
}

TEST(FisheyeDomain64, PolynomialLimitBracketsAnAnalyticFold)
{
  // theta_d = theta - 0.15 theta^3; theta_d' = 1 - 0.45 theta^2.
  const double fold = std::sqrt(1.0 / 0.45);
  for (auto type : {DistortionModelType::KB4, DistortionModelType::KB8})
  {
    auto model = fisheye(type);
    model.distortion.coeffs[0] = -0.15;
    const auto domain = getFisheyeDomain64(model);
    ASSERT_TRUE(domain.ok());
    EXPECT_EQ(domain.limit, FisheyeDomainLimit::MONOTONICITY_LIMIT);
    EXPECT_LT(domain.safe_theta_max, fold);
    EXPECT_GE(domain.theta_limit, fold);
    EXPECT_LT(domain.theta_limit - domain.safe_theta_max, 0.007);
    EXPECT_GT(1.0 - 0.45 * std::pow(domain.safe_theta_max, 2), 0.0);
    EXPECT_LE(1.0 - 0.45 * std::pow(domain.theta_limit, 2), 0.0);
    updateThetaMax64(model);
    EXPECT_EQ(model.projection.theta_max, domain.safe_theta_max);
    EXPECT_EQ(validateCameraModel64(model), StatusCode::OK);
  }
}

TEST(FisheyeDomain64, ReportsGeometricEndpointsForEveryIdealFamily)
{
  const std::pair<ProjectionModelType, DistortionModelType> families[] = {
    {ProjectionModelType::EQUIDISTANT, DistortionModelType::NONE},
    {ProjectionModelType::EQUISOLID, DistortionModelType::NONE},
    {ProjectionModelType::STEREOGRAPHIC, DistortionModelType::NONE},
    {ProjectionModelType::ORTHOGRAPHIC, DistortionModelType::NONE},
    {ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {ProjectionModelType::EQUIDISTANT, DistortionModelType::KB8}};
  for (const auto &[law, type] : families)
  {
    SCOPED_TRACE(std::string(toString(law)) + "+" + toString(type));
    auto model = fisheye(type, law);
    const auto domain = getFisheyeDomain64(model);
    ASSERT_TRUE(domain.ok());
    EXPECT_EQ(domain.limit, FisheyeDomainLimit::ANGULAR_LIMIT);
    const double endpoint =
      law == ProjectionModelType::ORTHOGRAPHIC ? constants::kHalfPi : constants::kPi;
    EXPECT_EQ(domain.theta_limit, endpoint);
    EXPECT_NEAR(domain.safe_theta_max, endpoint - 1e-4, 1e-7);
    updateThetaMax64(model);
    EXPECT_EQ(model.projection.theta_max, domain.safe_theta_max);
    EXPECT_EQ(validateCameraModel64(model), StatusCode::OK);
  }
}

TEST(FisheyeDomain64, IgnoresDeclaredCapAndLeavesInputUnchanged)
{
  auto model = fisheye();
  model.distortion.coeffs[0] = -0.15;
  const auto reference = getFisheyeDomain64(model);
  ASSERT_TRUE(reference.ok());
  for (double cap :
       {-1.0, 0.0, 0.4, 2.5, 8.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()})
  {
    model.projection.theta_max = cap;
    const auto before = model;
    const auto domain = getFisheyeDomain64(model);
    ASSERT_TRUE(domain.ok());
    EXPECT_EQ(domain.safe_theta_max, reference.safe_theta_max);
    EXPECT_EQ(domain.theta_limit, reference.theta_limit);
    EXPECT_EQ(domain.limit, reference.limit);
    if (std::isnan(cap))
      EXPECT_TRUE(std::isnan(model.projection.theta_max));
    else
      EXPECT_EQ(model.projection.theta_max, cap);
    EXPECT_EQ(model.projection.type, before.projection.type);
    EXPECT_EQ(model.projection.xi, before.projection.xi);
    EXPECT_EQ(model.projection.alpha, before.projection.alpha);
    EXPECT_EQ(model.projection.beta, before.projection.beta);
    EXPECT_EQ(model.intrinsics.fx, before.intrinsics.fx);
    EXPECT_EQ(model.intrinsics.fy, before.intrinsics.fy);
    EXPECT_EQ(model.intrinsics.cx, before.intrinsics.cx);
    EXPECT_EQ(model.intrinsics.cy, before.intrinsics.cy);
    EXPECT_EQ(model.intrinsics.skew, before.intrinsics.skew);
    EXPECT_EQ(model.distortion.coeffs, before.distortion.coeffs);
    EXPECT_EQ(model.distortion.type, before.distortion.type);
    EXPECT_EQ(model.distortion.count, before.distortion.count);
    EXPECT_EQ(model.distortion.is_rational, before.distortion.is_rational);
    EXPECT_EQ(model.distortion.has_thin_prism, before.distortion.has_thin_prism);
    EXPECT_EQ(model.distortion.has_tilt, before.distortion.has_tilt);
    EXPECT_EQ(model.distortion.tilt_matrix, before.distortion.tilt_matrix);
    EXPECT_EQ(model.distortion.inv_tilt_matrix, before.distortion.inv_tilt_matrix);
  }
}

TEST(FisheyeDomain64, UsesUpdatedCoefficientsAndTheKb8Tail)
{
  auto model = fisheye(DistortionModelType::KB8);
  const auto initial = getFisheyeDomain64(model);
  const double fold = 2.123456789123;
  model.distortion.coeffs[7] = -1.0 / (17.0 * std::pow(fold, 16));
  const auto changed = getFisheyeDomain64(model);
  ASSERT_TRUE(changed.ok());
  EXPECT_LT(changed.safe_theta_max, initial.safe_theta_max);
  EXPECT_LT(changed.safe_theta_max, fold);
  EXPECT_GE(changed.theta_limit, fold);
  EXPECT_EQ(changed.limit, FisheyeDomainLimit::MONOTONICITY_LIMIT);
  updateThetaMax64(model);
  EXPECT_EQ(model.projection.theta_max, changed.safe_theta_max);
}

TEST(FisheyeDomain64, DoesNotNarrowCoefficientsToFloat)
{
  // These two doubles map to the same float, but put the fold on opposite
  // sides of a scan sample. They must therefore produce different caps.
  const double scan_cap = double(constants::kPiF) - double(1e-4f);
  const double theta = 1e-3 + (scan_cap - 1e-3) * (200.0 / 512.0);
  const double critical = -1.0 / (3.0 * theta * theta);
  auto before = fisheye();
  auto after = before;
  before.distortion.coeffs[0] = critical * (1.0 + 1e-10);
  after.distortion.coeffs[0] = critical * (1.0 - 1e-10);
  ASSERT_EQ(float(before.distortion.coeffs[0]), float(after.distortion.coeffs[0]));
  const auto domain_before = getFisheyeDomain64(before);
  const auto domain_after = getFisheyeDomain64(after);
  ASSERT_TRUE(domain_before.ok());
  ASSERT_TRUE(domain_after.ok());
  EXPECT_LT(domain_before.safe_theta_max, theta);
  EXPECT_EQ(domain_before.theta_limit, theta);
  EXPECT_EQ(domain_after.safe_theta_max, theta);
}

TEST(FisheyeDomain64, DetectsAnInteriorFoldEvenWhenTheEndpointRecovers)
{
  auto model = fisheye();
  model.distortion.coeffs[0] = -0.7;
  model.distortion.coeffs[3] = 0.2;
  model.projection.theta_max = 1.3;
  const auto domain = getFisheyeDomain64(model);
  ASSERT_TRUE(domain.ok());
  EXPECT_EQ(domain.limit, FisheyeDomainLimit::MONOTONICITY_LIMIT);
  EXPECT_LT(domain.safe_theta_max, 0.8);
  const auto slope = [](double t) { return 1.0 - 2.1 * t * t + 1.8 * std::pow(t, 8); };
  EXPECT_GT(slope(domain.safe_theta_max), 0.0);
  EXPECT_LE(slope(domain.theta_limit), 0.0);
  EXPECT_GT(slope(model.projection.theta_max), 0.0);
}

TEST(FisheyeDomain64, DistinguishesNumericFailureFromAFold)
{
  auto model = fisheye(DistortionModelType::KB8);
  model.distortion.coeffs[7] = std::numeric_limits<double>::max() / 1000.0;
  const auto domain = getFisheyeDomain64(model);
  ASSERT_TRUE(domain.ok());
  EXPECT_EQ(domain.limit, FisheyeDomainLimit::NUMERIC_LIMIT);
  EXPECT_GT(domain.safe_theta_max, 0.0);
  EXPECT_LT(domain.safe_theta_max, domain.theta_limit);
  updateThetaMax64(model);
  EXPECT_EQ(model.projection.theta_max, domain.safe_theta_max);
  EXPECT_EQ(validateCameraModel64(model), StatusCode::OK);
}

TEST(FisheyeDomain64, RejectsModelsWithNoUsableStartingSample)
{
  auto model = fisheye();
  model.distortion.coeffs[0] = -1e8;  // Fold below the first 1e-3 rad sample.
  expectFailure(model);
  auto model32 = fromCameraModel64(model);
  updateThetaMax64(model);
  updateThetaMax(model32);
  EXPECT_EQ(model.projection.theta_max, 0.0);
  EXPECT_EQ(model32.projection.theta_max, 0.0f);
  EXPECT_EQ(validateCameraModel64(model), StatusCode::INVALID_MODEL);
  EXPECT_EQ(validateCameraModel(model32), StatusCode::INVALID_MODEL);
  model.distortion.coeffs[0] = std::numeric_limits<double>::max();
  expectFailure(model);  // The derivative overflows even at the start.
}

TEST(FisheyeDomain64, RejectsNonFisheyeAndMalformedModels)
{
  for (auto type :
       {ProjectionModelType::PINHOLE, ProjectionModelType::OMNIDIRECTIONAL,
        ProjectionModelType::DOUBLE_SPHERE, ProjectionModelType::EUCM,
        ProjectionModelType::UNKNOWN})
  {
    auto model = fisheye();
    model.projection.type = type;
    expectFailure(model);
  }
  auto model = fisheye();
  model.distortion.count = 255;
  expectFailure(model);
  model = fisheye();
  model.distortion.coeffs[0] = std::numeric_limits<double>::quiet_NaN();
  expectFailure(model);
  model = fisheye();
  model.distortion.is_rational = true;
  expectFailure(model);
  model = fisheye(DistortionModelType::NONE);
  model.distortion.coeffs[0] = 0.1;
  expectFailure(model);
  model = fisheye(DistortionModelType::KB4, ProjectionModelType::EQUISOLID);
  expectFailure(model);
  model = fisheye();
  model.intrinsics.fx = std::numeric_limits<double>::quiet_NaN();
  expectFailure(model, StatusCode::INVALID_INPUT);
  model = fisheye();
  model.intrinsics.fy = 0.0;
  expectFailure(model);
}

TEST(FisheyeDomain64, CallerCanApplyAnEvaluationMarginWithoutChangingAdoption)
{
  auto adoption = fisheye();
  adoption.distortion.coeffs[0] = -0.05;
  const auto domain = getFisheyeDomain64(adoption);
  ASSERT_TRUE(domain.ok());
  auto evaluation = adoption;
  evaluation.projection.theta_max = std::min(1.4, domain.safe_theta_max);
  EXPECT_EQ(rayToPixel64(adoption, rayAt(1.2)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel64(evaluation, rayAt(1.2)).status, StatusCode::OK);
  EXPECT_EQ(adoption.projection.theta_max, 1.0);
  evaluation.projection.theta_max = domain.safe_theta_max;
  EXPECT_EQ(rayToPixel64(evaluation, rayAt(domain.theta_limit)).status, StatusCode::OUT_OF_FOV);
  for (double fraction : {0.0, 0.5, 0.99, 1.0})
  {
    const auto ray = rayAt(fraction * domain.safe_theta_max);
    const auto pixel = rayToPixel64(evaluation, ray);
    ASSERT_TRUE(pixel.ok());
    const auto inverse = pixelToRay64(evaluation, pixel.pixel);
    ASSERT_TRUE(inverse.ok());
    EXPECT_LT((inverse.ray.direction - ray).norm(), 1e-9);
  }
}

TEST(FisheyeDomain64, ProjectionJacobianAgreesWithDifferencesInsideDerivedCap)
{
  for (auto type : {DistortionModelType::KB4, DistortionModelType::KB8})
  {
    auto model = fisheye(type);
    model.distortion.coeffs[0] = -0.15;
    const auto domain = getFisheyeDomain64(model);
    ASSERT_TRUE(domain.ok());
    model.projection.theta_max = domain.safe_theta_max;
    for (double fraction : {0.5, 0.995})
    {
      const Eigen::Vector3d ray = rayAt(fraction * domain.safe_theta_max);
      const auto analytic = rayToPixelWithJacobian64(model, ray);
      ASSERT_TRUE(analytic.ok());
      constexpr double h = 1e-6;
      for (int axis = 0; axis < 3; ++axis)
      {
        Eigen::Vector3d plus = ray;
        Eigen::Vector3d minus = ray;
        plus[axis] += h;
        minus[axis] -= h;
        const auto projected_plus = rayToPixel64(model, plus);
        const auto projected_minus = rayToPixel64(model, minus);
        ASSERT_TRUE(projected_plus.ok());
        ASSERT_TRUE(projected_minus.ok());
        EXPECT_NEAR(
          analytic.J(0, axis), (projected_plus.pixel.u - projected_minus.pixel.u) / (2 * h), 1e-6
        );
        EXPECT_NEAR(
          analytic.J(1, axis), (projected_plus.pixel.v - projected_minus.pixel.v) / (2 * h), 1e-6
        );
      }
    }
  }
}

}  // namespace
