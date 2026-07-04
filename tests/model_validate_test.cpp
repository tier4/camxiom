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

// Validation-tier contract tests.
//
// Two invariants added after an external numerics audit demonstrated real
// contract breaks with hand-mutated models:
//
//  1. Coefficient-free distortion types (NONE and the ideal trig fisheye
//     mappings) must carry an all-zero coeffs array. The scalar kernels
//     ignore coeffs[] for these types while the fixed-width SIMD batch
//     kernels apply leading entries unconditionally, so a model with stale
//     non-zero coefficients used to validate OK yet project ~100 px apart
//     between rayToPixel and rayToPixelBatch.
//
//  2. A polynomial-fisheye theta_max must sit inside the distortion
//     polynomial's positive monotone range. A cap past the fold used to
//     validate OK (the endpoint check only sees theta_d(theta_max) > 0),
//     while rayToPixel emitted pixels from the folded region that
//     pixelToRay then rejected (OUT_OF_FOV) or resolved to the wrong branch.
//
// The enforcement is two-tier and these tests pin the tier split:
//   - query tier (rayToPixel / pixelToRay / batch / Jacobians, every call):
//     structural checks incl. the all-zero rule and a derivative-at-cap sign
//     test (catches single-fold caps);
//   - full tier (public validateCameraModel / validateCameraModel64,
//     ValidatedCameraModel::tryMake, builders, factories): query tier plus
//     the sampled monotone-range certification (catches interior folds that
//     recover before the cap, invisible to any endpoint test).

#include <camxiom/batch.hpp>
#include <camxiom/internal/constants.hpp>
#include <camxiom/model.hpp>
#include <camxiom/projection.hpp>
#include <camxiom/projection64.hpp>
#include <camxiom/types64.hpp>
#include <camxiom/validated_model.hpp>

#include <gtest/gtest.h>

namespace
{

using namespace camxiom;

CameraModel makeIdealFisheye(const DistortionModelType type)
{
  CameraModel m;
  m.intrinsics.fx = 300.0f;
  m.intrinsics.fy = 300.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(constants::kPiF) - 1e-4f;
  m.distortion.type = type;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makePinholeNone()
{
  CameraModel m;
  m.intrinsics.fx = 400.0f;
  m.intrinsics.fy = 400.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeKb4(
  const float k1, const float k2, const float k3, const float k4, const float theta_max
)
{
  CameraModel m;
  m.intrinsics.fx = 300.0f;
  m.intrinsics.fy = 300.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = theta_max;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs = {k1, k2, k3, k4};
  return m;
}

// ---------------------------------------------------------------------------
// 1. Coefficient-free types: all coeffs entries must be zero
// ---------------------------------------------------------------------------

TEST(ValidateCoeffFree, IdealTrigTypesWithZeroCoeffsStayValid)
{
  for (const DistortionModelType type :
       {DistortionModelType::EQUIDISTANT, DistortionModelType::EQUISOLID,
        DistortionModelType::STEREOGRAPHIC})
  {
    const CameraModel m = makeIdealFisheye(type);
    EXPECT_EQ(validateCameraModel(m), StatusCode::OK) << "type=" << static_cast<int>(type);
    EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::OK);
  }
  CameraModel ortho = makeIdealFisheye(DistortionModelType::ORTHOGRAPHIC);
  ortho.projection.theta_max = static_cast<float>(constants::kHalfPiF) - 1e-4f;
  EXPECT_EQ(validateCameraModel(ortho), StatusCode::OK);
}

TEST(ValidateCoeffFree, DeclaredCountWithAllZeroCoeffsStaysValid)
{
  // A count > 0 is legal for the trig types (e.g. a CameraInfo carrying
  // d = [0, 0, 0, 0]); only non-zero values are contradictory.
  CameraModel m = makeIdealFisheye(DistortionModelType::EQUIDISTANT);
  m.distortion.count = 4U;
  EXPECT_EQ(validateCameraModel(m), StatusCode::OK);
}

TEST(ValidateCoeffFree, EquidistantWithStaleKb4CoeffsIsRejectedEverywhere)
{
  // The audit scenario: an EQUIDISTANT model carrying stale KB4 coefficients
  // validated OK, then rayToPixel (ignores coeffs) and rayToPixelBatch (SIMD
  // applies coeffs[0..3]) disagreed by ~98 px with both reporting OK.
  CameraModel m = makeIdealFisheye(DistortionModelType::EQUIDISTANT);
  m.distortion.count = 4U;
  m.distortion.coeffs = {-0.02f, 0.005f, -0.001f, 0.0002f};

  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::INVALID_MODEL);

  // Query paths reject the model identically (scalar, inverse, batch).
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.05f, 1.0f)).status, StatusCode::INVALID_MODEL);
  EXPECT_EQ(pixelToRay(m, Pixel2{320.0f, 240.0f}).status, StatusCode::INVALID_MODEL);

  // The raw-pointer batch variant reports model rejection as zero projected
  // points with the status replicated per lane.
  const float rays[3] = {0.1f, 0.05f, 1.0f};
  float u = 0.0f;
  float v = 0.0f;
  StatusCode status = StatusCode::OK;
  EXPECT_EQ(rayToPixelBatch(m, rays, 1, &u, &v, &status), 0);
  EXPECT_EQ(status, StatusCode::INVALID_MODEL);
}

TEST(ValidateCoeffFree, NoneTypeWithGarbageCoeffsIsRejectedEverywhere)
{
  // Same class through the plane-space kernels: type NONE ignores coeffs in
  // the scalar path while the SIMD pinhole forward applies the radtan tail.
  CameraModel m = makePinholeNone();
  m.distortion.coeffs[0] = -0.2f;
  m.distortion.coeffs[1] = 0.05f;

  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::INVALID_MODEL);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.05f, 1.0f)).status, StatusCode::INVALID_MODEL);

  const float rays[3] = {0.1f, 0.05f, 1.0f};
  float u = 0.0f;
  float v = 0.0f;
  StatusCode status = StatusCode::OK;
  EXPECT_EQ(rayToPixelBatch(m, rays, 1, &u, &v, &status), 0);
  EXPECT_EQ(status, StatusCode::INVALID_MODEL);
}

// ---------------------------------------------------------------------------
// 2. Polynomial fisheye: theta_max must sit inside the monotone range
// ---------------------------------------------------------------------------

TEST(ValidateFoldingCap, SingleFoldCapIsRejectedOnEveryPath)
{
  // k1 = -0.15: theta_d folds at theta = sqrt(-1/(3 k1)) ~ 1.49. A cap of
  // 2.5 sits past the fold; theta_d(2.5) ~ 0.156 > 0 satisfies the old
  // endpoint-only check, but the slope at the cap is negative, so the
  // query-tier derivative test rejects the model everywhere: rayToPixel used
  // to emit pixels (e.g. theta = 2.2) that pixelToRay then answered with
  // OUT_OF_FOV or a silently wrong small-branch ray.
  const CameraModel m = makeKb4(-0.15f, 0.0f, 0.0f, 0.0f, 2.5f);

  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::INVALID_MODEL);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.05f, 1.0f)).status, StatusCode::INVALID_MODEL);
  EXPECT_EQ(pixelToRay(m, Pixel2{320.0f, 240.0f}).status, StatusCode::INVALID_MODEL);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(m).has_value());
}

TEST(ValidateFoldingCap, UpdateThetaMaxRepairsTheFoldingModel)
{
  // The library's own derivation shrinks the cap into the monotone range;
  // the repaired model validates and round-trips.
  CameraModel m = makeKb4(-0.15f, 0.0f, 0.0f, 0.0f, 2.5f);
  updateThetaMax(m);
  ASSERT_LT(m.projection.theta_max, 1.5f);
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  const Eigen::Vector3f ray = Eigen::Vector3f(0.3f, 0.2f, 1.0f).normalized();
  const PixelResult px = rayToPixel(m, ray);
  ASSERT_EQ(px.status, StatusCode::OK);
  const RayResult back = pixelToRay(m, px.pixel);
  ASSERT_EQ(back.status, StatusCode::OK);
  const float angle_err = std::acos(std::clamp(ray.dot(back.ray.direction), -1.0f, 1.0f));
  EXPECT_LT(angle_err, 1e-4f);

  // And the double mirror of the repaired model passes the double oracle
  // (guards the cross-precision slack of the monotone-range certification).
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::OK);
}

TEST(ValidateFoldingCap, InteriorFoldThatRecoversIsCaughtByTheOracleOnly)
{
  // theta_d' = 1 + 3 k1 theta^2 + 9 k4 theta^8 with k1 = -0.7, k4 = 0.2 dips
  // negative around theta ~ 0.8 and recovers well before the cap 1.3, where
  // both the endpoint value (~1.88) and the endpoint slope (~+12) look
  // healthy. No endpoint test can see this; only the full-tier scan does.
  const CameraModel m = makeKb4(-0.7f, 0.0f, 0.0f, 0.2f, 1.3f);

  // Full tier (the public oracle and everything built on it) rejects...
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::INVALID_MODEL);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(m).has_value());

  // ...while the per-point query guard (structural tier) still accepts: this
  // pins the documented tier split. Only validateCameraModel() == OK
  // certifies the forward/inverse round-trip.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.05f, 0.02f, 1.0f)).status, StatusCode::OK);

  // updateThetaMax repairs this model too.
  CameraModel repaired = m;
  updateThetaMax(repaired);
  EXPECT_EQ(validateCameraModel(repaired), StatusCode::OK);
}

TEST(ValidateFoldingCap, ZeroCoefficientSeedAtPiCapStaysValid)
{
  // The calibration seed pins theta_max at pi with all-zero coefficients
  // (identity polynomial, monotone everywhere). The monotone-range scan tops
  // out at pi - 1e-4; the acceptance slack must keep the seed valid.
  const CameraModel m = makeKb4(0.0f, 0.0f, 0.0f, 0.0f, static_cast<float>(constants::kPiF));
  EXPECT_EQ(validateCameraModel(m), StatusCode::OK);
  EXPECT_EQ(validateCameraModel64(toCameraModel64(m)), StatusCode::OK);
}

}  // namespace
