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

// Smoke test for rayToPixel / pixelToRay round-trip across all 5 models.
//
// For each model we set up a plausible CameraModel, project a handful of
// 3D rays to pixels, then unproject the pixels back to rays and confirm we
// land within a few 1e-3 of the original direction.
//
// This is a SMOKE test: it catches gross brokenness (NaN, wrong direction,
// dispatch errors). It does not validate numerical precision tightly.

#include "camxiom/batch.hpp"
#include "camxiom/compat.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/remap.hpp"
#include "camxiom/remap_kernel.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>

using namespace camxiom;

namespace
{

void expectRoundTrip(const CameraModel &model, const Eigen::Vector3f &input_ray, float tolerance)
{
  const Eigen::Vector3f ray = input_ray.normalized();

  const PixelResult px = rayToPixel(model, ray);
  ASSERT_EQ(px.status, StatusCode::OK)
    << "rayToPixel failed for model " << toString(model.projection.type)
    << " status=" << toString(px.status);
  ASSERT_TRUE(std::isfinite(px.pixel.u)) << "u is not finite";
  ASSERT_TRUE(std::isfinite(px.pixel.v)) << "v is not finite";

  const RayResult back = pixelToRay(model, px.pixel);
  ASSERT_EQ(back.status, StatusCode::OK)
    << "pixelToRay failed for model " << toString(model.projection.type)
    << " status=" << toString(back.status);

  const Eigen::Vector3f recovered = back.ray.direction.normalized();
  const float dot = ray.dot(recovered);
  EXPECT_NEAR(dot, 1.0f, tolerance)
    << "Direction mismatch for model " << toString(model.projection.type) << " ray=(" << ray.x()
    << "," << ray.y() << "," << ray.z() << ")"
    << " px=(" << px.pixel.u << "," << px.pixel.v << ")"
    << " recovered=(" << recovered.x() << "," << recovered.y() << "," << recovered.z() << ")";
}

CameraModel makePinhole(bool with_distortion)
{
  CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = 1.5707f;  // unused for pinhole

  if (with_distortion)
  {
    m.distortion.type = DistortionModelType::RADTAN5;
    m.distortion.space = DistortionSpace::PLANE;
    m.distortion.coeffs[0] = -0.05f;  // k1
    m.distortion.coeffs[1] = 0.01f;   // k2
    m.distortion.coeffs[2] = 0.0f;    // p1
    m.distortion.coeffs[3] = 0.0f;    // p2
    m.distortion.coeffs[4] = 0.0f;    // k3
    m.distortion.count = 5U;
  }
  else
  {
    m.distortion.type = DistortionModelType::NONE;
    m.distortion.space = DistortionSpace::NONE;
    m.distortion.count = 0U;
  }
  return m;
}

CameraModel makeFisheyeEquidistant()
{
  CameraModel m;
  m.intrinsics.fx = 280.0f;
  m.intrinsics.fy = 280.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi) - 1e-4f;
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeOmnidirectional()
{
  CameraModel m;
  m.intrinsics.fx = 400.0f;
  m.intrinsics.fy = 400.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 1.0f;  // Mei mirror parameter
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeDoubleSphere()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 0.5f;
  m.projection.alpha = 0.5f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeEucm()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.alpha = 0.6f;
  m.projection.beta = 1.1f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pinhole
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, PinholeNoDistortionRoundTrip)
{
  const CameraModel m = makePinhole(/*with_distortion=*/false);
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-5f);
  expectRoundTrip(m, Eigen::Vector3f(0.1f, 0.05f, 1.0f), 1e-5f);
  expectRoundTrip(m, Eigen::Vector3f(-0.2f, 0.15f, 1.0f), 1e-5f);
  expectRoundTrip(m, Eigen::Vector3f(0.3f, -0.2f, 1.0f), 1e-5f);
}

TEST(ProjectionSmoke, PinholeRadtan5RoundTrip)
{
  const CameraModel m = makePinhole(/*with_distortion=*/true);
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.1f, 0.05f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(-0.2f, 0.15f, 1.0f), 1e-4f);
}

TEST(ProjectionSmoke, PinholeBehindCamera)
{
  const CameraModel m = makePinhole(false);
  const PixelResult px = rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, -1.0f));
  EXPECT_EQ(px.status, StatusCode::BEHIND_CAMERA);
}

// ---------------------------------------------------------------------------
// Fisheye (equidistant)
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, FisheyeEquidistantRoundTrip)
{
  const CameraModel m = makeFisheyeEquidistant();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  // Forward axis
  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-4f);
  // Off-axis, moderate
  expectRoundTrip(m, Eigen::Vector3f(0.5f, 0.3f, 1.0f), 1e-4f);
  // Wide angle (close to the side: θ ≈ 80°)
  expectRoundTrip(m, Eigen::Vector3f(1.0f, 0.0f, 0.2f), 1e-3f);
  // >180° FOV is what makes us prefer this over OpenCV: side-and-back ray.
  expectRoundTrip(m, Eigen::Vector3f(1.0f, 0.0f, -0.2f), 5e-3f);
}

// ---------------------------------------------------------------------------
// Omnidirectional (Mei)
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, OmnidirectionalRoundTrip)
{
  const CameraModel m = makeOmnidirectional();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.2f, 0.1f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.6f, 0.2f, 0.5f), 1e-3f);
}

TEST(ProjectionSmoke, OmnidirectionalRejectsRaysBeyondInjectivityLimit)
{
  // For xi > 1 the projection denominator z + xi*|p| is positive for every
  // direction, so only the monotonicity check |p| + xi*z > 0 rejects rays
  // beyond theta = acos(-1/xi) (~131.8 deg here); without it back-facing
  // rays alias onto valid-looking pixels.
  CameraModel m = makeOmnidirectional();
  m.projection.xi = 1.5f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).status, StatusCode::OUT_OF_FOV);
  // theta = 140 deg, beyond the limit.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.643f, 0.0f, -0.766f)).status, StatusCode::OUT_OF_FOV);
  // theta = 120 deg, inside the limit.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.866f, 0.0f, -0.5f)).status, StatusCode::OK);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f)).status, StatusCode::OK);

  // Same contract in double precision.
  CameraModel64 m64;
  m64.intrinsics.fx = 400.0;
  m64.intrinsics.fy = 400.0;
  m64.intrinsics.cx = 320.0;
  m64.intrinsics.cy = 240.0;
  m64.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m64.projection.theta_max = camxiom::constants::kPi;
  m64.projection.xi = 1.5;
  m64.distortion.type = DistortionModelType::NONE;
  m64.distortion.space = DistortionSpace::NONE;
  m64.distortion.count = 0U;
  ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.0, 0.0, -1.0)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.866, 0.0, -0.5)).status, StatusCode::OK);

  // Batch path (SIMD kernels on x86) must agree with the scalar statuses.
  constexpr int kCount = 9;
  float rays[3 * kCount];
  for (int i = 0; i < kCount; ++i)
  {
    const bool behind = (i % 2) == 1;
    rays[3 * i + 0] = 0.1f * static_cast<float>(i % 3);
    rays[3 * i + 1] = 0.0f;
    rays[3 * i + 2] = behind ? -1.0f : 1.0f;
  }
  float u_out[kCount];
  float v_out[kCount];
  StatusCode statuses[kCount];
  const int valid = rayToPixelBatch(m, rays, kCount, u_out, v_out, statuses);
  int expected_valid = 0;
  for (int i = 0; i < kCount; ++i)
  {
    const Eigen::Vector3f ray(rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]);
    const PixelResult scalar = rayToPixel(m, ray);
    EXPECT_EQ(statuses[i], scalar.status) << "batch/scalar status mismatch at " << i;
    if (scalar.status == StatusCode::OK)
    {
      ++expected_valid;
      EXPECT_NEAR(u_out[i], scalar.pixel.u, 1e-2f);
      EXPECT_NEAR(v_out[i], scalar.pixel.v, 1e-2f);
    }
  }
  EXPECT_EQ(valid, expected_valid);
}

// ---------------------------------------------------------------------------
// Double Sphere
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, DoubleSphereRoundTrip)
{
  const CameraModel m = makeDoubleSphere();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.2f, 0.1f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.5f, 0.3f, 0.8f), 1e-3f);
}

TEST(ProjectionSmoke, DoubleSphereRejectsRaysBeyondBijectivityRegion)
{
  // Real DS fits typically have alpha > 0.5 (values below are the Usenko 2018
  // fisheye fit). There the projection denominator is positive for *every*
  // direction, so only the w2 bijectivity check (z > -w2*d1) rejects
  // back-facing rays; without it they alias onto pixels near the principal
  // point with status OK.
  CameraModel m = makeDoubleSphere();
  m.projection.xi = -0.2308f;
  m.projection.alpha = 0.5741f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.0f, -1.0f)).status, StatusCode::OUT_OF_FOV);

  // Forward and wide-angle rays inside the region (limit ~127 deg here)
  // still project.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f)).status, StatusCode::OK);
  EXPECT_EQ(
    rayToPixel(m, Eigen::Vector3f(0.87f, 0.0f, 0.5f)).status,
    StatusCode::OK
  );  // theta = 60 deg

  // Same contract in double precision.
  CameraModel64 m64;
  m64.intrinsics.fx = 350.0;
  m64.intrinsics.fy = 350.0;
  m64.intrinsics.cx = 320.0;
  m64.intrinsics.cy = 240.0;
  m64.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m64.projection.theta_max = camxiom::constants::kPi;
  m64.projection.xi = -0.2308;
  m64.projection.alpha = 0.5741;
  m64.distortion.type = DistortionModelType::NONE;
  m64.distortion.space = DistortionSpace::NONE;
  m64.distortion.count = 0U;
  ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.0, 0.0, -1.0)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.0, 0.0, 1.0)).status, StatusCode::OK);

  // Batch path (SIMD kernels on x86) must agree with the scalar statuses.
  // 9 rays = one full AVX2 group of 8 plus a scalar tail element.
  constexpr int kCount = 9;
  float rays[3 * kCount];
  for (int i = 0; i < kCount; ++i)
  {
    const bool behind = (i % 2) == 1;
    rays[3 * i + 0] = 0.1f * static_cast<float>(i % 3);
    rays[3 * i + 1] = 0.0f;
    rays[3 * i + 2] = behind ? -1.0f : 1.0f;
  }
  float u_out[kCount];
  float v_out[kCount];
  StatusCode statuses[kCount];
  const int valid = rayToPixelBatch(m, rays, kCount, u_out, v_out, statuses);
  int expected_valid = 0;
  for (int i = 0; i < kCount; ++i)
  {
    const Eigen::Vector3f ray(rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]);
    const PixelResult scalar = rayToPixel(m, ray);
    EXPECT_EQ(statuses[i], scalar.status) << "batch/scalar status mismatch at " << i;
    if (scalar.status == StatusCode::OK)
    {
      ++expected_valid;
      EXPECT_NEAR(u_out[i], scalar.pixel.u, 1e-2f);
      EXPECT_NEAR(v_out[i], scalar.pixel.v, 1e-2f);
    }
  }
  EXPECT_EQ(valid, expected_valid);
}

// ---------------------------------------------------------------------------
// EUCM
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, BatchFallsBackToScalarPixelsOnEpsilonBoundary)
{
  // The SIMD kernels reject with wider epsilons than the scalar path (e.g.
  // fisheye rejects ray_norm_sq <= 1e-7 where scalar accepts down to 1e-8).
  // A ray in that gap is SIMD-invalid but scalar-valid: the SSE drivers used
  // to fall back for the status only, returning pixel (0,0) with status OK.
  // Batch output must match the scalar path per point — pixel included.
  // (Exercises the SIMD path on x86 CI; scalar fallback is trivially
  // consistent elsewhere.)
  const CameraModel m = makeFisheyeEquidistant();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  constexpr int kCount = 9;  // AVX2 body + SSE tail coverage
  float rays[3 * kCount];
  for (int i = 0; i < kCount; ++i)
  {
    rays[3 * i + 0] = 0.1f;
    rays[3 * i + 1] = 0.05f;
    rays[3 * i + 2] = 1.0f;
  }
  // Boundary lanes in both the AVX2 body and the SSE tail: norm^2 = 2e-8.
  for (const int lane : {1, 8})
  {
    rays[3 * lane + 0] = 1e-4f;
    rays[3 * lane + 1] = 0.0f;
    rays[3 * lane + 2] = 1e-4f;
  }

  float u_out[kCount];
  float v_out[kCount];
  StatusCode statuses[kCount];
  const int valid = rayToPixelBatch(m, rays, kCount, u_out, v_out, statuses);

  int expected_valid = 0;
  for (int i = 0; i < kCount; ++i)
  {
    const Eigen::Vector3f ray(rays[3 * i], rays[3 * i + 1], rays[3 * i + 2]);
    const PixelResult scalar = rayToPixel(m, ray);
    EXPECT_EQ(statuses[i], scalar.status) << "status mismatch at " << i;
    if (scalar.status == StatusCode::OK)
    {
      ++expected_valid;
      EXPECT_NEAR(u_out[i], scalar.pixel.u, 1e-2f) << "u mismatch at " << i;
      EXPECT_NEAR(v_out[i], scalar.pixel.v, 1e-2f) << "v mismatch at " << i;
    }
  }
  EXPECT_EQ(valid, expected_valid);
}

TEST(ProjectionSmoke, BatchInverseHonoursNonDefaultSolverOptions)
{
  // The SIMD inverse runs a fixed 10-iteration / 1e-6 schedule and only
  // guards accepted lanes with a 1 px round-trip check, so caller-supplied
  // tighter tolerances used to be silently degraded to that accuracy. With
  // non-default options the batch must match the scalar solve per point
  // (exercised by the x86 SIMD CI jobs).
  const CameraModel m = makePinhole(/*with_distortion=*/true);

  camxiom::SolverOptions strict;
  strict.max_iterations = 50;
  strict.residual_tolerance = 1e-9f;
  strict.step_tolerance = 1e-10f;

  constexpr int kCount = 9;
  float u_in[kCount];
  float v_in[kCount];
  for (int i = 0; i < kCount; ++i)
  {
    u_in[i] = 60.0f + 55.0f * static_cast<float>(i);
    v_in[i] = 45.0f + 40.0f * static_cast<float>(i);
  }
  float dirs[3 * kCount];
  StatusCode statuses[kCount];
  const int valid = pixelToRayBatch(m, u_in, v_in, kCount, dirs, statuses, strict);

  int expected_valid = 0;
  for (int i = 0; i < kCount; ++i)
  {
    const RayResult scalar = pixelToRay(m, Pixel2{u_in[i], v_in[i]}, strict);
    ASSERT_EQ(statuses[i], scalar.status) << "status mismatch at " << i;
    if (scalar.status == StatusCode::OK)
    {
      ++expected_valid;
      const Eigen::Vector3f batch_dir(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
      EXPECT_LT((batch_dir - scalar.ray.direction).norm(), 1e-6f) << "direction mismatch at " << i;
    }
  }
  EXPECT_EQ(valid, expected_valid);
}

TEST(ProjectionSmoke, Kb8BatchInverseMatchesScalar)
{
  // The SIMD polynomial-Newton inverse carries 4 coefficients only. For KB8
  // it used to drop k5..k8 silently, and whenever the resulting theta bias
  // stayed under the 1 px round-trip guard the batch returned it as OK —
  // systematically different from the scalar solve. KB8 batches must route
  // through the exact scalar path (exercised by the x86 SIMD CI jobs).
  CameraModel m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::KB8;
  m.distortion.count = 8U;
  m.distortion.coeffs[0] = -0.012f;
  m.distortion.coeffs[1] = 0.004f;
  m.distortion.coeffs[2] = -0.0007f;
  m.distortion.coeffs[3] = 0.0001f;
  m.distortion.coeffs[4] = 0.001f;  // k5..k8: invisible to the 4-coeff SIMD
  m.distortion.coeffs[5] = -0.0005f;
  m.distortion.coeffs[6] = 0.0002f;
  m.distortion.coeffs[7] = -0.0001f;
  // These high-order terms fold theta_d far below zero at the equidistant
  // default cap (theta_d(pi - 1e-4) ~ -2.4e4): with that cap every pixel was
  // OUT_OF_FOV, this test degenerated to comparing empty results, and the
  // endpoint check in validateCameraModel now rejects the cap outright. Cap
  // inside the polynomial's positive monotone range (peak near 1.6 rad) with
  // room for the widest test pixel below (r_d ~ 0.86 -> theta ~ 0.88 rad).
  m.projection.theta_max = 1.4f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  constexpr int kCount = 9;
  float u_in[kCount];
  float v_in[kCount];
  for (int i = 0; i < kCount; ++i)
  {
    u_in[i] = 320.0f + 28.0f * static_cast<float>(i);
    v_in[i] = 240.0f + 11.0f * static_cast<float>(i);
  }
  float dirs[3 * kCount];
  StatusCode statuses[kCount];
  const int valid = pixelToRayBatch(m, u_in, v_in, kCount, dirs, statuses);

  int expected_valid = 0;
  for (int i = 0; i < kCount; ++i)
  {
    const RayResult scalar = pixelToRay(m, Pixel2{u_in[i], v_in[i]});
    ASSERT_EQ(statuses[i], scalar.status) << "status mismatch at " << i;
    if (scalar.status == StatusCode::OK)
    {
      ++expected_valid;
      const Eigen::Vector3f batch_dir(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
      // The batch must be the same exact solve, not a 4-coefficient
      // approximation that merely stays under the 1 px guard. The dropped
      // k5..k8 terms bias the direction by ~3e-4 rad here; the exact
      // scalar route reproduces it bit-for-bit (component tolerance well
      // below the bias, above float rounding).
      EXPECT_LT((batch_dir - scalar.ray.direction).norm(), 1e-5f) << "direction mismatch at " << i;
    }
  }
  // Guard against the empty-comparison degeneration above: the direction
  // parity loop must actually run for the test to mean anything.
  EXPECT_GT(expected_valid, 0);
  EXPECT_EQ(valid, expected_valid);
}

TEST(ProjectionSmoke, EucmRejectsBehindCameraAtAlphaZero)
{
  // At alpha = 0 the EUCM denominator is plain Z, which entered the
  // validity logic only through abs(): the old alpha-gated FOV check let
  // points behind the camera project onto mirrored pixels with status OK.
  // The w-check at alpha = 0 degenerates to z > 0 (the pinhole limit).
  CameraModel m = makeEucm();
  m.projection.alpha = 0.0f;
  m.projection.beta = 1.0f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.0f, -1.0f)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.1f, 0.0f, 1.0f)).status, StatusCode::OK);
}

TEST(ProjectionSmoke, ThetaMaxRespectedByWideAngleModels)
{
  // types.hpp: wide-angle models accept rays up to their theta_max. Only
  // fisheye used to enforce this; omni / DS / EUCM ignored the field
  // entirely. theta = 70 deg must be rejected under a 1.0 rad (57.3 deg)
  // cap and accepted at the default cap of pi.
  const Eigen::Vector3f inside(0.643f, 0.0f, 0.766f);    // theta = 40 deg
  const Eigen::Vector3f outside(0.9397f, 0.0f, 0.342f);  // theta = 70 deg

  CameraModel models[3] = {makeOmnidirectional(), makeDoubleSphere(), makeEucm()};
  for (CameraModel &m : models)
  {
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
    // Default cap (pi): both rays project.
    EXPECT_EQ(rayToPixel(m, inside).status, StatusCode::OK) << toString(m.projection.type);
    EXPECT_EQ(rayToPixel(m, outside).status, StatusCode::OK) << toString(m.projection.type);

    m.projection.theta_max = 1.0f;
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
    EXPECT_EQ(rayToPixel(m, inside).status, StatusCode::OK) << toString(m.projection.type);
    const PixelResult px = rayToPixel(m, outside);
    EXPECT_EQ(px.status, StatusCode::OUT_OF_FOV)
      << toString(m.projection.type) << " must respect theta_max";
    // Round-trip consistency: a ray inside the cap still unprojects OK.
    const PixelResult px_in = rayToPixel(m, inside);
    ASSERT_EQ(px_in.status, StatusCode::OK);
    EXPECT_EQ(pixelToRay(m, px_in.pixel).status, StatusCode::OK) << toString(m.projection.type);
  }
}

TEST(ProjectionSmoke, EucmRoundTrip)
{
  const CameraModel m = makeEucm();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  expectRoundTrip(m, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.2f, 0.1f, 1.0f), 1e-4f);
  expectRoundTrip(m, Eigen::Vector3f(0.4f, 0.25f, 0.85f), 1e-3f);
}

// ---------------------------------------------------------------------------
// Validation rejects malformed models
// ---------------------------------------------------------------------------

TEST(ProjectionSmoke, ValidationRejectsUnknownModel)
{
  CameraModel m{};
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(ProjectionSmoke, ValidationRejectsZeroFocal)
{
  CameraModel m = makePinhole(false);
  m.intrinsics.fx = 0.0f;
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(ProjectionSmoke, ValidationRejectsCoefficientsBeyondCount)
{
  // RADTAN4 shares the RADTAN5 kernels, which read coeffs[4] as k3; a
  // non-zero entry beyond the declared count used to be silently applied by
  // the runtime paths while the calibration template zero-fills it — the
  // same model projected differently per path. The validator now enforces
  // the types.hpp contract that entries beyond `count` are meaningless.
  CameraModel m = makePinhole(/*with_distortion=*/true);
  m.distortion.type = DistortionModelType::RADTAN4;
  m.distortion.count = 4U;
  m.distortion.coeffs[4] = 0.5f;  // beyond the declared count
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  m.distortion.coeffs[4] = 0.0f;
  EXPECT_EQ(validateCameraModel(m), StatusCode::OK);
}

TEST(ProjectionSmoke, ValidationRejectsOrthographicBeyondHalfPi)
{
  // theta_d = sin(theta) folds over past pi/2: the inverse bracket assumes
  // monotonicity, so a wider cap silently turned solvable pixels into
  // OUT_OF_FOV and broke round-trips.
  CameraModel m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::ORTHOGRAPHIC;
  m.projection.theta_max = 2.0f;
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  m.projection.theta_max = static_cast<float>(camxiom::constants::kHalfPi) - 1e-4f;
  EXPECT_EQ(validateCameraModel(m), StatusCode::OK);
}

TEST(ProjectionSmoke, ValidationRejectsStereographicAtThePiPole)
{
  // theta_d = 2*tan(theta/2) has a pole at true double pi. Float pi (kPiF)
  // sits ~8.7e-8 PAST that pole, so with theta_max == kPiF the inverse
  // bracket's upper endpoint is a huge NEGATIVE number and every pixel —
  // including the image centre — came back OUT_OF_FOV while the model still
  // validated OK. The endpoint check must reject it.
  CameraModel m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::STEREOGRAPHIC;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);

  // Stopping short of the pole (detail::defaultFisheyeThetaMax's cap) is a
  // usable model: it validates AND the centre pixel unprojects.
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi) - 1e-4f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
  const RayResult centre = pixelToRay(m, Pixel2{320.0f, 240.0f});
  EXPECT_EQ(centre.status, StatusCode::OK);

  // Same pole crossing at double width: any theta_max in (true pi, kPiF]
  // passes the D47 pi bound yet breaks the whole FOV. kPiF widened to double
  // is such a value.
  CameraModel64 m64 = toCameraModel64(m);
  m64.projection.theta_max = static_cast<double>(camxiom::constants::kPiF);
  EXPECT_EQ(validateCameraModel64(m64), StatusCode::INVALID_MODEL);
  m64.projection.theta_max = camxiom::constants::kPi - 1e-4;
  EXPECT_EQ(validateCameraModel64(m64), StatusCode::OK);
}

TEST(ProjectionSmoke, ValidationRejectsPolynomialFisheyeFoldedBelowThetaMax)
{
  // A KB4 polynomial with a strongly negative k1 drives theta_d(theta_max)
  // below zero for a hand-set cap (theta_d(1.5) = 1.5 - 0.5*1.5^3 ~ -0.19):
  // the inverse bracket then rejects the entire image. Models built through
  // updateThetaMax never hit this (estimateSafeThetaMaxForPolynomialFisheye
  // stays inside the monotone range); a directly-authored cap must be
  // rejected at validation instead of failing pixel-by-pixel.
  CameraModel m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.count = 4U;
  m.distortion.coeffs.fill(0.0f);
  m.distortion.coeffs[0] = -0.5f;
  m.projection.theta_max = 1.5f;
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);

  // Inside the positive monotone range the same polynomial is fine
  // (theta_d(0.8) = 0.8 - 0.5*0.8^3 = 0.544).
  m.projection.theta_max = 0.8f;
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
  EXPECT_EQ(pixelToRay(m, Pixel2{320.0f, 240.0f}).status, StatusCode::OK);
}

TEST(ProjectionSmoke, ValidationRejectsPinholeWithAngleDistortion)
{
  CameraModel m = makePinhole(false);
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(ProjectionSmoke, IsRayProjectable)
{
  const CameraModel pinhole = makePinhole(false);

  // Forward ray: projectable; behind-camera: not.
  EXPECT_TRUE(camxiom::isRayProjectable(pinhole, Eigen::Vector3f(0.0f, 0.0f, 1.0f)));
  EXPECT_FALSE(camxiom::isRayProjectable(pinhole, Eigen::Vector3f(0.0f, 0.0f, -1.0f)));

  // Bounded overload: the optical axis lands at (cx, cy), inside 640x480;
  // a 1x1 image does not contain (cx, cy) unless cx/cy < 1.
  EXPECT_TRUE(camxiom::isRayProjectable(pinhole, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 640, 480));
  EXPECT_FALSE(camxiom::isRayProjectable(pinhole, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 1, 1));

  // A steep ray projects OK for the model (inside FOV cone) but far outside a
  // small image: model-domain true, bounds false.
  const Eigen::Vector3f steep(0.9f, 0.0f, 1.0f);
  ASSERT_TRUE(camxiom::isRayProjectable(pinhole, steep));
  EXPECT_FALSE(camxiom::isRayProjectable(pinhole, steep, 64, 64));

  // Fisheye: a ray beyond theta_max is rejected by the model domain.
  CameraModel fisheye = makeFisheyeEquidistant();
  fisheye.projection.theta_max = 1.0f;  // narrow the acceptance cone
  const float inside = 0.8f;
  const float outside = 1.2f;
  EXPECT_TRUE(
    camxiom::isRayProjectable(fisheye, Eigen::Vector3f(std::sin(inside), 0.0f, std::cos(inside)))
  );
  EXPECT_FALSE(
    camxiom::isRayProjectable(fisheye, Eigen::Vector3f(std::sin(outside), 0.0f, std::cos(outside)))
  );
}

TEST(ProjectionSmoke, ToStringCoversNewEnums)
{
  EXPECT_STREQ(camxiom::toString(camxiom::DistortionSpace::PLANE), "plane");
  EXPECT_STREQ(camxiom::toString(camxiom::DistortionSpace::ANGLE), "angle");
  EXPECT_STREQ(camxiom::toString(camxiom::RectifiedProjectionType::CYLINDRICAL), "cylindrical");
  EXPECT_STREQ(
    camxiom::toString(camxiom::RectifyFitPolicy::ALL_SOURCE_CONTAINED), "all_source_contained"
  );
  EXPECT_STREQ(
    camxiom::toString(camxiom::InvalidPixelPolicy::WRITE_NEGATIVE_ONE), "write_negative_one"
  );
  EXPECT_STREQ(camxiom::toString(camxiom::InterpolationMode::BILINEAR), "bilinear");
  EXPECT_STREQ(
    camxiom::toString(camxiom::PinholeCompatProfile::OPENCV_CALIB3D_D5), "opencv_calib3d_d5"
  );
  EXPECT_STREQ(
    camxiom::toString(camxiom::FisheyeCompatProfile::OPENCV_FISHEYE_D4), "opencv_fisheye_d4"
  );
  EXPECT_STREQ(
    camxiom::toString(camxiom::OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4), "opencv_omnidir_d4"
  );
  EXPECT_STREQ(camxiom::toString(camxiom::DoubleSphereCompatProfile::BASALT_D0), "basalt_d0");
  EXPECT_STREQ(camxiom::toString(camxiom::EucmCompatProfile::KALIBR_D0), "kalibr_d0");
}
