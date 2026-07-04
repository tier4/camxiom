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

// Property / fuzz invariants for the runtime projection API (roadmap Q1).
//
// The existing round-trip tests (projection_smoke_test) exercise a handful of
// hand-picked rays per model. This test instead pins *invariants* that must
// hold for MANY randomly-generated valid models and rays, so it catches
// regressions the fixed points miss. It is a CORE test (Ceres/ROS/OpenCV-free,
// always built) and uses a FIXED SEED, so a failure is fully reproducible and,
// once green, stays green (no wall-clock / nondeterministic flakiness).
//
// Invariants checked per (model, ray):
//   1. project ∘ unproject ≈ id : rayToPixel (OK + finite) then pixelToRay (OK)
//      recovers the ray direction (angle error under a per-family bound).
//   2. FOV-interior status OK    : rays sampled inside a conservative FOV cap
//      always project with StatusCode::OK.
//   3. positive-scale invariance : rayToPixel(s·ray), s>0, gives the same pixel
//      (projection depends only on direction, not magnitude).
//   4. determinism               : repeated rayToPixel is bit-for-bit identical.
//   5. pixel round-trip          : for pixels that land inside the image,
//      pixelToRay then rayToPixel returns the same pixel.
//
// Model generation stays inside valid, well-conditioned parameter ranges (every
// model asserts validateCameraModel == OK); the extreme FOV edge is covered by
// the dedicated smoke/model tests, whereas Q1 targets broad random coverage.

#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>

using namespace camxiom;

namespace
{

// Fixed seed => reproducible. Per-family streams are derived deterministically.
constexpr std::uint64_t kSeed = 0x9E3779B97F4A7C15ULL;
constexpr int kImgW = 640;
constexpr int kImgH = 480;
constexpr int kModelsPerFamily = 32;
constexpr int kRaysPerModel = 20;

const double kPi = static_cast<double>(camxiom::constants::kPi);
const double kRad2Deg = 180.0 / kPi;

float frand(std::mt19937_64 &rng, float lo, float hi)
{
  return std::uniform_real_distribution<float>(lo, hi)(rng);
}

float deg2rad(float deg) { return deg * static_cast<float>(kPi) / 180.0f; }

enum class Family {
  PinholeNone,
  PinholeRadtan5,
  FisheyeEquidistant,
  FisheyeKB4,
  Omni,
  DoubleSphere,
  Eucm
};

CameraModel makeCommonIntrinsics(std::mt19937_64 &rng, float fx_lo, float fx_hi)
{
  CameraModel m;
  const float fx = frand(rng, fx_lo, fx_hi);
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx * frand(rng, 0.98f, 1.02f);
  m.intrinsics.cx = 0.5f * kImgW + frand(rng, -15.0f, 15.0f);
  m.intrinsics.cy = 0.5f * kImgH + frand(rng, -15.0f, 15.0f);
  m.intrinsics.skew = 0.0f;
  return m;
}

CameraModel makeModel(Family fam, std::mt19937_64 &rng)
{
  switch (fam)
  {
    case Family::PinholeNone: {
      CameraModel m = makeCommonIntrinsics(rng, 300.0f, 800.0f);
      m.projection.type = ProjectionModelType::PINHOLE;
      m.distortion.type = DistortionModelType::NONE;
      m.distortion.space = DistortionSpace::NONE;
      m.distortion.count = 0U;
      return m;
    }
    case Family::PinholeRadtan5: {
      CameraModel m = makeCommonIntrinsics(rng, 300.0f, 800.0f);
      m.projection.type = ProjectionModelType::PINHOLE;
      m.distortion.type = DistortionModelType::RADTAN5;
      m.distortion.space = DistortionSpace::PLANE;
      m.distortion.coeffs[0] = frand(rng, -0.12f, 0.04f);      // k1
      m.distortion.coeffs[1] = frand(rng, -0.01f, 0.03f);      // k2
      m.distortion.coeffs[2] = frand(rng, -0.0012f, 0.0012f);  // p1
      m.distortion.coeffs[3] = frand(rng, -0.0012f, 0.0012f);  // p2
      m.distortion.coeffs[4] = frand(rng, -0.004f, 0.004f);    // k3
      m.distortion.count = 5U;
      return m;
    }
    case Family::FisheyeEquidistant: {
      CameraModel m = makeCommonIntrinsics(rng, 160.0f, 340.0f);
      m.projection.type = ProjectionModelType::FISHEYE_THETA;
      m.distortion.type = DistortionModelType::EQUIDISTANT;
      m.distortion.space = DistortionSpace::ANGLE;
      m.distortion.count = 0U;
      m.projection.theta_max = static_cast<float>(kPi) - 1e-4f;
      return m;
    }
    case Family::FisheyeKB4: {
      CameraModel m = makeCommonIntrinsics(rng, 160.0f, 340.0f);
      m.projection.type = ProjectionModelType::FISHEYE_THETA;
      m.distortion.type = DistortionModelType::KB4;
      m.distortion.space = DistortionSpace::ANGLE;
      m.distortion.coeffs[0] = frand(rng, -0.03f, 0.03f);
      m.distortion.coeffs[1] = frand(rng, -0.01f, 0.01f);
      m.distortion.coeffs[2] = frand(rng, -0.004f, 0.004f);
      m.distortion.coeffs[3] = frand(rng, -0.002f, 0.002f);
      m.distortion.count = 4U;
      m.projection.theta_max = static_cast<float>(kPi) - 1e-4f;
      updateThetaMax(m);  // recompute monotonic range from the KB4 coefficients
      return m;
    }
    case Family::Omni: {
      CameraModel m = makeCommonIntrinsics(rng, 250.0f, 500.0f);
      m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
      m.projection.xi = frand(rng, 0.3f, 1.1f);
      m.projection.theta_max = static_cast<float>(kPi);
      m.distortion.type = DistortionModelType::NONE;
      m.distortion.space = DistortionSpace::NONE;
      m.distortion.count = 0U;
      return m;
    }
    case Family::DoubleSphere: {
      CameraModel m = makeCommonIntrinsics(rng, 250.0f, 450.0f);
      m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
      m.projection.xi = frand(rng, -0.1f, 0.4f);
      m.projection.alpha = frand(rng, 0.4f, 0.65f);
      m.projection.theta_max = static_cast<float>(kPi);
      m.distortion.type = DistortionModelType::NONE;
      m.distortion.space = DistortionSpace::NONE;
      m.distortion.count = 0U;
      return m;
    }
    case Family::Eucm: {
      CameraModel m = makeCommonIntrinsics(rng, 250.0f, 450.0f);
      m.projection.type = ProjectionModelType::EUCM;
      m.projection.alpha = frand(rng, 0.4f, 0.7f);
      m.projection.beta = frand(rng, 0.9f, 1.3f);
      m.projection.theta_max = static_cast<float>(kPi);
      m.distortion.type = DistortionModelType::NONE;
      m.distortion.space = DistortionSpace::NONE;
      m.distortion.count = 0U;
      return m;
    }
  }
  return CameraModel{};
}

// Conservative FOV cap (radians) from which rays are sampled: comfortably
// inside each family's valid field of view so the forward projection is OK and
// the inverse is well conditioned.
float sampleThetaCap(Family fam, const CameraModel &m)
{
  switch (fam)
  {
    case Family::PinholeNone:
      return deg2rad(45.0f);
    case Family::PinholeRadtan5:
      return deg2rad(36.0f);
    case Family::FisheyeEquidistant:
      return std::min(deg2rad(120.0f), 0.85f * m.projection.theta_max);
    case Family::FisheyeKB4:
      return std::min(deg2rad(95.0f), 0.85f * m.projection.theta_max);
    case Family::Omni:
      return deg2rad(70.0f);
    case Family::DoubleSphere:
      return deg2rad(62.0f);
    case Family::Eucm:
      return deg2rad(58.0f);
  }
  return deg2rad(30.0f);
}

struct FamilyTolerance
{
  double angle_deg;  // max round-trip direction error
  double pixel;      // max pixel error (scale invariance + pixel round-trip)
};

void runFamily(Family fam, const char *name, FamilyTolerance tol)
{
  // Distinct but fixed stream per family.
  std::mt19937_64 rng(kSeed + static_cast<std::uint64_t>(fam) * 0x100000001B3ULL);

  double max_angle_deg = 0.0;
  double max_pixel_rt = 0.0;
  double max_scale_diff = 0.0;
  int checked = 0;
  int in_bounds = 0;

  for (int mi = 0; mi < kModelsPerFamily; ++mi)
  {
    const CameraModel m = makeModel(fam, rng);
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK)
      << name << " generated an invalid model #" << mi;

    const float cap = sampleThetaCap(fam, m);

    for (int ri = 0; ri < kRaysPerModel; ++ri)
    {
      const float theta = frand(rng, 0.0f, cap);
      const float phi = frand(rng, 0.0f, 2.0f * static_cast<float>(kPi));
      const float st = std::sin(theta);
      const Eigen::Vector3f ray =
        Eigen::Vector3f(st * std::cos(phi), st * std::sin(phi), std::cos(theta)).normalized();

      // (2) FOV-interior => OK, and (finite).
      const PixelResult px = rayToPixel(m, ray);
      ASSERT_EQ(px.status, StatusCode::OK)
        << name << " forward not OK: model#" << mi << " theta_deg=" << theta * kRad2Deg
        << " status=" << toString(px.status);
      ASSERT_TRUE(std::isfinite(px.pixel.u) && std::isfinite(px.pixel.v))
        << name << " produced a non-finite pixel";
      ++checked;

      // (4) determinism: identical call is bit-for-bit identical.
      const PixelResult px_again = rayToPixel(m, ray);
      EXPECT_EQ(px.pixel.u, px_again.pixel.u) << name << " rayToPixel u nondeterministic";
      EXPECT_EQ(px.pixel.v, px_again.pixel.v) << name << " rayToPixel v nondeterministic";

      // (3) positive-scale invariance.
      const float s = frand(rng, 0.2f, 4.0f);
      const PixelResult px_scaled = rayToPixel(m, (s * ray).eval());
      ASSERT_EQ(px_scaled.status, StatusCode::OK) << name << " scaled ray status changed";
      max_scale_diff = std::max(
        {max_scale_diff, std::abs(static_cast<double>(px_scaled.pixel.u - px.pixel.u)),
         std::abs(static_cast<double>(px_scaled.pixel.v - px.pixel.v))}
      );
      EXPECT_NEAR(px_scaled.pixel.u, px.pixel.u, tol.pixel) << name << " scale-variant u";
      EXPECT_NEAR(px_scaled.pixel.v, px.pixel.v, tol.pixel) << name << " scale-variant v";

      // (1) project ∘ unproject ≈ id.
      const RayResult back = pixelToRay(m, px.pixel);
      ASSERT_EQ(back.status, StatusCode::OK)
        << name << " inverse not OK: model#" << mi << " status=" << toString(back.status);
      const Eigen::Vector3f recovered = back.ray.direction.normalized();
      const double dot = std::min(1.0, std::max(-1.0, static_cast<double>(ray.dot(recovered))));
      const double angle_deg = std::acos(dot) * kRad2Deg;
      max_angle_deg = std::max(max_angle_deg, angle_deg);
      EXPECT_LT(angle_deg, tol.angle_deg)
        << name << " round-trip direction error too large: model#" << mi
        << " theta_deg=" << theta * kRad2Deg << " px=(" << px.pixel.u << "," << px.pixel.v << ")";

      // (5) pixel round-trip for pixels that land inside the image.
      if (px.pixel.u >= 0.0f && px.pixel.u <= static_cast<float>(kImgW) &&
          px.pixel.v >= 0.0f && px.pixel.v <= static_cast<float>(kImgH))
      {
        ++in_bounds;
        const PixelResult px_rt = rayToPixel(m, back.ray.direction);
        ASSERT_EQ(px_rt.status, StatusCode::OK) << name << " pixel round-trip forward not OK";
        max_pixel_rt = std::max(
          {max_pixel_rt, std::abs(static_cast<double>(px_rt.pixel.u - px.pixel.u)),
           std::abs(static_cast<double>(px_rt.pixel.v - px.pixel.v))}
        );
        EXPECT_NEAR(px_rt.pixel.u, px.pixel.u, tol.pixel) << name << " pixel round-trip u";
        EXPECT_NEAR(px_rt.pixel.v, px.pixel.v, tol.pixel) << name << " pixel round-trip v";
      }
    }
  }

  std::cout << "[Property] " << name << ": checks=" << checked << " inBounds=" << in_bounds
            << " maxAngleDeg=" << max_angle_deg << " maxPixelRT=" << max_pixel_rt
            << " maxScaleDiff=" << max_scale_diff << "  (tol angle=" << tol.angle_deg
            << "deg pixel=" << tol.pixel << ")\n";
}

}  // namespace

TEST(Property, PinholeNone) { runFamily(Family::PinholeNone, "pinhole/none", {0.05, 0.05}); }
TEST(Property, PinholeRadtan5)
{
  runFamily(Family::PinholeRadtan5, "pinhole/radtan5", {0.30, 0.50});
}
TEST(Property, FisheyeEquidistant)
{
  runFamily(Family::FisheyeEquidistant, "fisheye/equidistant", {0.30, 0.50});
}
TEST(Property, FisheyeKB4) { runFamily(Family::FisheyeKB4, "fisheye/kb4", {0.60, 1.00}); }
TEST(Property, Omnidirectional) { runFamily(Family::Omni, "omnidirectional", {0.30, 0.50}); }
TEST(Property, DoubleSphere) { runFamily(Family::DoubleSphere, "double_sphere", {0.30, 0.50}); }
TEST(Property, Eucm) { runFamily(Family::Eucm, "eucm", {0.30, 0.50}); }
