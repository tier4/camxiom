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

// Tests that actually PRODUCE DOMAIN_ERROR / NUMERIC_ERROR through the
// public projection API. Until this file those codes were only round-tripped
// through toString(): none of the real return sites in the per-model
// forward/inverse paths (rational denominators, EUCM denominators, overflow
// guards) had a test that reached one, so a branch regression — wrong
// epsilon, wrong code, or falling through to OK with a garbage pixel — was
// invisible. This is the same code shape as the D47 float/double boundary
// bug the roadmap tracks.
//
// Triggers are chosen to be EXACT in both precisions wherever possible:
// zero denominators assembled from exactly-representable inputs, so the
// tests cannot flake on rounding.

#include "camxiom/batch.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

using namespace camxiom;

namespace
{

CameraModel makePinhole(const float fx)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(constants::kHalfPi);
  return m;
}

// EUCM with alpha = 1/2, beta = 1: the projection denominator
// alpha*d + (1-alpha)*Z reduces to (||ray|| + Z) / 2, which vanishes
// EXACTLY (in float and double alike) for the straight-back ray (0,0,-1).
CameraModel makeHalfAlphaEucm()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.alpha = 0.5f;
  m.projection.beta = 1.0f;
  m.projection.theta_max = static_cast<float>(constants::kPi) - 1e-4f;
  return m;
}

// Pinhole + RATIONAL8 with k4 = -1 and every other coefficient zero: at the
// exactly-representable undistorted plane point (1, 0) — the ray (1,0,1) —
// the rational denominator 1 + k4*r^2 is exactly zero in both precisions.
CameraModel makeSingularRational()
{
  CameraModel m = makePinhole(500.0f);
  m.distortion.type = DistortionModelType::RATIONAL8;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 8U;
  m.distortion.coeffs[5] = -1.0f;  // k4
  m.distortion.is_rational = true;
  return m;
}

}  // namespace

TEST(ProjectionErrorPaths, EucmZeroDenominatorIsDomainError)
{
  const CameraModel m = makeHalfAlphaEucm();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  // The denominator check precedes the FOV checks, so this must surface as
  // DOMAIN_ERROR — not OUT_OF_FOV — in both precisions.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).status, StatusCode::DOMAIN_ERROR);

  const CameraModel64 m64 = toCameraModel64(m);
  ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.0, 0.0, -1.0)).status, StatusCode::DOMAIN_ERROR);
}

TEST(ProjectionErrorPaths, RationalZeroDenominatorIsDomainError)
{
  const CameraModel m = makeSingularRational();
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(1.0f, 0.0f, 1.0f)).status, StatusCode::DOMAIN_ERROR);

  const CameraModel64 m64 = toCameraModel64(m);
  ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(1.0, 0.0, 1.0)).status, StatusCode::DOMAIN_ERROR);

  // A nearby non-singular point must still project fine — the guard is a
  // point rejection, not a model rejection.
  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(0.5f, 0.0f, 1.0f)).status, StatusCode::OK);
}

TEST(ProjectionErrorPaths, DistortionOverflowIsNumericError)
{
  // k3 near FLT_MAX with r^2 = 100 overflows the float radial polynomial to
  // +inf inside the distortion evaluation. The input ray itself is finite
  // and inside the pinhole FOV, so this must surface as NUMERIC_ERROR — not
  // OK with an infinite pixel, and not INVALID_INPUT.
  CameraModel m = makePinhole(500.0f);
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5U;
  m.distortion.coeffs[4] = 3e38f;  // k3
  ASSERT_EQ(validateCameraModel(m), StatusCode::OK);

  EXPECT_EQ(rayToPixel(m, Eigen::Vector3f(10.0f, 0.0f, 1.0f)).status, StatusCode::NUMERIC_ERROR);

  // Same shape in double: the float-storable k3 needs a larger (still
  // finite, still in-FOV) plane radius before the double range overflows.
  const CameraModel64 m64 = toCameraModel64(m);
  ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK);
  EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(1e40, 0.0, 1.0)).status, StatusCode::NUMERIC_ERROR);
}

TEST(ProjectionErrorPaths, BatchForwardPropagatesTheScalarErrorStatuses)
{
  // The batch drivers must report these lanes with the same per-lane codes
  // as the scalar API, whether a lane went through a SIMD kernel or the
  // scalar fallback.
  struct Case
  {
    const char *name;
    CameraModel model;
    Eigen::Vector3f trigger;
  };
  const Case cases[] = {
    {"eucm-domain", makeHalfAlphaEucm(), {0.0f, 0.0f, -1.0f}},
    {"rational-domain", makeSingularRational(), {1.0f, 0.0f, 1.0f}},
  };

  for (const Case &c : cases)
  {
    // Mix the trigger lane between plain valid lanes so a SIMD group
    // contains both outcomes.
    constexpr int kN = 8;
    float rays_xyz[3 * kN];
    for (int i = 0; i < kN; ++i)
    {
      const Eigen::Vector3f ray = (i == 3) ? c.trigger : Eigen::Vector3f(0.1f, -0.05f, 1.0f);
      rays_xyz[3 * i + 0] = ray.x();
      rays_xyz[3 * i + 1] = ray.y();
      rays_xyz[3 * i + 2] = ray.z();
    }
    float u_out[kN];
    float v_out[kN];
    StatusCode statuses[kN];
    const int valid = rayToPixelBatch(c.model, rays_xyz, kN, u_out, v_out, statuses);

    int expected_valid = 0;
    for (int i = 0; i < kN; ++i)
    {
      const Eigen::Vector3f ray(rays_xyz[3 * i], rays_xyz[3 * i + 1], rays_xyz[3 * i + 2]);
      const PixelResult scalar = rayToPixel(c.model, ray);
      EXPECT_EQ(statuses[i], scalar.status) << c.name << " lane " << i;
      if (scalar.status == StatusCode::OK)
      {
        ++expected_valid;
      }
    }
    EXPECT_EQ(statuses[3], StatusCode::DOMAIN_ERROR)
      << c.name << ": the trigger lane did not produce DOMAIN_ERROR";
    EXPECT_EQ(valid, expected_valid) << c.name;
  }
}
