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

// Guards the layer convenience / umbrella headers (roadmap #2).
//
// The three aggregates <camxiom/core.hpp>, <camxiom/calibration.hpp> and
// <camxiom/camxiom.hpp> are pure supersets of the existing per-header public
// API. This test locks in two guarantees:
//
//   1. Each umbrella parses on its own. core.hpp is included first (nothing
//      before it) to prove it needs no prerequisite include, and it must stay
//      Ceres-free so it compiles in a core-only build. This is registered as a
//      CORE test, so it is built even when CAMXIOM_WITH_CERES is OFF: if any
//      core header ever started pulling in Ceres, this TU would stop compiling.
//
//   2. Symbols are reachable through the umbrellas: core geometry (rayToPixel /
//      pixelToRay / validateCameraModel) is exercised at runtime, and the
//      calibration layer's plain option/flag/result structs are constructed to
//      prove they are visible. Those calibration types are header-only
//      aggregates, so this needs no camxiom_calib link and stays valid in a
//      core-only build.

#include "camxiom/calibration.hpp"  // calibration layer aggregate (Ceres-free surface)
#include "camxiom/camxiom.hpp"      // all-in-one aggregate
#include "camxiom/core.hpp"         // runtime geometry layer aggregate

#include <Eigen/Core>

#include <gtest/gtest.h>

namespace
{

camxiom::CameraModel makePinhole()
{
  camxiom::CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = camxiom::ProjectionModelType::PINHOLE;
  m.projection.theta_max = 1.5707f;  // unused for pinhole
  m.distortion.type = camxiom::DistortionModelType::NONE;
  m.distortion.space = camxiom::DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// (2) Core geometry reachable through the umbrella includes.
// ---------------------------------------------------------------------------

// Version macros are part of the public surface and must be visible through
// every umbrella (core.hpp pulls camxiom/version.hpp).
#ifndef CAMXIOM_VERSION_MAJOR
#error "camxiom/version.hpp not reachable through the umbrellas"
#endif
static_assert(
  CAMXIOM_VERSION_CODE ==
    CAMXIOM_MAKE_VERSION(CAMXIOM_VERSION_MAJOR, CAMXIOM_VERSION_MINOR, CAMXIOM_VERSION_PATCH),
  "CAMXIOM_VERSION_CODE must compose from the components"
);

TEST(ConvenienceHeaders, VersionMacrosVisible)
{
  EXPECT_STREQ(CAMXIOM_VERSION_STRING, "0.1.0");
  EXPECT_GE(CAMXIOM_VERSION_CODE, CAMXIOM_MAKE_VERSION(0, 1, 0));
}

TEST(ConvenienceHeaders, CoreGeometryReachableThroughUmbrella)
{
  const camxiom::CameraModel m = makePinhole();
  ASSERT_EQ(camxiom::validateCameraModel(m), camxiom::StatusCode::OK);

  const Eigen::Vector3f ray = Eigen::Vector3f(0.1f, 0.05f, 1.0f).normalized();

  const camxiom::PixelResult px = camxiom::rayToPixel(m, ray);
  ASSERT_EQ(px.status, camxiom::StatusCode::OK);

  const camxiom::RayResult back = camxiom::pixelToRay(m, px.pixel);
  ASSERT_EQ(back.status, camxiom::StatusCode::OK);

  const float dot = ray.dot(back.ray.direction.normalized());
  EXPECT_NEAR(dot, 1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// (2) Calibration-layer types visible through the umbrella, without requiring
//     the camxiom_calib link (all constructed types are header-only aggregates).
// ---------------------------------------------------------------------------

TEST(ConvenienceHeaders, CalibrationTypesVisibleWithoutCalibLink)
{
  camxiom::optimizer::PnpSolverOptions pnp_opts{};
  EXPECT_EQ(pnp_opts.cost_type, camxiom::optimizer::PnpCostType::ANALYTICAL);

  camxiom::optimizer::PnpFlag flags = camxiom::optimizer::PnpFlag::FIX_INTRINSICS;
  EXPECT_TRUE(camxiom::optimizer::hasFlag(flags, camxiom::optimizer::PnpFlag::FIX_FOCAL_LENGTHS));

  camxiom::calib::CalibrationOptions calib_opts{};
  calib_opts.image_width = 640;
  calib_opts.image_height = 480;
  EXPECT_EQ(calib_opts.image_width, 640);
  EXPECT_EQ(calib_opts.image_height, 480);

  const camxiom::calib::CalibrationResult calib_result{};
  EXPECT_EQ(calib_result.status, camxiom::StatusCode::OK);

  const camxiom::init::PlanarObservation obs{};
  EXPECT_EQ(static_cast<int>(obs.board_pts.cols()), 0);
}
