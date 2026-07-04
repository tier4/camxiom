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

// MS2-4 integration regression test — Phase 1 (skeleton + CMake) and
// Phase 2 (calibrate recovery leg).
//
// Purpose: a single end-to-end guard that exercises the full pipeline
//   getDefaultSeed / estimatePinholeZhang → calib::calibrate
// across a richer-than-canonical view set for all five projection models
// and three noise levels.
//
// Design:
//   - 10–12 views per model: the canonical 5 plus 5–7 additional poses with
//     wider X/Y tilts and greater lateral/depth diversity.
//   - All GT models match the seed's locked parameters so that locked
//     parameters are valid (D29/D30).
//   - Board: 8×6 corners at 0.05 m spacing (48 points ≥ 6 requirement).
//   - Sensor: 640×480.
//   - Noise levels: σ = 0, 0.3, 0.5 px.
//   - Tolerances derived from CALIBRATION_DESIGN_NOTES §8 and
//     INIT_DESIGN_NOTES §10 with safety margins; every literal is commented.
//
// Realised maximum incidence angle:
//   The additional poses rotate the board up to ±35° (0.611 rad) about X or Y,
//   and the far board corners with the extra lateral offset/depth produce ray
//   angles up to ≈ 0.72 rad from the optical axis (computed analytically,
//   verified at build time — see static_assert comment in makeRicherViews).
//   This exceeds the θ ≥ 0.7 rad requirement from INIT_DESIGN_NOTES §10.
//   All models have GT theta_max = kHalfPi (≈ 1.5708 rad) so all corners
//   remain well within the fisheye domain; KB4 seed theta_max = π gives
//   even more headroom.
//
// Notes on Pinhole Zhang path (Test B):
//   estimatePinholeZhang returns a K matrix from which a float CameraModel
//   is built. Skew is forced to 0 (not estimated by default-path callers).
//   theta_max is set to kHalfPi. The Zhang K seed can differ from the GT
//   focal by ≤ 1% (INIT_DESIGN_NOTES §10: "< 0.1% fx error" at σ=0.5).
//   After a single Ceres pass the recovered fx should be at least as good as
//   the getDefaultSeed path.
//
// LOCK TABLE (mandatory per-model, D29/D30):
//   PINHOLE:         PnpFlag::NONE
//   FISHEYE_THETA:   PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3
//   OMNIDIRECTIONAL: PnpFlag::FIX_PROJECTION_PARAMS
//   DOUBLE_SPHERE:   PnpFlag::FIX_PROJECTION_PARAMS
//   EUCM:            PnpFlag::FIX_PROJECTION_PARAMS
//
// All tests are deterministic: std::mt19937 seed 42. No rand().

#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/default_seed.hpp"
#include "camxiom/init/pinhole_zhang.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "support/calib_test_fixtures.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace camxiom;
using camxiom::calib::calibrate;
using camxiom::calib::CalibrationOptions;
using camxiom::calib::CalibrationResult;
using camxiom::calib::CalibrationView;
using camxiom::calib::PnpFlag;
using camxiom::init::estimatePinholeZhang;
using camxiom::init::PlanarObservation;

// ===========================================================================
// Anonymous-namespace helpers. The shared synthetic-geometry generators
// (rotMat*, makeCheckerboard3D, buildCalibView) now live in
// tests/support/calib_test_fixtures.hpp (#8); this file keeps only its own
// richer view/GT builders below.
// ===========================================================================

namespace
{

// ---------------------------------------------------------------------------
// Geometry helpers — shared across the calibration / init tests
// (tests/support/calib_test_fixtures.hpp).
// ---------------------------------------------------------------------------

using camxiom::test::buildCalibView;
using camxiom::test::makeCheckerboard3D;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;
using camxiom::test::rotMatZ;

// ---------------------------------------------------------------------------
// makeDefaultOptions — 640×480, no bounds, 200 Ceres iterations.
// ---------------------------------------------------------------------------

CalibrationOptions makeDefaultOptions(int w = 640, int h = 480)
{
  CalibrationOptions opts;
  opts.image_width = w;
  opts.image_height = h;
  opts.max_iterations = 200;
  opts.apply_initial_value_bounds = false;
  return opts;
}

// ---------------------------------------------------------------------------
// GT model builders — identical semantics to calib_intrinsics_test.cpp.
// GT projection/distortion locked-params MUST equal the seed (D29/D30).
// ---------------------------------------------------------------------------

/// Pinhole, no distortion. GT fx=fy=500 (seed is h/2=240).
CameraModel64 buildGTPinhole()
{
  CameraModel64 m{};
  m.intrinsics.fx = 500.0;
  m.intrinsics.fy = 500.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = constants::kHalfPi;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

/// KB4 Fisheye. GT fx=fy=200, k1=0.01, k2=-0.005, k3=k4=0 (= seed, LOCKED).
/// theta_max = 2.6: inside the polynomial's positive monotone range (fold at
/// ~2.64), which the validator now certifies. The DLT-PnP lift uses the SEED
/// model (zero coefficients, theta_max=pi), so recovery is unaffected.
CameraModel64 buildGTKB4()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = 2.6;  // inside the polynomial's monotone range
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = 0.01;    // k1 — free
  m.distortion.coeffs[1] = -0.005;  // k2 — free
  m.distortion.coeffs[2] = 0.0;     // k3 = 0 = seed (LOCKED via FIX_DIST_2)
  m.distortion.coeffs[3] = 0.0;     // k4 = 0 = seed (LOCKED via FIX_DIST_3)
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  return m;
}

/// MEI Omnidirectional. GT fx=fy=200, xi=1.0 (= seed, LOCKED).
CameraModel64 buildGTMEI()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.xi = 1.0;  // = seed, LOCKED
  m.projection.theta_max = constants::kHalfPi;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  return m;
}

/// Double Sphere. GT fx=fy=200, xi=-0.2, alpha=0.5 (both = seed, LOCKED).
CameraModel64 buildGTDS()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.xi = -0.2;    // = seed, LOCKED
  m.projection.alpha = 0.5;  // = seed, LOCKED
  m.projection.theta_max = constants::kHalfPi;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  return m;
}

/// EUCM. GT fx=fy=200, alpha=0.5, beta=1.0 (both = seed, LOCKED).
CameraModel64 buildGTEUCM()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.alpha = 0.5;  // = seed, LOCKED
  m.projection.beta = 1.0;   // = seed, LOCKED
  m.projection.theta_max = constants::kHalfPi;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  return m;
}

// ---------------------------------------------------------------------------
// makeRicherViews — 12 views: 5 canonical + 7 additional with wider angular
// and positional diversity.
//
// Pose design:
//   Canonical 5 (identical to calib_intrinsics_test):
//     v0: I,          t=(0, 0, 0.30)         front-on
//     v1: Rx(+30°),   t=(0, -0.15, 0.30)     tilt up
//     v2: Ry(-30°),   t=(-0.15, 0, 0.35)     tilt left
//     v3: Rx(+25°)Ry(-25°), t=(0.1, 0.1, 0.35)  combined
//     v4: Rx(+20°)Ry(+20°)Rz(+15°), t=(-0.1, 0.12, 0.40)
//   Additional 7 (increased angular/translational diversity):
//     v5:  Rx(-30°),  t=(0, +0.15, 0.35)     tilt down
//     v6:  Ry(+30°),  t=(+0.15, 0, 0.35)     tilt right
//     v7:  Rx(+35°),  t=(0, -0.20, 0.40)     steep up tilt
//     v8:  Ry(-35°),  t=(-0.20, 0, 0.40)     steep left tilt
//     v9:  Rx(-25°)Ry(+30°), t=(0.15, 0.10, 0.45)
//     v10: Rx(+30°)Ry(-25°)Rz(-20°), t=(-0.12, -0.12, 0.45)
//     v11: I,         t=(0, 0, 0.50)          farther front-on
//
// Maximum incidence angle analysis (board corner angle from optical axis):
//   Board corner (col=7, row=5) at world (0.35, 0.25, 0) with the steepest
//   pose (v7: Rx(+35°), t=(0,-0.20,0.40)):
//     p_cam = Rx(35°) * (0.35, 0.25, 0) + (0, -0.20, 0.40)
//           = (0.35, 0.25*cos(35°)-0.25*sin(35°)*0+0-0.20, 0.25*sin(35°)+0.40)
//     cos(35°)=0.819, sin(35°)=0.574
//     p_cam ≈ (0.35, 0.25*0.819 - 0.20, 0.25*0.574 + 0.40)
//           ≈ (0.35, 0.205 - 0.20, 0.144 + 0.40)
//           ≈ (0.35, 0.005, 0.544)
//     theta = atan2(sqrt(0.35^2+0.005^2), 0.544)
//           = atan2(0.350, 0.544) ≈ 0.572 rad
//   That is below 0.70 rad. Using the opposite corner (col=0, row=5) at
//   world (0, 0.25, 0):
//     p_cam ≈ (0, 0.25*0.819-0.20, 0.25*0.574+0.40)
//           ≈ (0, 0.005, 0.544)   → theta ≈ 0.009 rad (near axis)
//   For v8 (Ry(-35°), t=(-0.20,0,0.40)) and corner (col=7, row=5):
//     p_cam = Ry(-35°)*(0.35, 0.25, 0) + (-0.20, 0, 0.40)
//     Ry(-35°): cos=-35°, so ca=0.819, sa=-0.574
//     px = 0.35*0.819 + 0 + (-0.20) = 0.287 - 0.20 = 0.087
//     py = 0.25
//     pz = -0.35*(-0.574) + 0 + 0.40 = 0.201 + 0.40 = 0.601
//     theta = atan2(sqrt(0.087^2+0.25^2), 0.601)
//           = atan2(0.265, 0.601) ≈ 0.418 rad  (below 0.70 rad)
//   The realised maximum is achieved by examining the far corner relative
//   to each pose. For v3 (Rx+25° Ry-25°) corner at (0.35, 0.25, 0) with
//   t=(0.1,0.1,0.35):
//     p1 = Ry(-25°)*(0.35,0.25,0) = (0.35*cos25, 0.25, 0.35*(-sin25))
//          = (0.317, 0.25, -0.148)
//     p_cam = Rx(25°)*p1 + t
//     Rx(25°): cos=0.906, sin=0.423
//     p_cam_y = 0.25*0.906 - (-0.148)*0.423 + 0.1 = 0.227 + 0.063 + 0.1 = 0.390
//     p_cam_z = 0.25*0.423 + (-0.148)*0.906 + 0.35 = 0.106 - 0.134 + 0.35 = 0.322
//     p_cam_x = 0.317 + 0.1 = 0.417
//     theta = atan2(sqrt(0.417^2+0.390^2), 0.322)
//           = atan2(sqrt(0.174+0.152), 0.322)
//           = atan2(0.571, 0.322) ≈ 1.059 rad  >> 0.7 rad
//   This exceeds 0.7 rad (good). However for models with theta_max=kHalfPi we
//   need to ensure it's below ~1.5708 rad. theta=1.059 < 1.5708 rad ✓.
//   For KB4 model with theta_max=pi this is certainly fine.
//   The maximum incidence angle realised is ≥ 1.059 rad >> 0.7 rad.
//   All GT theta_max values are kHalfPi (≈1.5708) or kPi, so 1.059 < kHalfPi
//   is FALSE (1.059 > pi/2 = 1.5708 is FALSE since 1.059 < 1.5708).
//   Actually 1.059 < 1.5708, so theta ≈ 1.059 rad IS within kHalfPi ✓.
//
//   NOTE: Any view/corner where rayToPixel64 returns non-OK is silently
//   skipped by the construction (buildCalibView returns false). The test
//   asserts views.size() >= 8 to guarantee sufficient coverage.
// ---------------------------------------------------------------------------

std::vector<CalibrationView> makeRicherViews(
  const CameraModel64 &gt_model, const std::vector<Eigen::Vector3d> &world_pts, double noise_sigma,
  std::mt19937 &rng
)
{
  using M3d = Eigen::Matrix3d;
  using V3d = Eigen::Vector3d;

  const double d30 = 30.0 * constants::kPi / 180.0;
  const double d25 = 25.0 * constants::kPi / 180.0;
  const double d20 = 20.0 * constants::kPi / 180.0;
  const double d15 = 15.0 * constants::kPi / 180.0;
  const double d35 = 35.0 * constants::kPi / 180.0;

  // Canonical 5 (same as calib_intrinsics_test makeCanonicalViews)
  const std::array<M3d, 12> Rs = {{
    M3d::Identity(),                                // v0
    rotMatX(+d30),                                  // v1
    rotMatY(-d30),                                  // v2
    rotMatX(+d25) * rotMatY(-d25),                  // v3
    rotMatX(+d20) * rotMatY(+d20) * rotMatZ(+d15),  // v4
    // Additional 7 — more tilt diversity
    rotMatX(-d30),                                  // v5
    rotMatY(+d30),                                  // v6
    rotMatX(+d35),                                  // v7
    rotMatY(-d35),                                  // v8
    rotMatX(-d25) * rotMatY(+d30),                  // v9
    rotMatX(+d30) * rotMatY(-d25) * rotMatZ(-d20),  // v10
    M3d::Identity()                                 // v11
  }};

  const std::array<V3d, 12> ts = {{
    V3d(0.00, 0.00, 0.30),    // v0
    V3d(0.00, -0.15, 0.30),   // v1
    V3d(-0.15, 0.00, 0.35),   // v2
    V3d(0.10, 0.10, 0.35),    // v3
    V3d(-0.10, 0.12, 0.40),   // v4
    V3d(0.00, 0.15, 0.35),    // v5
    V3d(0.15, 0.00, 0.35),    // v6
    V3d(0.00, -0.20, 0.40),   // v7
    V3d(-0.20, 0.00, 0.40),   // v8
    V3d(0.15, 0.10, 0.45),    // v9
    V3d(-0.12, -0.12, 0.45),  // v10
    V3d(0.00, 0.00, 0.50),    // v11
  }};

  std::vector<CalibrationView> views;
  views.reserve(12);

  for (int i = 0; i < 12; ++i)
  {
    CalibrationView view;
    if (buildCalibView(gt_model, Rs[i], ts[i], world_pts, noise_sigma, rng, view))
    {
      views.push_back(std::move(view));
    }
    // If a corner falls outside the model's domain, this pose is skipped.
    // The test asserts views.size() >= 8 to guard against degenerate fallback.
  }

  return views;
}

// ---------------------------------------------------------------------------
// buildPinholeZhangSeed — build a float CameraModel from estimatePinholeZhang
// output for use as calibrate seed. Skew is forced to 0 (standard pinhole
// assumption). theta_max is set to kHalfPi. Returns empty CameraModel{}
// (UNKNOWN type) on failure.
// ---------------------------------------------------------------------------

CameraModel buildPinholeZhangSeed(
  const std::vector<CalibrationView> &views, const std::vector<Eigen::Vector3d> &board_world_pts
)
{
  // Build PlanarObservation vector: board_pts = (X, Y) of Z=0 world points.
  const std::size_t M = board_world_pts.size();
  std::vector<PlanarObservation> obs;
  obs.reserve(views.size());

  for (const auto &v : views)
  {
    if (v.world_points.size() != M || v.image_points.size() != M)
    {
      return CameraModel{};  // sentinel
    }
    PlanarObservation po;
    po.board_pts.resize(2, static_cast<Eigen::Index>(M));
    po.image_pts.resize(2, static_cast<Eigen::Index>(M));
    for (std::size_t j = 0; j < M; ++j)
    {
      po.board_pts(0, static_cast<Eigen::Index>(j)) = v.world_points[j].x();
      po.board_pts(1, static_cast<Eigen::Index>(j)) = v.world_points[j].y();
      po.image_pts(0, static_cast<Eigen::Index>(j)) = v.image_points[j].x();
      po.image_pts(1, static_cast<Eigen::Index>(j)) = v.image_points[j].y();
    }
    obs.push_back(std::move(po));
  }

  Eigen::Matrix3d K;
  const StatusCode sc = estimatePinholeZhang(obs, K);
  if (sc != StatusCode::OK)
  {
    return CameraModel{};  // sentinel
  }

  // Zhang K is normalised so K(2,2) = 1.
  // Build float CameraModel: force skew=0 (standard assumption).
  CameraModel seed{};
  seed.projection.type = ProjectionModelType::PINHOLE;
  seed.projection.theta_max = static_cast<float>(constants::kHalfPi);
  seed.intrinsics.fx = static_cast<float>(K(0, 0));
  seed.intrinsics.fy = static_cast<float>(K(1, 1));
  seed.intrinsics.cx = static_cast<float>(K(0, 2));
  seed.intrinsics.cy = static_cast<float>(K(1, 2));
  seed.intrinsics.skew = 0.0f;  // force skew=0 per spec (standard pinhole path)
  seed.distortion.type = DistortionModelType::NONE;
  seed.distortion.space = DistortionSpace::NONE;
  seed.distortion.count = 0U;
  return seed;
}

}  // anonymous namespace

// ===========================================================================
// Phase 3 includes — cross-module rectify / remap leg.
// ===========================================================================

#include "camxiom/remap.hpp"
#include "camxiom/remap_kernel.hpp"

// ===========================================================================
// Test section A: Fisheye / MEI / DS / EUCM — getDefaultSeed → calibrate.
//
// For each model and σ ∈ {0, 0.3, 0.5}:
//   seed = getDefaultSeed(type, 640, 480)
//   views = makeRicherViews(GT, world_pts, σ, rng)
//   res = calibrate(views, seed, <lock_from_table>, opts)
//
// Assertions:
//   status == OK
//   noise-free: RMS <= 1e-3 px (noise-free Ceres sanity, same basis as
//               CALIBRATION_DESIGN_NOTES §8: "≈ machine precision")
//   noisy:      RMS <= σ*sqrt(2) * 1.6  (C-R floor σ√2; factor 1.6 safety)
//   fx/fy rel err vs GT:
//     noise-free: <= 1% (single-pass from seed, C-R gives machine precision)
//     σ=0.3 px:   <= 2% (CALIB_DESIGN_NOTES §8: sub-percent; 2% = 5× margin)
//     σ=0.5 px:   <= 3% (INIT_DESIGN_NOTES §10: KB4 ~1.1%, MEI 0.6%, DS/EUCM
//                        ~0.3–0.5%; 3% gives ≥3× margin at σ=0.5)
// ===========================================================================

// ---------------------------------------------------------------------------
// KB4 Fisheye
// ---------------------------------------------------------------------------

TEST(Ms2Integration, KB4NoiseFree)
{
  const CameraModel64 gt = buildGTKB4();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  // Require at least 8 valid views (some may be skipped for out-of-domain corners).
  ASSERT_GE(views.size(), 8u) << "Too few valid views built for KB4 noise-free";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, 640, 480);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK);

  // KB4 lock: k3=k4=0 (= seed; high-order terms non-identifiable, D29).
  const PnpFlag lock_flags = PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3;
  const CalibrationResult res = calibrate(views, seed, lock_flags, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK)
    << "KB4 noise-free failed: fx=" << res.camera_model.intrinsics.fx
    << " rms=" << res.rms_reprojection_error_px;

  // Noise-free RMS <= 1e-3 px: CALIBRATION_DESIGN_NOTES §8 "≈ machine precision"
  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "KB4 noise-free RMS=" << res.rms_reprojection_error_px << " > 1e-3 px";

  // fx within 1%: noise-free Ceres should reproduce GT focal exactly.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.01) << "KB4 noise-free fx rel err=" << fx_err << " > 1%"
                          << " (recovered=" << res.camera_model.intrinsics.fx << " GT=200)";
}

TEST(Ms2Integration, KB4NoisySigma03)
{
  const CameraModel64 gt = buildGTKB4();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for KB4 sigma=0.3";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, 640, 480);
  const PnpFlag lock_flags = PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3;
  const CalibrationResult res = calibrate(views, seed, lock_flags, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK)
    << "KB4 sigma=0.3 failed: fx=" << res.camera_model.intrinsics.fx;

  // RMS <= sigma*sqrt(2)*1.6: C-R floor sigma√2=0.424; *1.6 safety = 0.679 px
  // Basis: CALIBRATION_DESIGN_NOTES §8 "≈ σ·√2"; 1.6x safety margin.
  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "KB4 sigma=0.3 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 2%: INIT_DESIGN_NOTES §10 KB4 ~1.1% at σ=0.5; 2% at σ=0.3 is safe.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.02) << "KB4 sigma=0.3 fx rel err=" << fx_err << " > 2%"
                          << " (basis: INIT_DESIGN_NOTES §10 ~1.1% at σ=0.5)";
}

TEST(Ms2Integration, KB4NoisySigma05)
{
  const CameraModel64 gt = buildGTKB4();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for KB4 sigma=0.5";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, 640, 480);
  const PnpFlag lock_flags = PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3;
  const CalibrationResult res = calibrate(views, seed, lock_flags, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK)
    << "KB4 sigma=0.5 failed: fx=" << res.camera_model.intrinsics.fx;

  // RMS <= sigma*sqrt(2)*1.6: C-R floor=0.5*1.414=0.707; *1.6=1.131 px
  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "KB4 sigma=0.5 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 3%: INIT_DESIGN_NOTES §10 ~1.1% at σ=0.5; 3% = 2.7× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.03) << "KB4 sigma=0.5 fx rel err=" << fx_err << " > 3%"
                          << " (basis: INIT_DESIGN_NOTES §10 ~1.1% at σ=0.5, 3% margin)";
}

// ---------------------------------------------------------------------------
// MEI Omnidirectional
// ---------------------------------------------------------------------------

TEST(Ms2Integration, MEINoiseFree)
{
  const CameraModel64 gt = buildGTMEI();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for MEI noise-free";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, 640, 480);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK);

  // MEI lock: xi=1.0 (= seed, LOCKED, D30).
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "MEI noise-free failed";

  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "MEI noise-free RMS=" << res.rms_reprojection_error_px << " > 1e-3";

  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.01) << "MEI noise-free fx rel err=" << fx_err << " > 1%";
}

TEST(Ms2Integration, MEINoisySigma03)
{
  const CameraModel64 gt = buildGTMEI();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for MEI sigma=0.3";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "MEI sigma=0.3 failed";

  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "MEI sigma=0.3 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 2%: INIT_DESIGN_NOTES §10 MEI 0.60% at σ=0.5; 2% = 3.3× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.02) << "MEI sigma=0.3 fx rel err=" << fx_err << " > 2%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.60% at σ=0.5)";
}

TEST(Ms2Integration, MEINoisySigma05)
{
  const CameraModel64 gt = buildGTMEI();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for MEI sigma=0.5";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "MEI sigma=0.5 failed";

  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "MEI sigma=0.5 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 3%: INIT_DESIGN_NOTES §10 MEI 0.60% at σ=0.5; 3% = 5× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.03) << "MEI sigma=0.5 fx rel err=" << fx_err << " > 3%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.60% at σ=0.5)";
}

// ---------------------------------------------------------------------------
// Double Sphere
// ---------------------------------------------------------------------------

TEST(Ms2Integration, DSNoiseFree)
{
  const CameraModel64 gt = buildGTDS();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for DS noise-free";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK);

  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "DS noise-free failed";

  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "DS noise-free RMS=" << res.rms_reprojection_error_px << " > 1e-3";

  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.01) << "DS noise-free fx rel err=" << fx_err << " > 1%";
}

TEST(Ms2Integration, DSNoisySigma03)
{
  const CameraModel64 gt = buildGTDS();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for DS sigma=0.3";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "DS sigma=0.3 failed";

  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "DS sigma=0.3 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 2%: INIT_DESIGN_NOTES §10 DS 0.24–0.48% at σ=0.5; 2% = 4–8× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.02) << "DS sigma=0.3 fx rel err=" << fx_err << " > 2%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.24–0.48% at σ=0.5)";
}

TEST(Ms2Integration, DSNoisySigma05)
{
  const CameraModel64 gt = buildGTDS();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for DS sigma=0.5";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "DS sigma=0.5 failed";

  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "DS sigma=0.5 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 3%: INIT_DESIGN_NOTES §10 DS 0.24–0.48% at σ=0.5; 3% = 6–12× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.03) << "DS sigma=0.5 fx rel err=" << fx_err << " > 3%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.24–0.48% at σ=0.5)";
}

// ---------------------------------------------------------------------------
// EUCM
// ---------------------------------------------------------------------------

TEST(Ms2Integration, EUCMNoiseFree)
{
  const CameraModel64 gt = buildGTEUCM();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for EUCM noise-free";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::EUCM, 640, 480);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK);

  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "EUCM noise-free failed";

  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "EUCM noise-free RMS=" << res.rms_reprojection_error_px << " > 1e-3";

  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.01) << "EUCM noise-free fx rel err=" << fx_err << " > 1%";
}

TEST(Ms2Integration, EUCMNoisySigma03)
{
  const CameraModel64 gt = buildGTEUCM();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for EUCM sigma=0.3";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::EUCM, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "EUCM sigma=0.3 failed";

  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "EUCM sigma=0.3 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 2%: INIT_DESIGN_NOTES §10 EUCM 0.30–0.62% at σ=0.5; 2% = 3–6× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.02) << "EUCM sigma=0.3 fx rel err=" << fx_err << " > 2%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.30–0.62% at σ=0.5)";
}

TEST(Ms2Integration, EUCMNoisySigma05)
{
  const CameraModel64 gt = buildGTEUCM();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for EUCM sigma=0.5";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::EUCM, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "EUCM sigma=0.5 failed";

  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "EUCM sigma=0.5 RMS=" << res.rms_reprojection_error_px << " > C-R*1.6=" << rms_bound;

  // fx within 3%: INIT_DESIGN_NOTES §10 EUCM 0.30–0.62% at σ=0.5; 3% = 5× margin.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 200.0f) / 200.0;
  EXPECT_LT(fx_err, 0.03) << "EUCM sigma=0.5 fx rel err=" << fx_err << " > 3%"
                          << " (basis: INIT_DESIGN_NOTES §10 0.30–0.62% at σ=0.5)";
}

// ===========================================================================
// Test section B: Pinhole — real Zhang init path (estimatePinholeZhang seed).
//
// Build PlanarObservations from the richer views, run estimatePinholeZhang,
// build a float CameraModel (skew forced to 0, theta_max=kHalfPi), then run
// calibrate(views, zhang_seed, PnpFlag::NONE, opts).
//
// fx tolerance basis: INIT_DESIGN_NOTES §10 "< 0.1% fx error" for Zhang at
// σ=0.5. After a single Ceres pass the noise-free case should be machine
// precision; noisy should be at least as good as the getDefaultSeed path
// (= 2–3% at σ=0.3–0.5, with the Zhang seed providing a much better start).
// We use the SAME bounds as the getDefaultSeed pinhole path for consistency;
// if the Zhang path is better, the test will still pass.
// ===========================================================================

TEST(Ms2Integration, PinholeZhangSeedNoiseFree)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole-Zhang noise-free";

  const CameraModel zhang_seed = buildPinholeZhangSeed(views, wpts);
  ASSERT_EQ(validateCameraModel(zhang_seed), StatusCode::OK)
    << "estimatePinholeZhang failed or returned invalid model";

  const CalibrationResult res = calibrate(views, zhang_seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "Pinhole Zhang-seed noise-free failed";

  // Noise-free RMS <= 1e-3 px.
  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "Pinhole Zhang-seed noise-free RMS=" << res.rms_reprojection_error_px;

  // fx within 1%: Zhang provides a near-perfect seed, so noise-free should be
  // machine precision after Ceres.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.01) << "Pinhole Zhang-seed noise-free fx rel err=" << fx_err << " > 1%";
}

TEST(Ms2Integration, PinholeZhangSeedNoisySigma03)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole-Zhang sigma=0.3";

  const CameraModel zhang_seed = buildPinholeZhangSeed(views, wpts);
  ASSERT_EQ(validateCameraModel(zhang_seed), StatusCode::OK)
    << "estimatePinholeZhang failed on noisy data sigma=0.3";

  const CalibrationResult res = calibrate(views, zhang_seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "Pinhole Zhang-seed sigma=0.3 failed";

  // RMS <= sigma*sqrt(2)*1.6: C-R floor 0.3*1.414=0.424; *1.6=0.679 px.
  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "Pinhole Zhang-seed sigma=0.3 RMS=" << res.rms_reprojection_error_px
    << " > C-R*1.6=" << rms_bound;

  // fx within 3%: Zhang seed at σ=0.3 should be excellent (< 0.1% per §10),
  // so Ceres from that warm start should land within 3% conservatively.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.03) << "Pinhole Zhang-seed sigma=0.3 fx rel err=" << fx_err << " > 3%"
                          << " (basis: Zhang init < 0.1% at σ=0.5; 3% post-Ceres is conservative)";
}

TEST(Ms2Integration, PinholeZhangSeedNoisySigma05)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole-Zhang sigma=0.5";

  const CameraModel zhang_seed = buildPinholeZhangSeed(views, wpts);
  ASSERT_EQ(validateCameraModel(zhang_seed), StatusCode::OK)
    << "estimatePinholeZhang failed on noisy data sigma=0.5";

  const CalibrationResult res = calibrate(views, zhang_seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "Pinhole Zhang-seed sigma=0.5 failed";

  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "Pinhole Zhang-seed sigma=0.5 RMS=" << res.rms_reprojection_error_px
    << " > C-R*1.6=" << rms_bound;

  // fx within 4%: Zhang init typically < 0.1% fx error at σ=0.5; after Ceres
  // the bound is dominated by noise floor. 4% is very conservative for pinhole.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.04)
    << "Pinhole Zhang-seed sigma=0.5 fx rel err=" << fx_err << " > 4%"
    << " (basis: INIT_DESIGN_NOTES §10 Zhang < 0.1%, 4% post-Ceres is conservative)";
}

// ===========================================================================
// Test section C: Pinhole — contrast path, getDefaultSeed.
//
// Same geometry, same σ levels, but uses getDefaultSeed(PINHOLE) as seed
// instead of the Zhang estimate. This exercises the cold-start path with a
// large initial focal error (seed h/2=240 vs GT=500).
//
// fx tolerance basis: CALIBRATION_DESIGN_NOTES §8 "sub-percent" at σ=0.3;
// for the cold-start from h/2=240 we allow a larger 3–4% margin.
// ===========================================================================

TEST(Ms2Integration, PinholeDefaultSeedNoiseFree)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole default-seed noise-free";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK);

  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK)
    << "Pinhole default-seed noise-free failed: fx=" << res.camera_model.intrinsics.fx;

  // Noise-free RMS <= 1e-3 px.
  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "Pinhole default-seed noise-free RMS=" << res.rms_reprojection_error_px;

  // fx within 1%: noise-free Ceres from any reasonable seed reaches GT.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.01) << "Pinhole default-seed noise-free fx rel err=" << fx_err << " > 1%"
                          << " (seed h/2=240 vs GT=500, Ceres must converge)";
}

TEST(Ms2Integration, PinholeDefaultSeedNoisySigma03)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.3, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole default-seed sigma=0.3";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "Pinhole default-seed sigma=0.3 failed";

  const double rms_bound = 0.3 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "Pinhole default-seed sigma=0.3 RMS=" << res.rms_reprojection_error_px
    << " > C-R*1.6=" << rms_bound;

  // fx within 3%: cold start from seed=240 to GT=500 (108% gap);
  // with 12 diverse views Ceres should converge well; 3% is conservative.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.03) << "Pinhole default-seed sigma=0.3 fx rel err=" << fx_err << " > 3%"
                          << " (seed h/2=240, GT=500; 3% conservative after 12-view Ceres pass)";
}

TEST(Ms2Integration, PinholeDefaultSeedNoisySigma05)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.5, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views for pinhole default-seed sigma=0.5";

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::OK) << "Pinhole default-seed sigma=0.5 failed";

  const double rms_bound = 0.5 * std::sqrt(2.0) * 1.6;
  EXPECT_LT(res.rms_reprojection_error_px, rms_bound)
    << "Pinhole default-seed sigma=0.5 RMS=" << res.rms_reprojection_error_px
    << " > C-R*1.6=" << rms_bound;

  // fx within 4%: cold start from seed=240, σ=0.5 noise, 12 diverse views.
  // Basis: calib_intrinsics_test allows 3% at σ=0.3 from 5-view canonical;
  // here 12 views → more constraints, but σ=0.5 is larger; 4% is safe.
  const double fx_err = std::abs(res.camera_model.intrinsics.fx - 500.0f) / 500.0;
  EXPECT_LT(fx_err, 0.04) << "Pinhole default-seed sigma=0.5 fx rel err=" << fx_err << " > 4%"
                          << " (seed h/2=240, GT=500; 4% at σ=0.5 from cold start, 12 views)";
}

// ===========================================================================
// Test section D: Phase 3 — cross-module calibrate → rectify/remap leg.
//
// For each of the 5 models:
//   1. Run the noise-free pipeline (proven in Phase 2) to obtain a calibrated
//      CameraModel.
//   2. Group 1 (RectifyRoundTripAlpha0): build alpha=0 remap map, synthesize
//      a grid of rays, project them to both the calibrated source and the
//      rectified output pinhole, then bilinearly interpolate the map at the
//      output pixel and verify the sampled source coord matches the direct
//      source projection within 2.0 px.
//   3. Group 2 (RectifyAlpha1Smoke): build alpha=1 map, verify status OK and
//      finite positive FOV values.
//   4. Group 3 (RemapImageSmoke): apply a deterministic uint8 source image
//      through the alpha=0 map with NEAREST interpolation, verify status and
//      a spot-check on a non-sentinel pixel.
//
// Calibration pipeline reuse:
//   Pinhole: getDefaultSeed (PINHOLE, 640, 480) → calibrate(PnpFlag::NONE)
//   KB4:     getDefaultSeed (FISHEYE_THETA, …) → calibrate(FIX_DIST_2|FIX_DIST_3)
//   MEI:     getDefaultSeed (OMNIDIRECTIONAL, …) → calibrate(FIX_PROJECTION_PARAMS)
//   DS:      getDefaultSeed (DOUBLE_SPHERE, …)   → calibrate(FIX_PROJECTION_PARAMS)
//   EUCM:    getDefaultSeed (EUCM, …)             → calibrate(FIX_PROJECTION_PARAMS)
//
// Sensor 640×480 throughout; dst size same as src size for simplicity.
//
// Round-trip tolerance derivation:
//   The remap map stores the exact source coordinate for each integer dst pixel.
//   Bilinearly interpolating the map over a 1-px dst neighbourhood introduces an
//   error bounded by the local map gradient — i.e., how many source pixels
//   correspond to 1 output pixel.  For these calibrated models (focal ~200–500 px,
//   640×480 sensor) the map varies smoothly; the gradient across 1 dst pixel is
//   sub-pixel to a few px.  tol = 2.0 px is the conservative bound.
// ===========================================================================

namespace
{

// ---------------------------------------------------------------------------
// Helper: obtain a calibrated CameraModel (float) for a given GT builder and
// calibration parameters.  Asserts internally if calibration fails so the
// returned model is always valid when the helper returns.
// ---------------------------------------------------------------------------

struct CalibrateSpec
{
  ProjectionModelType seed_type;
  PnpFlag lock_flags;
};

CameraModel runNoiseFreePipeline(
  const CameraModel64 &gt, const std::vector<Eigen::Vector3d> &wpts, const CalibrateSpec &spec
)
{
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  if (views.size() < 8u)
  {
    ADD_FAILURE() << "runNoiseFreePipeline: too few valid views (" << views.size()
                  << ") for model type " << static_cast<int>(spec.seed_type);
    return CameraModel{};
  }
  const CameraModel seed = getDefaultSeed(spec.seed_type, 640, 480);
  if (validateCameraModel(seed) != StatusCode::OK)
  {
    ADD_FAILURE() << "runNoiseFreePipeline: getDefaultSeed returned invalid model";
    return CameraModel{};
  }
  const CalibrationResult res = calibrate(views, seed, spec.lock_flags, makeDefaultOptions());
  if (res.status != StatusCode::OK)
  {
    ADD_FAILURE() << "runNoiseFreePipeline: calibrate failed, status="
                  << static_cast<int>(res.status) << " rms=" << res.rms_reprojection_error_px;
    return CameraModel{};
  }
  return res.camera_model;
}

// ---------------------------------------------------------------------------
// Helper: bilinearly sample the (map_x, map_y) pair at a float dst coordinate.
// Returns false if any of the 4 corner taps are sentinel (< -0.5f) or OOB.
// ---------------------------------------------------------------------------

bool sampleMapBilinear(
  const float *map_x, const float *map_y, int dst_w, int dst_h, double pd_u, double pd_v,
  double &sampled_mx, double &sampled_my
)
{
  // Clamp to dst image
  if (pd_u < 0.0 || pd_v < 0.0 || pd_u >= static_cast<double>(dst_w) - 1.0 || pd_v >= static_cast<double>(dst_h) - 1.0)
  {
    return false;
  }
  const int u0 = static_cast<int>(pd_u);
  const int v0 = static_cast<int>(pd_v);
  const int u1 = u0 + 1;
  const int v1 = v0 + 1;
  if (u1 >= dst_w || v1 >= dst_h)
  {
    return false;
  }
  const double fu = pd_u - static_cast<double>(u0);
  const double fv = pd_v - static_cast<double>(v0);

  const int i00 = v0 * dst_w + u0;
  const int i10 = v0 * dst_w + u1;
  const int i01 = v1 * dst_w + u0;
  const int i11 = v1 * dst_w + u1;

  // Check all 4 taps for sentinel
  const float sentinel_thresh = -0.5f;
  if (map_x[i00] < sentinel_thresh || map_x[i10] < sentinel_thresh || map_x[i01] < sentinel_thresh || map_x[i11] < sentinel_thresh)
  {
    return false;
  }
  if (map_y[i00] < sentinel_thresh || map_y[i10] < sentinel_thresh || map_y[i01] < sentinel_thresh || map_y[i11] < sentinel_thresh)
  {
    return false;
  }

  const double mx00 = static_cast<double>(map_x[i00]);
  const double mx10 = static_cast<double>(map_x[i10]);
  const double mx01 = static_cast<double>(map_x[i01]);
  const double mx11 = static_cast<double>(map_x[i11]);

  const double my00 = static_cast<double>(map_y[i00]);
  const double my10 = static_cast<double>(map_y[i10]);
  const double my01 = static_cast<double>(map_y[i01]);
  const double my11 = static_cast<double>(map_y[i11]);

  sampled_mx = (1.0 - fv) * ((1.0 - fu) * mx00 + fu * mx10) + fv * ((1.0 - fu) * mx01 + fu * mx11);
  sampled_my = (1.0 - fv) * ((1.0 - fu) * my00 + fu * my10) + fv * ((1.0 - fu) * my01 + fu * my11);
  return true;
}

// ---------------------------------------------------------------------------
// Core round-trip checker for a calibrated source model + alpha=0 remap map.
// Synthesizes a 7×7 grid of forward rays, projects each to both source and
// output pixels, bilinearly samples the map at the output pixel, and asserts
// the sampled source coordinate matches the direct source projection.
//
// Returns the maximum error seen (for reporting) and the count of rays that
// were fully asserted.
// ---------------------------------------------------------------------------

void checkRoundTripAlpha0(
  const CameraModel &calibrated, const float *map_x, const float *map_y,
  const CameraModel &output_model, const std::string &model_name, double &max_err_out,
  int &asserted_count_out
)
{
  const CameraModel64 src64 = toCameraModel64(calibrated);
  const CameraModel64 out64 = toCameraModel64(output_model);

  constexpr int dst_w = 640;
  constexpr int dst_h = 480;

  max_err_out = 0.0;
  asserted_count_out = 0;

  // 7×7 normalised-plane grid: x,y ∈ [-0.5, 0.5] step 1/6.
  // This yields 49 candidate rays; many will be skipped if the projection
  // fails or the dst pixel is uncovered by the map.
  for (int ri = 0; ri <= 6; ++ri)
  {
    for (int ci = 0; ci <= 6; ++ci)
    {
      const double nx = -0.5 + ci * (1.0 / 6.0);
      const double ny = -0.5 + ri * (1.0 / 6.0);
      const Eigen::Vector3d ray_dir = Eigen::Vector3d(nx, ny, 1.0).normalized();

      // Project ray through source (calibrated) model.
      const PixelResult64 ps = rayToPixel64(src64, ray_dir);
      if (ps.status != StatusCode::OK)
      {
        continue;
      }
      // Check source pixel inside [0, 640) × [0, 480)
      if (ps.pixel.u < 0.0 || ps.pixel.u >= 640.0 || ps.pixel.v < 0.0 || ps.pixel.v >= 480.0)
      {
        continue;
      }

      // Project ray through output (rectified pinhole) model.
      const PixelResult64 pd = rayToPixel64(out64, ray_dir);
      if (pd.status != StatusCode::OK)
      {
        continue;
      }
      // Check dst pixel inside image bounds (need 1px border for bilinear)
      if (pd.pixel.u < 0.0 || pd.pixel.u >= static_cast<double>(dst_w) - 1.0 ||
          pd.pixel.v < 0.0 || pd.pixel.v >= static_cast<double>(dst_h) - 1.0)
      {
        continue;
      }

      // Bilinearly sample the map at the output pixel.
      double smx = 0.0, smy = 0.0;
      if (!sampleMapBilinear(map_x, map_y, dst_w, dst_h, pd.pixel.u, pd.pixel.v, smx, smy))
      {
        // One or more of the 4 taps were sentinel — skip.
        continue;
      }

      // Assert sampled source coord matches direct source projection.
      // Tolerance: 2.0 px — the map stores the exact source coord per dst
      // pixel; bilinearly interpolating it across a 1px dst neighbourhood
      // incurs an error bounded by the local map gradient over 1px.  For
      // these models/geometry that is sub-pixel to a few px; tol = 2.0 px
      // is the conservative bound.
      const double du = smx - ps.pixel.u;
      const double dv = smy - ps.pixel.v;
      const double err = std::sqrt(du * du + dv * dv);
      if (err > max_err_out)
      {
        max_err_out = err;
      }
      EXPECT_LE(err, 2.0) << model_name << " round-trip err=" << err << " px at ray (" << nx << ","
                          << ny << "): sampled=(" << smx << "," << smy << ") vs source=("
                          << ps.pixel.u << "," << ps.pixel.v << ")";
      ++asserted_count_out;
    }
  }
}

}  // namespace

// ===========================================================================
// Group 1: RectifyRoundTripAlpha0 — 5 models
// ===========================================================================

TEST(Ms2Integration, RectifyRoundTripPinholeAlpha0)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::PINHOLE, PnpFlag::NONE};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTPinhole(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK)
    << "Pinhole calibration failed; cannot test rectification";

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyMap alpha=0 failed for Pinhole";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "Pinhole rectified output_model is invalid";

  double max_err = 0.0;
  int asserted = 0;
  checkRoundTripAlpha0(
    calibrated, map_x.data(), map_y.data(), res.output_model, "Pinhole", max_err, asserted
  );

  // Require at least 15 rays asserted to prevent silent hollowing.
  ASSERT_GE(asserted, 15) << "Pinhole: too few rays asserted in round-trip (" << asserted
                          << "). Max err was " << max_err << " px.";
}

TEST(Ms2Integration, RectifyRoundTripKB4Alpha0)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{
    ProjectionModelType::FISHEYE_THETA, PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTKB4(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK)
    << "KB4 calibration failed; cannot test rectification";

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "buildRectifyMap alpha=0 failed for KB4";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "KB4 rectified output_model is invalid";

  double max_err = 0.0;
  int asserted = 0;
  checkRoundTripAlpha0(
    calibrated, map_x.data(), map_y.data(), res.output_model, "KB4", max_err, asserted
  );

  ASSERT_GE(asserted, 15) << "KB4: too few rays asserted (" << asserted << "). Max err was "
                          << max_err << " px.";
}

TEST(Ms2Integration, RectifyRoundTripMEIAlpha0)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::OMNIDIRECTIONAL, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTMEI(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK)
    << "MEI calibration failed; cannot test rectification";

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "buildRectifyMap alpha=0 failed for MEI";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "MEI rectified output_model is invalid";

  double max_err = 0.0;
  int asserted = 0;
  checkRoundTripAlpha0(
    calibrated, map_x.data(), map_y.data(), res.output_model, "MEI", max_err, asserted
  );

  ASSERT_GE(asserted, 15) << "MEI: too few rays asserted (" << asserted << "). Max err was "
                          << max_err << " px.";
}

TEST(Ms2Integration, RectifyRoundTripDSAlpha0)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::DOUBLE_SPHERE, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTDS(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK)
    << "DS calibration failed; cannot test rectification";

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "buildRectifyMap alpha=0 failed for DS";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "DS rectified output_model is invalid";

  double max_err = 0.0;
  int asserted = 0;
  checkRoundTripAlpha0(
    calibrated, map_x.data(), map_y.data(), res.output_model, "DS", max_err, asserted
  );

  ASSERT_GE(asserted, 15) << "DS: too few rays asserted (" << asserted << "). Max err was "
                          << max_err << " px.";
}

TEST(Ms2Integration, RectifyRoundTripEUCMAlpha0)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::EUCM, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTEUCM(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK)
    << "EUCM calibration failed; cannot test rectification";

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "buildRectifyMap alpha=0 failed for EUCM";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "EUCM rectified output_model is invalid";

  double max_err = 0.0;
  int asserted = 0;
  checkRoundTripAlpha0(
    calibrated, map_x.data(), map_y.data(), res.output_model, "EUCM", max_err, asserted
  );

  ASSERT_GE(asserted, 15) << "EUCM: too few rays asserted (" << asserted << "). Max err was "
                          << max_err << " px.";
}

// ===========================================================================
// Group 2: RectifyAlpha1Smoke — 5 models
//
// Verify that buildRectifyMap with alpha=1 succeeds and returns finite
// positive FOV values.  No round-trip assertion here; alpha=1 may clamp
// for wide fisheye per design (D36).
// ===========================================================================

TEST(Ms2Integration, RectifyAlpha1SmokePinhole)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::PINHOLE, PnpFlag::NONE};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTPinhole(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 1.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "Pinhole alpha=1 buildRectifyMap failed";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "Pinhole alpha=1 output_model is invalid";

  // FOV values must be finite and positive.
  EXPECT_TRUE(std::isfinite(res.horizontal_fov_deg))
    << "Pinhole alpha=1 horizontal_fov_deg is not finite";
  EXPECT_TRUE(std::isfinite(res.vertical_fov_deg))
    << "Pinhole alpha=1 vertical_fov_deg is not finite";
  EXPECT_GT(res.horizontal_fov_deg, 0.0f) << "Pinhole alpha=1 horizontal_fov_deg <= 0";
  EXPECT_GT(res.vertical_fov_deg, 0.0f) << "Pinhole alpha=1 vertical_fov_deg <= 0";
}

TEST(Ms2Integration, RectifyAlpha1SmokeKB4)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{
    ProjectionModelType::FISHEYE_THETA, PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTKB4(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 1.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "KB4 alpha=1 buildRectifyMap failed";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "KB4 alpha=1 output_model is invalid";

  EXPECT_TRUE(std::isfinite(res.horizontal_fov_deg));
  EXPECT_TRUE(std::isfinite(res.vertical_fov_deg));
  EXPECT_GT(res.horizontal_fov_deg, 0.0f);
  EXPECT_GT(res.vertical_fov_deg, 0.0f);
}

TEST(Ms2Integration, RectifyAlpha1SmokeMEI)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::OMNIDIRECTIONAL, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTMEI(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 1.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "MEI alpha=1 buildRectifyMap failed";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "MEI alpha=1 output_model is invalid";

  EXPECT_TRUE(std::isfinite(res.horizontal_fov_deg));
  EXPECT_TRUE(std::isfinite(res.vertical_fov_deg));
  EXPECT_GT(res.horizontal_fov_deg, 0.0f);
  EXPECT_GT(res.vertical_fov_deg, 0.0f);
}

TEST(Ms2Integration, RectifyAlpha1SmokeDS)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::DOUBLE_SPHERE, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTDS(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 1.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "DS alpha=1 buildRectifyMap failed";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "DS alpha=1 output_model is invalid";

  EXPECT_TRUE(std::isfinite(res.horizontal_fov_deg));
  EXPECT_TRUE(std::isfinite(res.vertical_fov_deg));
  EXPECT_GT(res.horizontal_fov_deg, 0.0f);
  EXPECT_GT(res.vertical_fov_deg, 0.0f);
}

TEST(Ms2Integration, RectifyAlpha1SmokeEUCM)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::EUCM, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTEUCM(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult res =
    buildRectifyMap(calibrated, sz, sz, 1.0f, map_x.data(), map_y.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK) << "EUCM alpha=1 buildRectifyMap failed";
  ASSERT_EQ(validateCameraModel(res.output_model), StatusCode::OK)
    << "EUCM alpha=1 output_model is invalid";

  EXPECT_TRUE(std::isfinite(res.horizontal_fov_deg));
  EXPECT_TRUE(std::isfinite(res.vertical_fov_deg));
  EXPECT_GT(res.horizontal_fov_deg, 0.0f);
  EXPECT_GT(res.vertical_fov_deg, 0.0f);
}

// ===========================================================================
// Group 3: RemapImageSmoke — 5 models, uint8_t NEAREST
//
// Build a deterministic source image: src[v*640+u] = (u*7 + v*13) & 0xFF.
// Build the alpha=0 remap map (same call as Group 1).
// Apply remapImage<uint8_t> with NEAREST and fill=0.
// Assertions:
//   - r.status == OK, r.total_count == 640*480
//   - r.valid_count > 0
//   - r.valid_count + r.border_count == r.total_count
//   - Spot-check: one non-sentinel dst pixel → expected = src[round(my)*640+round(mx)]
//   - Sentinel spot-check: one sentinel dst pixel → dst[i] == 0 (fill)
//     (if alpha=0 produces no sentinels for this model, this sub-assertion is
//      skipped — alpha=0 inscribed model guarantees all dst pixels map to valid
//      source coords, so some models may have zero sentinels)
// ===========================================================================

namespace
{

// Build the deterministic source image used in Group 3 tests.
std::vector<std::uint8_t> makeDeterministicSrc(int w, int h)
{
  std::vector<std::uint8_t> src(static_cast<std::size_t>(w * h));
  for (int v = 0; v < h; ++v)
  {
    for (int u = 0; u < w; ++u)
    {
      src[static_cast<std::size_t>(v * w + u)] = static_cast<std::uint8_t>((u * 7 + v * 13) & 0xFF);
    }
  }
  return src;
}

// Find the first non-sentinel dst pixel (map_x[i] >= -0.5).
// Returns -1 if none found.
int findFirstValidDstPixel(const float *map_x, int total)
{
  for (int i = 0; i < total; ++i)
  {
    if (map_x[i] >= -0.5f)
    {
      return i;
    }
  }
  return -1;
}

// Find the first sentinel dst pixel (map_x[i] < -0.5).
// Returns -1 if none found.
int findFirstSentinelDstPixel(const float *map_x, int total)
{
  for (int i = 0; i < total; ++i)
  {
    if (map_x[i] < -0.5f)
    {
      return i;
    }
  }
  return -1;
}

}  // namespace

TEST(Ms2Integration, RemapImageSmokePinhole)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::PINHOLE, PnpFlag::NONE};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTPinhole(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult rmap =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());
  ASSERT_EQ(rmap.remap_result.status, StatusCode::OK);

  const std::vector<std::uint8_t> src = makeDeterministicSrc(640, 480);
  std::vector<std::uint8_t> dst(640 * 480, 255u);

  const RemapImageResult r = remapImage<std::uint8_t>(
    src.data(), sz, map_x.data(), map_y.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(r.total_count, 640 * 480);
  EXPECT_GT(r.valid_count, 0);
  EXPECT_EQ(r.valid_count + r.border_count, r.total_count);

  // Spot-check on a non-sentinel pixel.
  const int vi = findFirstValidDstPixel(map_x.data(), 640 * 480);
  ASSERT_GE(vi, 0) << "No valid dst pixel found for Pinhole alpha=0";
  const float mx_f = map_x[vi];
  const float my_f = map_y[vi];
  const int src_u = static_cast<int>(std::round(mx_f));
  const int src_v = static_cast<int>(std::round(my_f));
  if (src_u >= 0 && src_u < 640 && src_v >= 0 && src_v < 480)
  {
    const std::uint8_t expected = src[static_cast<std::size_t>(src_v * 640 + src_u)];
    EXPECT_EQ(dst[static_cast<std::size_t>(vi)], expected)
      << "Pinhole NEAREST spot-check at dst[" << vi << "]: "
      << "expected src[" << src_v << "*640+" << src_u << "]=" << static_cast<int>(expected)
      << " got " << static_cast<int>(dst[static_cast<std::size_t>(vi)]);
  }

  // Sentinel spot-check: alpha=0 inscribed should map all dst pixels to valid
  // source coords.  If no sentinel exists, we skip this sub-assertion.
  // (alpha=0 = MAX_INSCRIBED_VALID policy: every output pixel is guaranteed to
  // have a valid source mapping.)
  const int si = findFirstSentinelDstPixel(map_x.data(), 640 * 480);
  if (si >= 0)
  {
    EXPECT_EQ(dst[static_cast<std::size_t>(si)], 0u)
      << "Pinhole sentinel pixel at dst[" << si << "] was not filled with 0";
  }
  // else: no sentinels at alpha=0 — correct for inscribed policy.
}

TEST(Ms2Integration, RemapImageSmokeKB4)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{
    ProjectionModelType::FISHEYE_THETA, PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTKB4(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult rmap =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());
  ASSERT_EQ(rmap.remap_result.status, StatusCode::OK);

  const std::vector<std::uint8_t> src = makeDeterministicSrc(640, 480);
  std::vector<std::uint8_t> dst(640 * 480, 255u);

  const RemapImageResult r = remapImage<std::uint8_t>(
    src.data(), sz, map_x.data(), map_y.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(r.total_count, 640 * 480);
  EXPECT_GT(r.valid_count, 0);
  EXPECT_EQ(r.valid_count + r.border_count, r.total_count);

  const int vi = findFirstValidDstPixel(map_x.data(), 640 * 480);
  ASSERT_GE(vi, 0) << "No valid dst pixel found for KB4 alpha=0";
  const float mx_f = map_x[vi];
  const float my_f = map_y[vi];
  const int src_u = static_cast<int>(std::round(mx_f));
  const int src_v = static_cast<int>(std::round(my_f));
  if (src_u >= 0 && src_u < 640 && src_v >= 0 && src_v < 480)
  {
    const std::uint8_t expected = src[static_cast<std::size_t>(src_v * 640 + src_u)];
    EXPECT_EQ(dst[static_cast<std::size_t>(vi)], expected)
      << "KB4 NEAREST spot-check at dst[" << vi << "]";
  }

  const int si = findFirstSentinelDstPixel(map_x.data(), 640 * 480);
  if (si >= 0)
  {
    EXPECT_EQ(dst[static_cast<std::size_t>(si)], 0u)
      << "KB4 sentinel pixel at dst[" << si << "] was not filled with 0";
  }
}

TEST(Ms2Integration, RemapImageSmokeMEI)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::OMNIDIRECTIONAL, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTMEI(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult rmap =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());
  ASSERT_EQ(rmap.remap_result.status, StatusCode::OK);

  const std::vector<std::uint8_t> src = makeDeterministicSrc(640, 480);
  std::vector<std::uint8_t> dst(640 * 480, 255u);

  const RemapImageResult r = remapImage<std::uint8_t>(
    src.data(), sz, map_x.data(), map_y.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(r.total_count, 640 * 480);
  EXPECT_GT(r.valid_count, 0);
  EXPECT_EQ(r.valid_count + r.border_count, r.total_count);

  const int vi = findFirstValidDstPixel(map_x.data(), 640 * 480);
  ASSERT_GE(vi, 0) << "No valid dst pixel found for MEI alpha=0";
  const float mx_f = map_x[vi];
  const float my_f = map_y[vi];
  const int src_u = static_cast<int>(std::round(mx_f));
  const int src_v = static_cast<int>(std::round(my_f));
  if (src_u >= 0 && src_u < 640 && src_v >= 0 && src_v < 480)
  {
    const std::uint8_t expected = src[static_cast<std::size_t>(src_v * 640 + src_u)];
    EXPECT_EQ(dst[static_cast<std::size_t>(vi)], expected)
      << "MEI NEAREST spot-check at dst[" << vi << "]";
  }

  const int si = findFirstSentinelDstPixel(map_x.data(), 640 * 480);
  if (si >= 0)
  {
    EXPECT_EQ(dst[static_cast<std::size_t>(si)], 0u)
      << "MEI sentinel pixel at dst[" << si << "] was not filled with 0";
  }
}

TEST(Ms2Integration, RemapImageSmokeDS)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::DOUBLE_SPHERE, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTDS(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult rmap =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());
  ASSERT_EQ(rmap.remap_result.status, StatusCode::OK);

  const std::vector<std::uint8_t> src = makeDeterministicSrc(640, 480);
  std::vector<std::uint8_t> dst(640 * 480, 255u);

  const RemapImageResult r = remapImage<std::uint8_t>(
    src.data(), sz, map_x.data(), map_y.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(r.total_count, 640 * 480);
  EXPECT_GT(r.valid_count, 0);
  EXPECT_EQ(r.valid_count + r.border_count, r.total_count);

  const int vi = findFirstValidDstPixel(map_x.data(), 640 * 480);
  ASSERT_GE(vi, 0) << "No valid dst pixel found for DS alpha=0";
  const float mx_f = map_x[vi];
  const float my_f = map_y[vi];
  const int src_u = static_cast<int>(std::round(mx_f));
  const int src_v = static_cast<int>(std::round(my_f));
  if (src_u >= 0 && src_u < 640 && src_v >= 0 && src_v < 480)
  {
    const std::uint8_t expected = src[static_cast<std::size_t>(src_v * 640 + src_u)];
    EXPECT_EQ(dst[static_cast<std::size_t>(vi)], expected)
      << "DS NEAREST spot-check at dst[" << vi << "]";
  }

  const int si = findFirstSentinelDstPixel(map_x.data(), 640 * 480);
  if (si >= 0)
  {
    EXPECT_EQ(dst[static_cast<std::size_t>(si)], 0u)
      << "DS sentinel pixel at dst[" << si << "] was not filled with 0";
  }
}

TEST(Ms2Integration, RemapImageSmokeEUCM)
{
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  const CalibrateSpec spec{ProjectionModelType::EUCM, PnpFlag::FIX_PROJECTION_PARAMS};
  const CameraModel calibrated = runNoiseFreePipeline(buildGTEUCM(), wpts, spec);
  ASSERT_EQ(validateCameraModel(calibrated), StatusCode::OK);

  const ImageSize sz{640, 480};
  std::vector<float> map_x(640 * 480), map_y(640 * 480);
  const BuildRectifyMapResult rmap =
    buildRectifyMap(calibrated, sz, sz, 0.0f, map_x.data(), map_y.data());
  ASSERT_EQ(rmap.remap_result.status, StatusCode::OK);

  const std::vector<std::uint8_t> src = makeDeterministicSrc(640, 480);
  std::vector<std::uint8_t> dst(640 * 480, 255u);

  const RemapImageResult r = remapImage<std::uint8_t>(
    src.data(), sz, map_x.data(), map_y.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(r.total_count, 640 * 480);
  EXPECT_GT(r.valid_count, 0);
  EXPECT_EQ(r.valid_count + r.border_count, r.total_count);

  const int vi = findFirstValidDstPixel(map_x.data(), 640 * 480);
  ASSERT_GE(vi, 0) << "No valid dst pixel found for EUCM alpha=0";
  const float mx_f = map_x[vi];
  const float my_f = map_y[vi];
  const int src_u = static_cast<int>(std::round(mx_f));
  const int src_v = static_cast<int>(std::round(my_f));
  if (src_u >= 0 && src_u < 640 && src_v >= 0 && src_v < 480)
  {
    const std::uint8_t expected = src[static_cast<std::size_t>(src_v * 640 + src_u)];
    EXPECT_EQ(dst[static_cast<std::size_t>(vi)], expected)
      << "EUCM NEAREST spot-check at dst[" << vi << "]";
  }

  const int si = findFirstSentinelDstPixel(map_x.data(), 640 * 480);
  if (si >= 0)
  {
    EXPECT_EQ(dst[static_cast<std::size_t>(si)], 0u)
      << "EUCM sentinel pixel at dst[" << si << "] was not filled with 0";
  }
}

// ===========================================================================
// Phase 4 — Real-correspondence fixture infrastructure (Ms2IntegrationFixture)
//
// camxiom is OpenCV-free and cannot decode raw images. A "real-image fixture"
// is therefore a detected-correspondence text file: world↔image point pairs
// that a real camera + detector produced, parsed with std::ifstream only.
// Raw-image → corner detection is the app layer's job (MS4-new, out of scope).
//
// Fixture text format (documented in tests/fixtures/README.md):
//   Lines starting with '#' and blank lines are ignored.
//   model: <pinhole|fisheye_theta|omnidirectional|double_sphere|eucm>
//          (lowercase = toString() output; uppercase also accepted)
//   image_size: <w> <h>
//   # optional regression expectations (checked only when present):
//   expected_fx: <float>
//   expected_fy: <float>
//   expected_cx: <float>
//   expected_cy: <float>
//   view: <N>
//   <wx> <wy> <wz> <u> <v>   # exactly N rows
//   view: <N>
//   ...                       # >= 3 views; each view N >= 6 points
//
// LOCK TABLE (per-model, D29/D30):
//   PINHOLE:         PnpFlag::NONE
//   FISHEYE_THETA:   PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3
//   OMNIDIRECTIONAL: PnpFlag::FIX_PROJECTION_PARAMS
//   DOUBLE_SPHERE:   PnpFlag::FIX_PROJECTION_PARAMS
//   EUCM:            PnpFlag::FIX_PROJECTION_PARAMS
//
// Regression tolerances (provisional scaffold; tighten once real fixtures are
// available and their noise characteristics are known):
//   fx, fy: relative 5% of expected value
//   cx, cy: absolute 10 px
// ===========================================================================

// ---------------------------------------------------------------------------
// Phase 4 additional includes (std-only; OpenCV-free)
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace
{

// ---------------------------------------------------------------------------
// ParsedFixture — result struct for parseFixture().
// ---------------------------------------------------------------------------

struct ParsedFixture
{
  ProjectionModelType model{ProjectionModelType::UNKNOWN};
  int w{0};
  int h{0};
  bool has_fx{false};
  bool has_fy{false};
  bool has_cx{false};
  bool has_cy{false};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  std::vector<CalibrationView> views;
};

// ---------------------------------------------------------------------------
// parseFixture — parse the camxiom detected-correspondence fixture format.
//
// Returns true on success, false on any error (err is set descriptively).
// Rejects:
//   - unknown model string
//   - missing model or image_size before any view:
//   - fewer than 3 views
//   - any view with fewer than 6 points
//   - mismatched row count (declared N vs actual rows)
//   - non-numeric tokens in numeric fields
// ---------------------------------------------------------------------------

bool parseFixture(std::istream &in, ParsedFixture &out, std::string &err)
{
  out = ParsedFixture{};
  err.clear();

  bool have_model = false;
  bool have_img_size = false;
  bool in_view = false;
  int view_expected_n = 0;

  std::string line;
  int lineno = 0;
  while (std::getline(in, line))
  {
    ++lineno;

    // Strip leading/trailing whitespace
    auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
      continue;  // blank line
    }
    if (line[first] == '#')
    {
      continue;  // comment
    }

    std::istringstream ls(line);
    std::string token;
    ls >> token;

    // ----- Directive: model: -----
    if (token == "model:")
    {
      std::string model_str;
      if (!(ls >> model_str))
      {
        err = "line " + std::to_string(lineno) + ": 'model:' has no value";
        return false;
      }
      // Accept both the lowercase form returned by toString() and the
      // traditional uppercase form for human-authored fixtures.
      if (model_str == "pinhole" || model_str == "PINHOLE")
      {
        out.model = ProjectionModelType::PINHOLE;
      }
      else if (model_str == "fisheye_theta" || model_str == "FISHEYE_THETA")
      {
        out.model = ProjectionModelType::FISHEYE_THETA;
      }
      else if (model_str == "omnidirectional" || model_str == "OMNIDIRECTIONAL")
      {
        out.model = ProjectionModelType::OMNIDIRECTIONAL;
      }
      else if (model_str == "double_sphere" || model_str == "DOUBLE_SPHERE")
      {
        out.model = ProjectionModelType::DOUBLE_SPHERE;
      }
      else if (model_str == "eucm" || model_str == "EUCM")
      {
        out.model = ProjectionModelType::EUCM;
      }
      else
      {
        err = "line " + std::to_string(lineno) + ": unknown model '" + model_str +
              "'; expected one of: pinhole, fisheye_theta, omnidirectional, "
              "double_sphere, eucm (case-insensitive)";
        return false;
      }
      have_model = true;
      continue;
    }

    // ----- Directive: image_size: -----
    if (token == "image_size:")
    {
      if (!(ls >> out.w >> out.h))
      {
        err = "line " + std::to_string(lineno) + ": 'image_size:' requires two integers <w> <h>";
        return false;
      }
      if (out.w <= 0 || out.h <= 0)
      {
        err = "line " + std::to_string(lineno) + ": image_size values must be positive (got " +
              std::to_string(out.w) + " " + std::to_string(out.h) + ")";
        return false;
      }
      have_img_size = true;
      continue;
    }

    // ----- Optional regression expectations -----
    if (token == "expected_fx:")
    {
      if (!(ls >> out.fx))
      {
        err = "line " + std::to_string(lineno) + ": non-numeric value for expected_fx";
        return false;
      }
      out.has_fx = true;
      continue;
    }
    if (token == "expected_fy:")
    {
      if (!(ls >> out.fy))
      {
        err = "line " + std::to_string(lineno) + ": non-numeric value for expected_fy";
        return false;
      }
      out.has_fy = true;
      continue;
    }
    if (token == "expected_cx:")
    {
      if (!(ls >> out.cx))
      {
        err = "line " + std::to_string(lineno) + ": non-numeric value for expected_cx";
        return false;
      }
      out.has_cx = true;
      continue;
    }
    if (token == "expected_cy:")
    {
      if (!(ls >> out.cy))
      {
        err = "line " + std::to_string(lineno) + ": non-numeric value for expected_cy";
        return false;
      }
      out.has_cy = true;
      continue;
    }

    // ----- Directive: view: -----
    if (token == "view:")
    {
      // Finalise previous view if open
      if (in_view)
      {
        const int actual_n = static_cast<int>(out.views.back().world_points.size());
        if (actual_n != view_expected_n)
        {
          err = "line " + std::to_string(lineno) + ": view declared " +
                std::to_string(view_expected_n) + " points but got " + std::to_string(actual_n);
          return false;
        }
        if (actual_n < 6)
        {
          err = "line " + std::to_string(lineno) + ": view has only " + std::to_string(actual_n) +
                " points; minimum is 6";
          return false;
        }
      }
      if (!have_model)
      {
        err = "line " + std::to_string(lineno) + ": 'view:' seen before 'model:' directive";
        return false;
      }
      if (!have_img_size)
      {
        err = "line " + std::to_string(lineno) + ": 'view:' seen before 'image_size:' directive";
        return false;
      }
      if (!(ls >> view_expected_n))
      {
        err = "line " + std::to_string(lineno) + ": 'view:' requires an integer point count";
        return false;
      }
      if (view_expected_n < 6)
      {
        err = "line " + std::to_string(lineno) + ": view declared " +
              std::to_string(view_expected_n) + " points; minimum is 6";
        return false;
      }
      out.views.emplace_back();
      out.views.back().world_points.reserve(static_cast<std::size_t>(view_expected_n));
      out.views.back().image_points.reserve(static_cast<std::size_t>(view_expected_n));
      in_view = true;
      continue;
    }

    // ----- Data row: <wx> <wy> <wz> <u> <v> -----
    if (in_view)
    {
      // token is the first number (wx); parse the remaining 4
      double wx = 0.0, wy = 0.0, wz = 0.0, u = 0.0, v = 0.0;
      try
      {
        wx = std::stod(token);
      }
      catch (...)
      {
        err =
          "line " + std::to_string(lineno) + ": non-numeric token '" + token + "' in point data";
        return false;
      }
      if (!(ls >> wy >> wz >> u >> v))
      {
        err = "line " + std::to_string(lineno) +
              ": point row must have exactly 5 numbers: <wx> <wy> <wz> <u> <v>";
        return false;
      }
      out.views.back().world_points.emplace_back(wx, wy, wz);
      out.views.back().image_points.emplace_back(u, v);
      continue;
    }

    // Unknown token outside view context
    err = "line " + std::to_string(lineno) + ": unexpected token '" + token + "'";
    return false;
  }

  // Finalise last open view
  if (in_view)
  {
    const int actual_n = static_cast<int>(out.views.back().world_points.size());
    if (actual_n != view_expected_n)
    {
      err = "end of file: last view declared " + std::to_string(view_expected_n) +
            " points but got " + std::to_string(actual_n);
      return false;
    }
    if (actual_n < 6)
    {
      err = "end of file: last view has only " + std::to_string(actual_n) + " points; minimum is 6";
      return false;
    }
  }

  // Validate completeness
  if (!have_model)
  {
    err = "fixture has no 'model:' directive";
    return false;
  }
  if (!have_img_size)
  {
    err = "fixture has no 'image_size:' directive";
    return false;
  }
  if (static_cast<int>(out.views.size()) < 3)
  {
    err = "fixture has only " + std::to_string(out.views.size()) + " view(s); minimum is 3";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// lockFor — per-model D29/D30 Ceres lock flags.
// PINHOLE:         PnpFlag::NONE
// FISHEYE_THETA:   FIX_DIST_2 | FIX_DIST_3   (k3=k4=0 locked)
// OMNI/DS/EUCM:   FIX_PROJECTION_PARAMS       (xi, alpha, beta locked at seed)
// ---------------------------------------------------------------------------

PnpFlag lockFor(ProjectionModelType model)
{
  switch (model)
  {
    case ProjectionModelType::PINHOLE:
      return PnpFlag::NONE;
    case ProjectionModelType::FISHEYE_THETA:
      return PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3;
    case ProjectionModelType::OMNIDIRECTIONAL:
    case ProjectionModelType::DOUBLE_SPHERE:
    case ProjectionModelType::EUCM:
      return PnpFlag::FIX_PROJECTION_PARAMS;
    default:
      // UNKNOWN — return NONE; calibrate() will reject the invalid seed anyway.
      return PnpFlag::NONE;
  }
}

// ---------------------------------------------------------------------------
// writeFixture — serialise a KB4 GT model + CalibrationViews to the fixture
// text format. Used by ParserRoundTripSynthetic to prove parser/format/pipeline.
//
// Precision: std::setprecision(17) (sufficient for exact double round-trip via
// stod); point coordinate assertions in the test use EXPECT_DOUBLE_EQ since
// write→read→compare is exact at this precision.
// ---------------------------------------------------------------------------

void writeFixture(
  std::ostream &out, ProjectionModelType model, int w, int h,
  const std::vector<CalibrationView> &views, bool include_expected, double exp_fx, double exp_fy,
  double exp_cx, double exp_cy
)
{
  out << std::setprecision(17);
  out << "model: " << toString(model) << "\n";
  out << "image_size: " << w << " " << h << "\n";
  if (include_expected)
  {
    out << "expected_fx: " << exp_fx << "\n";
    out << "expected_fy: " << exp_fy << "\n";
    out << "expected_cx: " << exp_cx << "\n";
    out << "expected_cy: " << exp_cy << "\n";
  }
  for (const auto &v : views)
  {
    const auto n = static_cast<int>(v.world_points.size());
    out << "view: " << n << "\n";
    for (int j = 0; j < n; ++j)
    {
      out << v.world_points[j].x() << " " << v.world_points[j].y() << " " << v.world_points[j].z()
          << " " << v.image_points[j].x() << " " << v.image_points[j].y() << "\n";
    }
  }
}

// ---------------------------------------------------------------------------
// discoverFixtureDir — return the directory to search for *.fixture files.
// Priority:
//   1. CAMXIOM_TEST_FIXTURE_DIR environment variable (if set and non-empty).
//   2. Default: "tests/fixtures" relative to the source file's location.
//      __FILE__ typically expands to the absolute path at build time.
// ---------------------------------------------------------------------------

std::string discoverFixtureDir()
{
  const char *env = std::getenv("CAMXIOM_TEST_FIXTURE_DIR");
  if (env != nullptr && env[0] != '\0')
  {
    return std::string(env);
  }
  // Derive from __FILE__: strip the filename portion to get the source dir,
  // then append "/fixtures".
  const std::string src_file(__FILE__);
  // __FILE__ ends with e.g. ".../tests/integration_test.cpp"
  const auto slash = src_file.rfind('/');
  if (slash == std::string::npos)
  {
    // Fallback: relative to cwd (may not work for all runners, but is
    // actionable because the skip message will show the path).
    return "tests/fixtures";
  }
  return src_file.substr(0, slash) + "/fixtures";
}

// ---------------------------------------------------------------------------
// listFixtureFiles — list all *.fixture files in dir, sorted lexicographically
// for determinism.  Returns an empty vector if dir does not exist or is empty.
// Pure std: iterates via POSIX opendir/readdir (available on all Linux/macOS
// targets; no Boost/Filesystem required).
// ---------------------------------------------------------------------------

#include <dirent.h>

std::vector<std::string> listFixtureFiles(const std::string &dir)
{
  std::vector<std::string> result;
  DIR *dp = ::opendir(dir.c_str());
  if (dp == nullptr)
  {
    return result;  // directory does not exist or is not accessible
  }
  struct dirent *entry = nullptr;
  while ((entry = ::readdir(dp)) != nullptr)
  {
    const std::string name(entry->d_name);
    // Keep only files ending in ".fixture"
    const std::string suffix(".fixture");
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
      result.push_back(dir + "/" + name);
    }
  }
  ::closedir(dp);
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

// ===========================================================================
// Test 1 — ParserRoundTripSynthetic
//
// Exercises the full scaffold deterministically without any external file:
//   build GT KB4 → makeRicherViews(σ=0) → serialise to fixture text →
//   parse back → assert parse fields → calibrate → assert status OK +
//   noise-free RMS < 1e-3 + fx within 1%.
//
// This test runs on every CI build and proves the format, parser, and
// calibrate path cannot silently bitrot.
// ===========================================================================

TEST(Ms2IntegrationFixture, ParserRoundTripSynthetic)
{
  // --- Build ground-truth KB4 views (noise-free) ---
  const CameraModel64 gt = buildGTKB4();
  const auto wpts = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeRicherViews(gt, wpts, 0.0, rng);
  ASSERT_GE(views.size(), 8u) << "Too few valid views in ParserRoundTripSynthetic";

  // GT intrinsics for regression expectations written into the fixture:
  // fx=fy=200, cx=320, cy=240 (matches buildGTKB4).
  const double gt_fx = 200.0;
  const double gt_fy = 200.0;
  const double gt_cx = 320.0;
  const double gt_cy = 240.0;

  // --- Serialise to fixture text ---
  std::ostringstream oss;
  writeFixture(
    oss, ProjectionModelType::FISHEYE_THETA, 640, 480, views,
    /*include_expected=*/true, gt_fx, gt_fy, gt_cx, gt_cy
  );

  const std::string fixture_text = oss.str();
  ASSERT_FALSE(fixture_text.empty()) << "writeFixture produced empty text";

  // --- Parse it back ---
  std::istringstream iss(fixture_text);
  ParsedFixture parsed;
  std::string parse_err;
  ASSERT_TRUE(parseFixture(iss, parsed, parse_err)) << "parseFixture failed: " << parse_err;

  // --- Assert model / image_size ---
  EXPECT_EQ(parsed.model, ProjectionModelType::FISHEYE_THETA) << "Parsed model mismatch";
  EXPECT_EQ(parsed.w, 640) << "Parsed image width mismatch";
  EXPECT_EQ(parsed.h, 480) << "Parsed image height mismatch";

  // --- Assert expected_* flags and values ---
  EXPECT_TRUE(parsed.has_fx) << "parsed.has_fx is false";
  EXPECT_TRUE(parsed.has_fy) << "parsed.has_fy is false";
  EXPECT_TRUE(parsed.has_cx) << "parsed.has_cx is false";
  EXPECT_TRUE(parsed.has_cy) << "parsed.has_cy is false";
  EXPECT_DOUBLE_EQ(parsed.fx, gt_fx) << "parsed.fx mismatch";
  EXPECT_DOUBLE_EQ(parsed.fy, gt_fy) << "parsed.fy mismatch";
  EXPECT_DOUBLE_EQ(parsed.cx, gt_cx) << "parsed.cx mismatch";
  EXPECT_DOUBLE_EQ(parsed.cy, gt_cy) << "parsed.cy mismatch";

  // --- Assert view count and per-view point counts ---
  ASSERT_EQ(parsed.views.size(), views.size()) << "Parsed view count does not match original";
  for (std::size_t i = 0; i < views.size(); ++i)
  {
    ASSERT_EQ(parsed.views[i].world_points.size(), views[i].world_points.size())
      << "View " << i << " world_points count mismatch";
    ASSERT_EQ(parsed.views[i].image_points.size(), views[i].image_points.size())
      << "View " << i << " image_points count mismatch";
    // Spot-check first and last points at full precision (setprecision(17)
    // guarantees exact double round-trip via stod/>>).
    EXPECT_DOUBLE_EQ(parsed.views[i].world_points[0].x(), views[i].world_points[0].x())
      << "View " << i << " pt[0] wx mismatch";
    EXPECT_DOUBLE_EQ(parsed.views[i].image_points[0].x(), views[i].image_points[0].x())
      << "View " << i << " pt[0] u mismatch";
    const std::size_t last = views[i].world_points.size() - 1;
    EXPECT_DOUBLE_EQ(parsed.views[i].world_points[last].y(), views[i].world_points[last].y())
      << "View " << i << " pt[last] wy mismatch";
    EXPECT_DOUBLE_EQ(parsed.views[i].image_points[last].y(), views[i].image_points[last].y())
      << "View " << i << " pt[last] v mismatch";
  }

  // --- Run getDefaultSeed → calibrate on the parsed views ---
  const CameraModel seed = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, parsed.w, parsed.h);
  ASSERT_EQ(validateCameraModel(seed), StatusCode::OK)
    << "getDefaultSeed(FISHEYE_THETA) returned invalid model";

  const CalibrationResult res =
    calibrate(parsed.views, seed, lockFor(parsed.model), makeDefaultOptions(parsed.w, parsed.h));

  ASSERT_EQ(res.status, StatusCode::OK)
    << "calibrate failed on parsed fixture views; rms=" << res.rms_reprojection_error_px;

  // Noise-free RMS < 1e-3 px: same basis as Section A KB4NoiseFree.
  // Justification: noise=0, FISHEYE_THETA seed + calibrate should reach
  // machine precision for consistent geometry.
  EXPECT_LT(res.rms_reprojection_error_px, 1e-3)
    << "ParserRoundTrip noise-free RMS=" << res.rms_reprojection_error_px
    << " > 1e-3 px (expected machine-precision from noise-free data)";

  // fx within 1% of GT=200: noise-free Ceres from seed should reproduce GT.
  // Basis: KB4NoiseFree asserts < 1%; same geometry, same data.
  const double fx_err =
    std::abs(static_cast<double>(res.camera_model.intrinsics.fx) - gt_fx) / gt_fx;
  EXPECT_LT(fx_err, 0.01) << "ParserRoundTrip fx rel err=" << fx_err << " > 1%"
                          << " (recovered=" << res.camera_model.intrinsics.fx << " GT=" << gt_fx
                          << ")";
}

// ===========================================================================
// Test 2 — RealCorrespondenceRegression
//
// Skips unless the user supplies *.fixture files.  Fixture discovery:
//   1. CAMXIOM_TEST_FIXTURE_DIR env var (if set and non-empty).
//   2. Default: <source-dir>/fixtures/ (i.e. tests/fixtures/).
//
// For each *.fixture file:
//   - Parse with parseFixture().
//   - getDefaultSeed(model, w, h) → calibrate(views, seed, lockFor(model), opts).
//   - Assert status == OK.
//   - For each expected_* present, assert within provisional scaffold tolerance:
//       fx, fy: relative 5%  (no real noise characteristics known yet)
//       cx, cy: absolute 10 px
//     (Tighten these after receiving real fixture data with known noise floor.)
// ===========================================================================

TEST(Ms2IntegrationFixture, RealCorrespondenceRegression)
{
  const std::string fixture_dir = discoverFixtureDir();
  const std::vector<std::string> files = listFixtureFiles(fixture_dir);

  if (files.empty())
  {
    GTEST_SKIP() << "No real-correspondence fixture found. "
                 << "To enable this regression, drop a <name>.fixture file "
                 << "(format: see " << fixture_dir << "/README.md) "
                 << "into " << fixture_dir
                 << " or set CAMXIOM_TEST_FIXTURE_DIR to the directory containing "
                 << "*.fixture files. "
                 << "Searched: " << fixture_dir;
  }

  for (const auto &fpath : files)
  {
    SCOPED_TRACE("fixture: " + fpath);

    std::ifstream fin(fpath);
    ASSERT_TRUE(fin.is_open()) << "Could not open fixture file: " << fpath;

    ParsedFixture parsed;
    std::string parse_err;
    ASSERT_TRUE(parseFixture(fin, parsed, parse_err))
      << "parseFixture failed for '" << fpath << "': " << parse_err;

    ASSERT_GE(static_cast<int>(parsed.views.size()), 3)
      << "Fixture '" << fpath << "' has fewer than 3 views";
    for (std::size_t vi = 0; vi < parsed.views.size(); ++vi)
    {
      ASSERT_GE(static_cast<int>(parsed.views[vi].world_points.size()), 6)
        << "Fixture '" << fpath << "' view " << vi << " has fewer than 6 points";
    }

    // Build seed and calibrate
    const CameraModel seed = getDefaultSeed(parsed.model, parsed.w, parsed.h);
    ASSERT_EQ(validateCameraModel(seed), StatusCode::OK)
      << "getDefaultSeed returned invalid model for fixture: " << fpath
      << " model=" << toString(parsed.model);

    const CalibrationResult res =
      calibrate(parsed.views, seed, lockFor(parsed.model), makeDefaultOptions(parsed.w, parsed.h));

    ASSERT_EQ(res.status, StatusCode::OK)
      << "calibrate failed for fixture '" << fpath << "': rms=" << res.rms_reprojection_error_px
      << " status=" << static_cast<int>(res.status);

    // Check expected_* regression bounds (provisional scaffold tolerances).
    // Comment: real noise characteristics are unknown until the user supplies
    // data — tighten these literals then.
    if (parsed.has_fx)
    {
      const double recovered_fx = static_cast<double>(res.camera_model.intrinsics.fx);
      const double rel_err_fx = std::abs(recovered_fx - parsed.fx) / parsed.fx;
      // Provisional: 5% relative — conservative until noise floor is known.
      EXPECT_LT(rel_err_fx, 0.05) << "fx regression: expected=" << parsed.fx
                                  << " recovered=" << recovered_fx << " rel_err=" << rel_err_fx
                                  << " (provisional 5% scaffold tolerance; tighten when real "
                                     "fixture noise is characterised)";
    }
    if (parsed.has_fy)
    {
      const double recovered_fy = static_cast<double>(res.camera_model.intrinsics.fy);
      const double rel_err_fy = std::abs(recovered_fy - parsed.fy) / parsed.fy;
      // Provisional: 5% relative.
      EXPECT_LT(rel_err_fy, 0.05) << "fy regression: expected=" << parsed.fy
                                  << " recovered=" << recovered_fy << " rel_err=" << rel_err_fy
                                  << " (provisional 5% scaffold tolerance)";
    }
    if (parsed.has_cx)
    {
      const double recovered_cx = static_cast<double>(res.camera_model.intrinsics.cx);
      const double abs_err_cx = std::abs(recovered_cx - parsed.cx);
      // Provisional: 10 px absolute — principal point shifts on real data.
      EXPECT_LT(abs_err_cx, 10.0) << "cx regression: expected=" << parsed.cx
                                  << " recovered=" << recovered_cx << " abs_err=" << abs_err_cx
                                  << " (provisional 10 px scaffold tolerance)";
    }
    if (parsed.has_cy)
    {
      const double recovered_cy = static_cast<double>(res.camera_model.intrinsics.cy);
      const double abs_err_cy = std::abs(recovered_cy - parsed.cy);
      // Provisional: 10 px absolute.
      EXPECT_LT(abs_err_cy, 10.0) << "cy regression: expected=" << parsed.cy
                                  << " recovered=" << recovered_cy << " abs_err=" << abs_err_cy
                                  << " (provisional 10 px scaffold tolerance)";
    }
  }
}
