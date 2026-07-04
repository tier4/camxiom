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

// Tests for CalibrationResult parameter uncertainty (C3) and — added in a
// later commit — automatic observability / degeneracy diagnostics (C4).
//
// C3 recovers, from the reduced normal equations JᵀJ at the solution (per-view
// extrinsics marginalised via the Schur complement, scaled by the residual
// variance), a 1-sigma standard deviation for each FREE intrinsic / distortion
// / projection parameter. The two properties that make this useful and that we
// pin here:
//   1. A perspective-diverse (tilted) view set makes the focal length well
//      determined -> small fx/fy std.
//   2. An all-fronto-parallel view set couples focal length with per-view depth
//      -> focal length is (near-)unobservable, which must show up either as a
//      large fx/fy std or as uncertainty_available == false (rank-deficient S).
// Plus: parameters held fixed by lock_flags are never listed, and non-OK /
// no-free-parameter cases leave the uncertainty empty.
//
// All tests are deterministic (std::mt19937 seed 42). Noise is required for a
// non-zero residual variance, so the recovery tests inject sigma = 0.3 px.

#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/default_seed.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// C5 internal pure helpers. These live under src/ (NOT the installed public
// API); the CMake target adds ${CMAKE_CURRENT_SOURCE_DIR}/src to this test's
// PRIVATE include path so we can unit-test them directly.
#include "calib/uncertainty_detail.hpp"
#include "optimizer/pnp/pnp_parameter_bounds.hpp"
#include "support/calib_test_fixtures.hpp"

using namespace camxiom;
using camxiom::calib::calibrate;
using camxiom::calib::CalibrationOptions;
using camxiom::calib::CalibrationResult;
using camxiom::calib::CalibrationView;
using camxiom::calib::PnpFlag;
using camxiom::test::buildCalibView;
using camxiom::test::makeCheckerboard3D;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;
using camxiom::test::rotMatZ;

namespace
{

CalibrationOptions makeDefaultOptions(int w = 640, int h = 480)
{
  CalibrationOptions opts;
  opts.image_width = w;
  opts.image_height = h;
  opts.max_iterations = 200;
  opts.apply_initial_value_bounds = false;
  return opts;
}

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

CameraModel64 buildGTKB4()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  // Inside the polynomial's monotone range (k1=0.01/k2=-0.005 folds at
  // ~2.64); the validator certifies the cap, see calib_intrinsics_test.
  m.projection.theta_max = 2.6;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = 0.01;
  m.distortion.coeffs[1] = -0.005;
  m.distortion.coeffs[2] = 0.0;
  m.distortion.coeffs[3] = 0.0;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  return m;
}

CameraModel64 buildGTMEI()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.xi = 1.0;
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

// Perspective-diverse poses (identical to the canonical kb4/mei/ds/eucm setup):
// tilts break the focal <-> depth coupling, so the focal is observable.
std::vector<CalibrationView> makeDiverseViews(
  const CameraModel64 &gt, const std::vector<Eigen::Vector3d> &world, double noise_sigma,
  std::mt19937 &rng
)
{
  const double d30 = 30.0 * constants::kPi / 180.0;
  const double d25 = 25.0 * constants::kPi / 180.0;
  const double d20 = 20.0 * constants::kPi / 180.0;
  const double d15 = 15.0 * constants::kPi / 180.0;

  struct Pose
  {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
  };
  const std::vector<Pose> poses = {
    {Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0, 0.0, 0.30)},
    {rotMatX(+d30), Eigen::Vector3d(0.0, -0.15, 0.30)},
    {rotMatY(-d30), Eigen::Vector3d(-0.15, 0.0, 0.35)},
    {rotMatX(+d25) * rotMatY(-d25), Eigen::Vector3d(0.1, 0.1, 0.35)},
    {rotMatX(+d20) * rotMatY(+d20) * rotMatZ(+d15), Eigen::Vector3d(-0.1, 0.12, 0.40)},
  };

  std::vector<CalibrationView> views(poses.size());
  for (std::size_t i = 0; i < poses.size(); ++i)
  {
    if (!buildCalibView(gt, poses[i].R, poses[i].t, world, noise_sigma, rng, views[i]))
    {
      return {};
    }
  }
  return views;
}

// All-fronto-parallel poses (R == Identity): every board plane is parallel to
// the image plane, so per-view depth t_z absorbs the focal scale and the focal
// length is (near-)unobservable.
std::vector<CalibrationView> makeFrontoParallelViews(
  const CameraModel64 &gt, const std::vector<Eigen::Vector3d> &world, double noise_sigma,
  std::mt19937 &rng
)
{
  const std::vector<Eigen::Vector3d> ts = {
    Eigen::Vector3d(0.00, 0.00, 0.50),  Eigen::Vector3d(0.05, 0.02, 0.60),
    Eigen::Vector3d(-0.04, 0.03, 0.70), Eigen::Vector3d(0.02, -0.04, 0.55),
    Eigen::Vector3d(-0.03, 0.02, 0.65),
  };
  std::vector<CalibrationView> views(ts.size());
  for (std::size_t i = 0; i < ts.size(); ++i)
  {
    if (!buildCalibView(gt, Eigen::Matrix3d::Identity(), ts[i], world, noise_sigma, rng, views[i]))
    {
      return {};
    }
  }
  return views;
}

bool hasLabel(const CalibrationResult &r, const std::string &name)
{
  return std::find(r.uncertainty_labels.begin(), r.uncertainty_labels.end(), name) !=
         r.uncertainty_labels.end();
}

bool hasVecLabel(const std::vector<std::string> &v, const std::string &name)
{
  return std::find(v.begin(), v.end(), name) != v.end();
}

double stdFor(const CalibrationResult &r, const std::string &name)
{
  for (std::size_t i = 0; i < r.uncertainty_labels.size(); ++i)
  {
    if (r.uncertainty_labels[i] == name)
    {
      return r.parameter_std[i];
    }
  }
  return -1.0;
}

}  // namespace

// ===========================================================================
// C3: uncertainty is populated for a well-conditioned pinhole calibration.
// ===========================================================================

TEST(CalibUncertainty, DiversePinholeUncertaintyAvailable)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);

  ASSERT_TRUE(res.uncertainty_available)
    << "diverse pinhole views should yield a full-rank normal matrix";
  // Pinhole with no distortion / no projection params: exactly fx,fy,cx,cy free.
  ASSERT_EQ(res.uncertainty_labels.size(), 4u);
  ASSERT_EQ(res.parameter_std.size(), 4u);
  EXPECT_TRUE(hasLabel(res, "fx"));
  EXPECT_TRUE(hasLabel(res, "fy"));
  EXPECT_TRUE(hasLabel(res, "cx"));
  EXPECT_TRUE(hasLabel(res, "cy"));

  for (double s : res.parameter_std)
  {
    EXPECT_TRUE(std::isfinite(s));
    EXPECT_GT(s, 0.0);
  }
  // Sanity band: with 5 diverse views of a 48-corner board at sigma=0.3 px the
  // focal is well determined; its std should be a handful of pixels, not
  // hundreds. 50 px is a loose upper guard (the tight assertion is the ratio
  // test below).
  EXPECT_LT(stdFor(res, "fx"), 50.0);
  EXPECT_LT(stdFor(res, "fy"), 50.0);
}

// ===========================================================================
// C3: the key detector — fronto-parallel views make the focal unobservable,
// which the uncertainty must reveal (large fx std, or unavailable).
// ===========================================================================

TEST(CalibUncertainty, FrontoParallelFocalPoorlyObservable)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);

  std::mt19937 rng_div(42U);
  const auto diverse = makeDiverseViews(gt, world, 0.3, rng_div);
  ASSERT_FALSE(diverse.empty());

  std::mt19937 rng_fp(42U);
  const auto fronto = makeFrontoParallelViews(gt, world, 0.3, rng_fp);
  ASSERT_FALSE(fronto.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const auto opts = makeDefaultOptions();

  const CalibrationResult res_div = calibrate(diverse, seed, PnpFlag::NONE, opts);
  ASSERT_EQ(res_div.status, StatusCode::OK);
  ASSERT_TRUE(res_div.uncertainty_available);
  const double diverse_fx_std = stdFor(res_div, "fx");
  ASSERT_GT(diverse_fx_std, 0.0);

  const CalibrationResult res_fp = calibrate(fronto, seed, PnpFlag::NONE, opts);
  // The solve itself can still "succeed" (residuals are small; focal trades off
  // against depth), but the focal is not determined.
  if (res_fp.uncertainty_available)
  {
    const double fp_fx_std = stdFor(res_fp, "fx");
    ASSERT_GT(fp_fx_std, 0.0);
    EXPECT_GT(fp_fx_std, 5.0 * diverse_fx_std)
      << "fronto-parallel focal std (" << fp_fx_std << ") should be far larger "
      << "than the perspective-diverse focal std (" << diverse_fx_std << ")";
  }
  else
  {
    SUCCEED() << "fronto-parallel focal fully unobservable: rank-deficient S "
                 "-> uncertainty_available=false (the strongest degeneracy signal)";
  }
}

// ===========================================================================
// C3: parameters held fixed by lock_flags are never listed.
// ===========================================================================

TEST(CalibUncertainty, LockedDistortionParamsNotListed)
{
  const CameraModel64 gt = buildGTKB4();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, 640, 480);
  // KB4 identifiability lock: k3=k4 fixed (FIX_DIST_2 | FIX_DIST_3).
  const PnpFlag flags = PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3;
  const CalibrationResult res = calibrate(views, seed, flags, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);

  ASSERT_TRUE(res.uncertainty_available);
  // Free: fx, fy, cx, cy, dist[0], dist[1]. Fixed: dist[2], dist[3].
  EXPECT_TRUE(hasLabel(res, "fx"));
  EXPECT_TRUE(hasLabel(res, "dist[0]"));
  EXPECT_TRUE(hasLabel(res, "dist[1]"));
  EXPECT_FALSE(hasLabel(res, "dist[2]")) << "dist[2] is locked by FIX_DIST_2 and must not appear";
  EXPECT_FALSE(hasLabel(res, "dist[3]")) << "dist[3] is locked by FIX_DIST_3 and must not appear";
  EXPECT_EQ(res.uncertainty_labels.size(), 6u);
  EXPECT_EQ(res.parameter_std.size(), res.uncertainty_labels.size());
}

TEST(CalibUncertainty, LockedProjectionParamsNotListed)
{
  const CameraModel64 gt = buildGTMEI();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, 640, 480);
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_PROJECTION_PARAMS, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);

  ASSERT_TRUE(res.uncertainty_available);
  // MEI with xi locked and no distortion: only fx,fy,cx,cy are free.
  EXPECT_FALSE(hasLabel(res, "xi")) << "xi is locked by FIX_PROJECTION_PARAMS and must not appear";
  EXPECT_EQ(res.uncertainty_labels.size(), 4u);
}

// ===========================================================================
// C3: no free intrinsics -> nothing to report.
// ===========================================================================

TEST(CalibUncertainty, NoFreeIntrinsicsLeavesUncertaintyEmpty)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.0, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  // Pose-only refinement: intrinsics fixed and pinhole has no distortion, so
  // there are zero free intrinsic parameters.
  const CalibrationResult res =
    calibrate(views, seed, PnpFlag::FIX_INTRINSICS, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);

  EXPECT_FALSE(res.uncertainty_available);
  EXPECT_TRUE(res.uncertainty_labels.empty());
  EXPECT_TRUE(res.parameter_std.empty());
  // Zero free parameters -> S is not built -> BOTH diagnostics unavailable
  // (fail-closed) and no near-bound to report.
  EXPECT_FALSE(res.observability_available);
  EXPECT_FALSE(res.observability_ok);
  EXPECT_FALSE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(res.parameters_at_or_near_bounds.empty());
}

// ===========================================================================
// C3: a non-OK (early-return) calibration leaves the uncertainty empty.
// ===========================================================================

TEST(CalibUncertainty, NonOkResultLeavesUncertaintyEmpty)
{
  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const std::vector<CalibrationView> empty_views;
  const CalibrationResult res = calibrate(empty_views, seed, PnpFlag::NONE, makeDefaultOptions());

  ASSERT_EQ(res.status, StatusCode::INVALID_INPUT);
  EXPECT_FALSE(res.uncertainty_available);
  EXPECT_TRUE(res.uncertainty_labels.empty());
  EXPECT_TRUE(res.parameter_std.empty());
  // Fail-closed defaults on an early return: everything unavailable.
  EXPECT_FALSE(res.observability_available);
  EXPECT_FALSE(res.observability_ok);
  EXPECT_FALSE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(res.parameters_at_or_near_bounds.empty());
}

// ===========================================================================
// C4: observability diagnostic — healthy vs degenerate observation sets.
// ===========================================================================

TEST(CalibUncertainty, DiversePinholeObservabilityOk)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);

  EXPECT_TRUE(res.observability_available)
    << "S was built and analysed, so the diagnostic value exists";
  EXPECT_TRUE(res.observability_ok)
    << "diverse pinhole should be well observable; cond=" << res.normalized_condition_number
    << " min_sv=" << res.min_singular_value;
  EXPECT_TRUE(res.underdetermined_parameters.empty());
  EXPECT_TRUE(std::isfinite(res.normalized_condition_number));
  EXPECT_GE(res.normalized_condition_number, 1.0);
  EXPECT_GT(res.min_singular_value, 0.0);
}

TEST(CalibUncertainty, FrontoParallelObservabilityFlagsFocal)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto fronto = makeFrontoParallelViews(gt, world, 0.3, rng);
  ASSERT_FALSE(fronto.empty());

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(fronto, seed, PnpFlag::NONE, makeDefaultOptions());

  // The reduced normal matrix is formed (per-view pose blocks are invertible),
  // so the C4 numbers are populated even though the focal is unobservable:
  // a SUCCESSFULLY diagnosed degeneracy (available=true, ok=false).
  EXPECT_TRUE(res.observability_available)
    << "S was built and analysed; a diagnosed degeneracy is still available";
  EXPECT_FALSE(res.observability_ok)
    << "fronto-parallel focal must be flagged; cond=" << res.normalized_condition_number
    << " min_sv=" << res.min_singular_value;
  ASSERT_FALSE(res.underdetermined_parameters.empty())
    << "the weakest direction should name at least one parameter";
  const bool names_focal = hasVecLabel(res.underdetermined_parameters, "fx") ||
                           hasVecLabel(res.underdetermined_parameters, "fy");
  EXPECT_TRUE(names_focal) << "fronto-parallel degeneracy couples the focal length with depth, so "
                           << "fx and/or fy should dominate the weakest direction";
  // The condition number should be large (weak/near-null direction).
  EXPECT_GT(res.normalized_condition_number, 100.0);
}

// ===========================================================================
// C5 helpers.
// ===========================================================================
namespace
{

namespace det = camxiom::calib::detail;
namespace optd = camxiom::optimizer::detail;
using camxiom::optimizer::PnpBound;

// Local copies of the production pose-Jacobian formulas (src/calib/intrinsics.cpp,
// anonymous namespace) so the two-stage Schur test can build the SAME joint
// normal matrix independently.
Eigen::Matrix3d c5_skew3(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

Eigen::Matrix3d c5_angleAxisPointJacobian(const Eigen::Vector3d &omega, const Eigen::Vector3d &p)
{
  const double theta2 = omega.squaredNorm();
  const Eigen::Vector3d w_cross_p = omega.cross(p);
  if (theta2 < 1e-20)
  {
    return -c5_skew3(p);
  }
  const double theta = std::sqrt(theta2);
  const double inv_t = 1.0 / theta;
  const double inv_t2 = inv_t * inv_t;
  const double st = std::sin(theta);
  const double ct = std::cos(theta);
  const double A = st * inv_t;
  const double B = (1.0 - ct) * inv_t2;
  const double C = (theta * ct - st) * inv_t * inv_t2;
  const double D = (theta * st - 2.0 * (1.0 - ct)) * inv_t2 * inv_t2;
  const Eigen::Vector3d ww_cross_p = omega.cross(w_cross_p);
  return -A * c5_skew3(p) - B * (c5_skew3(w_cross_p) + c5_skew3(omega) * c5_skew3(p)) +
         (C * w_cross_p + D * ww_cross_p) * omega.transpose();
}

// Ground-truth pinhole + RADTAN5 with moderate, well-observable distortion.
CameraModel64 buildGTPinholeRadtan()
{
  CameraModel64 m{};
  m.intrinsics.fx = 500.0;
  m.intrinsics.fy = 500.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = constants::kHalfPi;
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = -0.20;   // k1
  m.distortion.coeffs[1] = 0.08;    // k2
  m.distortion.coeffs[2] = 0.002;   // p1
  m.distortion.coeffs[3] = -0.002;  // p2
  m.distortion.coeffs[4] = 0.01;    // k3
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

// Pinhole + RADTAN5 seed with intrinsics near the GT and zero initial distortion.
CameraModel buildRadtan5Seed()
{
  CameraModel m{};
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(constants::kHalfPi);
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5U;
  m.distortion.coeffs.fill(0.0f);
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

// A single tilted mini-view of exactly the DLT minimum (6) corners.
CalibrationView makeMiniView(const CameraModel64 &gt, std::mt19937 &rng, double noise_sigma)
{
  const auto world = makeCheckerboard3D(2, 3, 0.05);  // 6 coplanar corners
  CalibrationView v;
  buildCalibView(
    gt, rotMatX(0.20) * rotMatY(-0.15), Eigen::Vector3d(0.02, -0.01, 0.35), world, noise_sigma, rng,
    v
  );
  return v;
}

}  // namespace

// ===========================================================================
// C5 ④(b): the weak-subspace projector aggregation is basis-invariant.
// This is the most reliable guard (calibration rarely produces two clean weak
// modes), so it is a direct pure-helper unit test (#5).
// ===========================================================================

TEST(CalibUncertaintyC5, WeakSubspaceProjectorIsBasisInvariant)
{
  // A 6-parameter space with a 3-dimensional weak subspace V (6x3). The columns
  // need NOT be orthonormal: P = (V Q)(V Q)ᵀ = V Vᵀ holds for any orthogonal Q.
  Eigen::MatrixXd V(6, 3);
  V << 0.9, 0.1, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.05, 0.1, 0.1, 0.02, 0.0, 0.3, 0.9, 0.05, 0.0, 0.1;
  const std::vector<std::string> labels = {"fx", "fy", "cx", "cy", "dist[0]", "dist[1]"};

  // Random orthogonal Q (3x3) from the QR of a fixed-seed Gaussian matrix.
  std::mt19937 rng(12345U);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd G(3, 3);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) G(r, c) = nd(rng);
  const Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(G).householderQ();
  ASSERT_NEAR((Q * Q.transpose() - Eigen::MatrixXd::Identity(3, 3)).norm(), 0.0, 1e-12);

  const det::WeakSubspaceScores a = det::aggregateWeakSubspace(V, labels);
  const det::WeakSubspaceScores b = det::aggregateWeakSubspace((V * Q).eval(), labels);

  ASSERT_EQ(a.score.size(), b.score.size());
  for (Eigen::Index i = 0; i < a.score.size(); ++i)
  {
    EXPECT_NEAR(a.score(i), b.score(i), 1e-12) << "score row " << i;
  }
  EXPECT_EQ(a.underdetermined, b.underdetermined)
    << "the reported labels must not depend on the eigenbasis";

  // The score is the projector diagonal diag(V Vᵀ).
  const Eigen::MatrixXd P = V * V.transpose();
  for (Eigen::Index i = 0; i < 6; ++i)
  {
    EXPECT_NEAR(a.score(i), P(i, i), 1e-12);
  }
  // With this V the dominant contributors (fx, fy, dist[0]) clear the threshold.
  EXPECT_TRUE(hasVecLabel(a.underdetermined, "fx"));
  EXPECT_TRUE(hasVecLabel(a.underdetermined, "fy"));
  EXPECT_TRUE(hasVecLabel(a.underdetermined, "dist[0]"));
  EXPECT_FALSE(hasVecLabel(a.underdetermined, "cx"));
  EXPECT_FALSE(hasVecLabel(a.underdetermined, "cy"));
}

TEST(CalibUncertaintyC5, WeakSubspaceEmptyBasisReportsNothing)
{
  const std::vector<std::string> labels = {"fx", "fy"};
  const det::WeakSubspaceScores a = det::aggregateWeakSubspace(Eigen::MatrixXd(2, 0), labels);
  EXPECT_EQ(a.score.size(), 2);
  EXPECT_EQ(a.score(0), 0.0);
  EXPECT_TRUE(a.underdetermined.empty());
}

// ===========================================================================
// C5 ⑤: near-bound proximity — pure-helper coverage (lower / upper / interior /
// one-sided / tolerance boundary) and the shared bound descriptor.
// ===========================================================================

TEST(CalibUncertaintyC5, ClassifyBoundProximityTwoSided)
{
  const optd::ScalarBound b{true, true, 0.0, 1.0};              // alpha in [0, 1]
  EXPECT_FALSE(det::classifyBoundProximity(0.5, b).anyNear());  // interior
  {
    const det::BoundProximity p = det::classifyBoundProximity(1.0, b);
    EXPECT_TRUE(p.near_upper);
    EXPECT_FALSE(p.near_lower);
  }
  {
    const det::BoundProximity p = det::classifyBoundProximity(0.0, b);
    EXPECT_TRUE(p.near_lower);
    EXPECT_FALSE(p.near_upper);
  }
}

TEST(CalibUncertaintyC5, ClassifyBoundProximityToleranceBoundary)
{
  const optd::ScalarBound b{true, true, 0.0, 1.0};
  // For x ~ 1, max(|upper|, |x|) == 1, so tol_upper == kNearBoundRelTol.
  const double tol = det::kNearBoundRelTol;
  // Exactly on the bound is near.
  EXPECT_TRUE(det::classifyBoundProximity(1.0, b).near_upper);
  // Clearly inside the tolerance band is near.
  EXPECT_TRUE(det::classifyBoundProximity(1.0 - 0.9 * tol, b).near_upper);
  // Clearly outside the tolerance band is not near.
  EXPECT_FALSE(det::classifyBoundProximity(1.0 - 1.1 * tol, b).near_upper);
  EXPECT_FALSE(det::classifyBoundProximity(0.5, b).near_upper);
}

TEST(CalibUncertaintyC5, ClassifyBoundProximityOneSided)
{
  // EUCM beta: lower bound only (has_upper == false).
  const optd::ScalarBound beta{true, false, optd::kEucmBetaLower, 0.0};
  {
    const det::BoundProximity p = det::classifyBoundProximity(1.0, beta);
    EXPECT_FALSE(p.near_upper) << "a missing upper side is never 'near'";
    EXPECT_FALSE(p.near_lower);
  }
  {
    const det::BoundProximity p = det::classifyBoundProximity(optd::kEucmBetaLower, beta);
    EXPECT_TRUE(p.near_lower);
    EXPECT_FALSE(p.near_upper);
  }
}

TEST(CalibUncertaintyC5, BoundDescriptorEucmBetaIsLowerOnly)
{
  CameraModel m{};
  m.projection.type = ProjectionModelType::EUCM;
  const optd::CalibrationParameterBounds b = optd::computeCalibrationParameterBounds(
    m, PnpFlag::NONE, PnpBound::createLowerBound(), PnpBound::createUpperBound()
  );
  // alpha (slot 1): two-sided [0, 1].
  EXPECT_TRUE(b.projection[1].has_lower);
  EXPECT_TRUE(b.projection[1].has_upper);
  EXPECT_DOUBLE_EQ(b.projection[1].lower, 0.0);
  EXPECT_DOUBLE_EQ(b.projection[1].upper, 1.0);
  // beta (slot 2): lower only.
  EXPECT_TRUE(b.projection[2].has_lower);
  EXPECT_FALSE(b.projection[2].has_upper);
  EXPECT_DOUBLE_EQ(b.projection[2].lower, optd::kEucmBetaLower);
  // xi (slot 0): unused by EUCM -> unbounded.
  EXPECT_FALSE(b.projection[0].has_lower);
  EXPECT_FALSE(b.projection[0].has_upper);
}

TEST(CalibUncertaintyC5, BoundDescriptorDoubleSphereAndFixedFlags)
{
  CameraModel m{};
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  const optd::CalibrationParameterBounds b = optd::computeCalibrationParameterBounds(
    m, PnpFlag::NONE, PnpBound::createLowerBound(), PnpBound::createUpperBound()
  );
  EXPECT_TRUE(b.projection[0].has_lower && b.projection[0].has_upper);  // xi
  EXPECT_DOUBLE_EQ(b.projection[0].lower, -1.0 + optd::kDoubleSphereXiMargin);
  EXPECT_DOUBLE_EQ(b.projection[0].upper, 1.0 - optd::kDoubleSphereXiMargin);
  EXPECT_TRUE(b.projection[1].has_lower && b.projection[1].has_upper);  // alpha
  EXPECT_FALSE(b.projection[2].has_lower);                              // beta unused by DS
  EXPECT_TRUE(b.focal_lengths[0].has_lower && b.focal_lengths[0].has_upper);

  // Fixed blocks carry no bounds (so the near-bound diagnostic never lists them).
  const optd::CalibrationParameterBounds bf = optd::computeCalibrationParameterBounds(
    m, PnpFlag::FIX_FOCAL_LENGTHS | PnpFlag::FIX_PROJECTION_PARAMS, PnpBound::createLowerBound(),
    PnpBound::createUpperBound()
  );
  EXPECT_FALSE(bf.focal_lengths[0].has_lower);
  EXPECT_FALSE(bf.focal_lengths[0].has_upper);
  EXPECT_FALSE(bf.projection[0].has_lower);
  EXPECT_FALSE(bf.projection[1].has_lower);
}

// ===========================================================================
// C5 #6: two-stage Schur identity — the diagnostic covariance equals the
// intrinsic block of the full joint-normal inverse.
//   (a) S⁻¹ (Schur, σ²-free linear algebra) == (H⁻¹)_AA
//   (b) σ²·(H⁻¹)_AA diagonal == production parameter_std²  (σ²/DOF path)
// Interior, well-conditioned, OLS pinhole+RADTAN5 (bounds not active).
// ===========================================================================

TEST(CalibUncertaintyC5, SchurIdentityMatchesJointNormalAndProduction)
{
  const CameraModel64 gt = buildGTPinholeRadtan();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(7U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  const CameraModel seed = buildRadtan5Seed();
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());
  ASSERT_EQ(res.status, StatusCode::OK);
  ASSERT_TRUE(res.uncertainty_available);
  ASSERT_EQ(res.uncertainty_labels.size(), 9u)
    << "pinhole+RADTAN5, no locks: fx,fy,cx,cy,dist[0..4]";

  // Interior sanity (#6): observable, well-conditioned, no parameter on a bound.
  EXPECT_TRUE(res.observability_available);
  EXPECT_TRUE(res.observability_ok)
    << "cond=" << res.normalized_condition_number << " min_sv=" << res.min_singular_value;
  EXPECT_FALSE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(res.parameters_at_or_near_bounds.empty());

  // --- Rebuild the joint normal H and the Schur complement S independently. ---
  const int p = 9;
  const std::size_t nv = views.size();
  const int dim = p + 6 * static_cast<int>(nv);
  const CameraModel64 m64 = toCameraModel64(res.camera_model);
  const double fx = static_cast<double>(res.camera_model.intrinsics.fx);
  const double fy = static_cast<double>(res.camera_model.intrinsics.fy);
  const double sk = static_cast<double>(res.camera_model.intrinsics.skew);

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
  std::vector<Eigen::MatrixXd> Bv(nv, Eigen::MatrixXd::Zero(p, 6));
  std::vector<Eigen::Matrix<double, 6, 6>> Dv(nv, Eigen::Matrix<double, 6, 6>::Zero());
  double rss = 0.0;
  std::size_t n_valid = 0;
  long n_active = 0;

  for (std::size_t v = 0; v < nv; ++v)
  {
    const Eigen::Matrix3d &Rv = res.per_view_rotations[v];
    const Eigen::Vector3d &tv = res.per_view_translations[v];
    const Eigen::AngleAxisd aa(Rv);
    const Eigen::Vector3d omega = aa.axis() * aa.angle();
    bool active = false;
    const auto &view = views[v];
    for (std::size_t j = 0; j < view.world_points.size(); ++j)
    {
      const Eigen::Vector3d &Pw = view.world_points[j];
      const Eigen::Vector3d p_cam = Rv * Pw + tv;
      const camxiom::FullProjectionJacobian64 fj =
        camxiom::rayToPixelWithFullJacobian64(m64, p_cam);
      if (fj.status != StatusCode::OK)
      {
        continue;
      }
      const Eigen::Vector2d &obs = view.image_points[j];
      const double du = obs.x() - fj.pixel.u;
      const double dv = obs.y() - fj.pixel.v;
      rss += du * du + dv * dv;
      ++n_valid;
      active = true;

      Eigen::Matrix<double, 2, 9> Ji = Eigen::Matrix<double, 2, 9>::Zero();
      Ji(0, 0) = -fj.xd;
      Ji(1, 1) = -fj.yd;
      Ji(0, 2) = -1.0;
      Ji(1, 3) = -1.0;
      for (int k = 0; k < 5; ++k)
      {
        const auto kk = static_cast<std::size_t>(k);
        Ji(0, 4 + k) = -(fx * fj.dxd_ddist[kk] + sk * fj.dyd_ddist[kk]);
        Ji(1, 4 + k) = -(fy * fj.dyd_ddist[kk]);
      }
      const Eigen::Matrix3d J_aa = c5_angleAxisPointJacobian(omega, Pw);
      Eigen::Matrix<double, 2, 6> Je;
      Je.block<2, 3>(0, 0) = -(fj.J_point * J_aa);
      Je.block<2, 3>(0, 3) = -fj.J_point;

      A.noalias() += Ji.transpose() * Ji;
      Bv[v].noalias() += Ji.transpose() * Je;
      Dv[v].noalias() += Je.transpose() * Je;

      const int pc = p + 6 * static_cast<int>(v);
      H.block(0, 0, p, p).noalias() += Ji.transpose() * Ji;
      H.block(0, pc, p, 6).noalias() += Ji.transpose() * Je;
      H.block(pc, 0, 6, p).noalias() += Je.transpose() * Ji;
      H.block(pc, pc, 6, 6).noalias() += Je.transpose() * Je;
    }
    if (active)
    {
      ++n_active;
    }
  }
  ASSERT_EQ(n_valid, world.size() * nv) << "all corners must remain valid at the solution";

  Eigen::MatrixXd S = A;
  for (std::size_t v = 0; v < nv; ++v)
  {
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Dv[v]);
    ASSERT_EQ(ldlt.info(), Eigen::Success);
    S.noalias() -= Bv[v] * ldlt.solve(Bv[v].transpose());
  }
  const Eigen::MatrixXd Sinv = S.inverse();
  const Eigen::MatrixXd Hinv_AA = H.inverse().topLeftCorner(p, p);

  // (a) σ²-free identity: the whole S⁻¹ (Schur assembly: column order, off-
  // diagonals, block partition) equals the intrinsic block of the joint inverse.
  const double rel_err = (Sinv - Hinv_AA).norm() / Hinv_AA.norm();
  EXPECT_LT(rel_err, 1e-6) << "Schur S^-1 must equal (H^-1)_AA; rel_err=" << rel_err;

  // (b) σ²/DOF path: σ²·(H⁻¹)_AA diagonal == production parameter_std².
  const long d_pose = 6L * n_active;
  const long dof = 2L * static_cast<long>(n_valid) - p - d_pose;
  ASSERT_GT(dof, 0);
  const double sigma2 = rss / static_cast<double>(dof);
  for (int i = 0; i < p; ++i)
  {
    const double expected_var = sigma2 * Hinv_AA(i, i);
    const double prod_std = res.parameter_std[static_cast<std::size_t>(i)];
    const double prod_var = prod_std * prod_std;
    EXPECT_NEAR(prod_var, expected_var, 1e-6 * std::abs(expected_var) + 1e-15)
      << "variance mismatch for " << res.uncertainty_labels[static_cast<std::size_t>(i)];
  }
}

// ===========================================================================
// C5 #7: state transitions.
// ===========================================================================

// estimate_uncertainty == false skips C3, C4 AND the near-bound proximity.
TEST(CalibUncertaintyC5, EstimateUncertaintyFalseSkipsAllDiagnostics)
{
  const CameraModel64 gt = buildGTPinhole();
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  CalibrationOptions opts = makeDefaultOptions();
  opts.estimate_uncertainty = false;

  const CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, opts);
  ASSERT_EQ(res.status, StatusCode::OK);

  EXPECT_FALSE(res.uncertainty_available);
  EXPECT_FALSE(res.observability_available);
  EXPECT_FALSE(res.observability_ok);
  EXPECT_FALSE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(res.parameters_at_or_near_bounds.empty());
}

// dof <= 0 (single mini view, 6 corners, 9 free params): C3 cannot form a
// covariance (fail-closed, no NaN / negative variance), while C4 may still
// report on S (it is σ²-independent).
TEST(CalibUncertaintyC5, DofNonPositiveLeavesUncertaintyUnavailable)
{
  const CameraModel64 gt = buildGTPinholeRadtan();
  std::mt19937 rng(3U);
  const std::vector<CalibrationView> views = {makeMiniView(gt, rng, 0.0)};
  ASSERT_EQ(views[0].world_points.size(), 6u);

  const CameraModel seed = buildRadtan5Seed();
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, makeDefaultOptions());
  ASSERT_TRUE(res.status == StatusCode::OK || res.status == StatusCode::NON_CONVERGED)
    << "status=" << static_cast<int>(res.status);

  // 2*6 - 9 - 6 = -3 <= 0 -> no covariance, no leaked NaNs.
  EXPECT_FALSE(res.uncertainty_available);
  EXPECT_TRUE(res.parameter_std.empty());
  EXPECT_TRUE(res.uncertainty_labels.empty());
  EXPECT_TRUE(std::isfinite(res.min_singular_value));
  EXPECT_FALSE(std::isnan(res.normalized_condition_number));
}

// C5 ⑤ end-to-end: the focal length capped just below its true value parks on
// its upper bound and is flagged, while the interior principal point is not.
// (Focal capping is used rather than pp capping because a focal error is
// cleanly absorbed by per-view depth, leaving cx/cy undisturbed and interior.)
TEST(CalibUncertaintyC5, NearBoundFocalParkedAtUpperBound)
{
  const CameraModel64 gt = buildGTPinhole();  // fx=fy=500, cx=320, cy=240
  const auto world = makeCheckerboard3D(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeDiverseViews(gt, world, 0.3, rng);
  ASSERT_FALSE(views.empty());

  CalibrationOptions opts = makeDefaultOptions();
  // apply_initial_value_bounds boxes focal AND (legacy) pp to seed*(1 +/- r).
  opts.apply_initial_value_bounds = true;
  opts.bound_relative_tolerance = 0.10;

  // Seed focal 10% below truth: the focal upper bound (seed*1.1 = 495) sits just
  // below the true 500, so focal parks on it. Seed pp on truth so its wide band
  // (+/-10% => cx in [288,352], cy in [216,264]) keeps cx/cy interior.
  CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  seed.intrinsics.fx = 450.0f;
  seed.intrinsics.fy = 450.0f;
  seed.intrinsics.cx = 320.0f;
  seed.intrinsics.cy = 240.0f;

  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, opts);
  ASSERT_TRUE(res.status == StatusCode::OK || res.status == StatusCode::NON_CONVERGED);

  EXPECT_TRUE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(hasVecLabel(res.parameters_at_or_near_bounds, "fx"))
    << "fx capped at its upper bound must be flagged (fx=" << res.camera_model.intrinsics.fx << ")";
  EXPECT_FALSE(hasVecLabel(res.parameters_at_or_near_bounds, "cx"))
    << "cx (interior, centred on truth) must not be flagged (cx=" << res.camera_model.intrinsics.cx
    << ")";
  EXPECT_FALSE(hasVecLabel(res.parameters_at_or_near_bounds, "cy"))
    << "cy (interior, centred on truth) must not be flagged (cy=" << res.camera_model.intrinsics.cy
    << ")";
}

// C5 #11 end-to-end: near-bound is returned EVEN when C3 is unavailable. A
// single mini view (dof <= 0) leaves uncertainty_available == false, yet the
// focal parked on its upper bound is still reported.
TEST(CalibUncertaintyC5, NearBoundReturnedEvenWhenUncertaintyUnavailable)
{
  const CameraModel64 gt = buildGTPinholeRadtan();  // fx=fy=500
  std::mt19937 rng(5U);
  const std::vector<CalibrationView> views = {makeMiniView(gt, rng, 0.0)};

  CalibrationOptions opts = makeDefaultOptions();
  // A focal band NARROWER than the near-bound tolerance itself: the half-width
  // (500 * 5e-4 = 0.25) is below tol (~kNearBoundRelTol * 500 = 0.5), so wherever
  // the (sparse, under-determined) solve leaves the focal inside the band, it is
  // provably within tolerance of a bound. This makes the near-bound signal
  // deterministic without depending on the focal being strongly pulled.
  opts.apply_initial_value_bounds = true;
  opts.bound_relative_tolerance = 5e-4;

  const CameraModel seed = buildRadtan5Seed();  // focal band centred on 500
  const CalibrationResult res = calibrate(views, seed, PnpFlag::NONE, opts);
  ASSERT_TRUE(res.status == StatusCode::OK || res.status == StatusCode::NON_CONVERGED);

  EXPECT_FALSE(res.uncertainty_available) << "dof<=0 -> no covariance";
  EXPECT_TRUE(res.uncertainty_has_parameters_near_bounds);
  EXPECT_TRUE(hasVecLabel(res.parameters_at_or_near_bounds, "fx"))
    << "near-bound must survive an unavailable covariance (fx=" << res.camera_model.intrinsics.fx
    << ")";
}
