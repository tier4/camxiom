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

// Tests for the soft principal-point prior mechanism
// (CalibrationOptions::apply_principal_point_prior /
//  PnpSolverOptions::principal_point_prior).
//
// Motivating scenario: a narrow-FOV (telephoto) camera. At ~30 deg FOV the
// principal point is near-unobservable from planar views — the likelihood is
// flat over tens of pixels — so an unconstrained solve drifts cx/cy along the
// degenerate valley until it parks on a box bound. The prior adds
// ((cx-ref)/sigma, (cy-ref)/sigma) to the objective so the MAP solution stays
// interior, while wide-FOV (informative) data must remain unaffected.
//
// All tests are deterministic (std::mt19937 fixed seeds). No rand().

#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "support/calib_test_fixtures.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

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

namespace
{

// ---------------------------------------------------------------------------
// Narrow-FOV fixture: 1920x1080, fx=fy=3800 (diagonal FOV ~= 32 deg),
// PINHOLE + RADTAN5 with tangential and k3 at zero (the physical reality the
// prior + lock configuration targets).
// ---------------------------------------------------------------------------

constexpr int kNarrowW = 1920;
constexpr int kNarrowH = 1080;
constexpr float kNarrowFx = 3800.0F;

CameraModel makeRadtan5Pinhole(
  float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2, float k3
)
{
  CameraModel m{};
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fy;
  m.intrinsics.cx = cx;
  m.intrinsics.cy = cy;
  m.intrinsics.skew = 0.0F;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(constants::kHalfPi);
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.count = 5U;
  m.distortion.coeffs.fill(0.0F);
  m.distortion.coeffs[0] = k1;
  m.distortion.coeffs[1] = k2;
  m.distortion.coeffs[2] = p1;
  m.distortion.coeffs[3] = p2;
  m.distortion.coeffs[4] = k3;
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

CameraModel makeNarrowGT()
{
  return makeRadtan5Pinhole(
    kNarrowFx, kNarrowFx, kNarrowW / 2.0F, kNarrowH / 2.0F, -0.05F, 0.10F, 0.0F, 0.0F, 0.0F
  );
}

/// Seed for the narrow solves: sane focal (a few % off GT, as a homography /
/// yaml init would supply), principal point at the image centre, zero
/// distortion.
CameraModel makeNarrowSeed()
{
  return makeRadtan5Pinhole(
    3650.0F, 3650.0F, kNarrowW / 2.0F, kNarrowH / 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F
  );
}

/// Board poses for the narrow camera: a 0.7 x 0.5 m board (6x8 corners,
/// 0.1 m pitch, centred on its own origin) swept over a 3x3 image-position
/// grid at ~4.5 m with alternating tilts, plus near/far high-tilt views.
std::vector<CalibrationView> makeNarrowViews(double noise_sigma, unsigned rng_seed)
{
  const CameraModel64 gt64 = toCameraModel64(makeNarrowGT());

  // Centre the board on its own origin so tilts rotate about the middle.
  std::vector<Eigen::Vector3d> board = makeCheckerboard3D(6, 8, 0.1);
  for (auto &p : board)
  {
    p -= Eigen::Vector3d(0.35, 0.25, 0.0);
  }

  const double deg = constants::kPi / 180.0;
  std::vector<Eigen::Matrix3d> Rs;
  std::vector<Eigen::Vector3d> ts;
  const double xs[3] = {-0.55, 0.0, 0.55};
  const double ys[3] = {-0.28, 0.0, 0.28};
  int k = 0;
  for (double dy : ys)
  {
    for (double dx : xs)
    {
      // Alternate tilt axis/sign across the grid for pose diversity.
      const double tilt = ((k % 2) == 0 ? +20.0 : -20.0) * deg;
      Rs.push_back(((k % 4) < 2) ? rotMatX(tilt) : rotMatY(tilt));
      ts.emplace_back(dx, dy, 4.5);
      ++k;
    }
  }
  // Near / far strongly tilted views.
  Rs.push_back(rotMatX(+35.0 * deg));
  ts.emplace_back(0.0, -0.1, 3.5);
  Rs.push_back(rotMatY(-35.0 * deg));
  ts.emplace_back(-0.3, 0.0, 5.5);
  Rs.push_back(rotMatX(-30.0 * deg) * rotMatY(+15.0 * deg));
  ts.emplace_back(0.3, 0.15, 5.0);

  std::mt19937 rng(rng_seed);
  std::vector<CalibrationView> views(Rs.size());
  for (std::size_t i = 0; i < Rs.size(); ++i)
  {
    if (!buildCalibView(gt64, Rs[i], ts[i], board, noise_sigma, rng, views[i]))
    {
      return {};
    }
  }
  return views;
}

CalibrationOptions makeNarrowOptions()
{
  CalibrationOptions opts;
  opts.image_width = kNarrowW;
  opts.image_height = kNarrowH;
  opts.max_iterations = 200;
  // The bound configuration the ICC uses: principal point boxed to
  // centre +/- 10%.
  opts.apply_principal_point_bounds = true;
  opts.principal_point_reference_x = kNarrowW / 2.0;
  opts.principal_point_reference_y = kNarrowH / 2.0;
  opts.principal_point_bound_relative_tolerance = 0.10;
  return opts;
}

CalibrationOptions withPrior(CalibrationOptions opts, double sigma_px)
{
  opts.apply_principal_point_prior = true;
  opts.principal_point_prior_reference_x = opts.principal_point_reference_x;
  opts.principal_point_prior_reference_y = opts.principal_point_reference_y;
  opts.principal_point_prior_sigma_px = sigma_px;
  return opts;
}

/// p1, p2, k3 pinned (at the zero they hold in the seed): the reduced model
/// the narrow-FOV policy solves.
constexpr PnpFlag kNarrowLocks = PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3 | PnpFlag::FIX_DIST_4;

constexpr double kPriorSigmaPx = 19.2;  // 1% of image width

double stdOf(const CalibrationResult &r, const std::string &label)
{
  for (std::size_t i = 0; i < r.uncertainty_labels.size(); ++i)
  {
    if (r.uncertainty_labels[i] == label)
    {
      return r.parameter_std[i];
    }
  }
  ADD_FAILURE() << "label not found in uncertainty_labels: " << label;
  return std::numeric_limits<double>::quiet_NaN();
}

// ===========================================================================
// 1. Narrow FOV + prior: principal point stays interior (never parks on the
//    box bound) and the fit quality cost of the prior is negligible.
// ===========================================================================
TEST(CalibPriorTest, NarrowFovPriorKeepsPrincipalPointInterior)
{
  const std::vector<CalibrationView> views = makeNarrowViews(0.3, 42U);
  ASSERT_FALSE(views.empty());

  const CalibrationResult without =
    calibrate(views, makeNarrowSeed(), kNarrowLocks, makeNarrowOptions());
  const CalibrationResult with =
    calibrate(views, makeNarrowSeed(), kNarrowLocks, withPrior(makeNarrowOptions(), kPriorSigmaPx));

  ASSERT_EQ(with.status, StatusCode::OK);
  EXPECT_NEAR(with.camera_model.intrinsics.cx, kNarrowW / 2.0, 20.0);
  EXPECT_NEAR(with.camera_model.intrinsics.cy, kNarrowH / 2.0, 20.0);
  EXPECT_FALSE(with.uncertainty_has_parameters_near_bounds)
    << "prior-backed solution must sit interior to the box bounds";
  EXPECT_TRUE(with.parameters_at_or_near_bounds.empty());

  // The prior may cost a little reprojection accuracy on degenerate data, but
  // only a little (the incident dataset measured ~0.02 px).
  ASSERT_EQ(without.status, StatusCode::OK);
  EXPECT_LE(
    with.rms_reprojection_error_px, without.rms_reprojection_error_px + 0.05
  );
}

// ===========================================================================
// 2. Wide FOV (informative data): the prior must not move the solution.
// ===========================================================================
TEST(CalibPriorTest, PriorLeavesWideFovSolutionUnchanged)
{
  // Wide pinhole+RadTan5: 640x480, fx=fy=500 (diag FOV ~77 deg), mild k1/k2.
  const CameraModel gt =
    makeRadtan5Pinhole(500.0F, 500.0F, 322.0F, 238.0F, -0.05F, 0.02F, 0.0F, 0.0F, 0.0F);
  const CameraModel64 gt64 = toCameraModel64(gt);

  std::vector<Eigen::Vector3d> board = makeCheckerboard3D(6, 8, 0.03);
  for (auto &p : board)
  {
    p -= Eigen::Vector3d(0.105, 0.075, 0.0);
  }
  const double deg = constants::kPi / 180.0;
  const std::vector<Eigen::Matrix3d> Rs = {
    Eigen::Matrix3d::Identity(), rotMatX(30.0 * deg), rotMatY(-30.0 * deg),
    rotMatX(25.0 * deg) * rotMatY(-25.0 * deg), rotMatX(-20.0 * deg) * rotMatY(20.0 * deg)};
  const std::vector<Eigen::Vector3d> ts = {
    {0.0, 0.0, 0.35}, {0.0, -0.05, 0.35}, {-0.05, 0.0, 0.40}, {0.05, 0.05, 0.40},
    {-0.05, 0.05, 0.45}};

  std::mt19937 rng(7U);
  std::vector<CalibrationView> views(Rs.size());
  for (std::size_t i = 0; i < Rs.size(); ++i)
  {
    ASSERT_TRUE(buildCalibView(gt64, Rs[i], ts[i], board, 0.2, rng, views[i]));
  }

  const CameraModel seed =
    makeRadtan5Pinhole(480.0F, 480.0F, 320.0F, 240.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);

  CalibrationOptions opts;
  opts.image_width = 640;
  opts.image_height = 480;
  opts.apply_principal_point_bounds = true;
  opts.principal_point_reference_x = 320.0;
  opts.principal_point_reference_y = 240.0;

  const CalibrationResult without = calibrate(views, seed, kNarrowLocks, opts);
  const CalibrationResult with =
    calibrate(views, seed, kNarrowLocks, withPrior(opts, 6.4));  // 1% of width

  ASSERT_EQ(without.status, StatusCode::OK);
  ASSERT_EQ(with.status, StatusCode::OK);
  ASSERT_TRUE(without.uncertainty_available);

  // "Unchanged" in the statistical sense: the prior may not move any
  // parameter by more than the data's own 1-sigma uncertainty (i.e. the
  // shift is immaterial next to what the data itself can resolve), and the
  // fit quality must be untouched.
  EXPECT_LE(
    std::abs(with.camera_model.intrinsics.fx - without.camera_model.intrinsics.fx),
    stdOf(without, "fx")
  );
  EXPECT_LE(
    std::abs(with.camera_model.intrinsics.cx - without.camera_model.intrinsics.cx),
    stdOf(without, "cx")
  );
  EXPECT_LE(
    std::abs(with.camera_model.intrinsics.cy - without.camera_model.intrinsics.cy),
    stdOf(without, "cy")
  );
  EXPECT_NEAR(
    with.rms_reprojection_error_px, without.rms_reprojection_error_px, 1e-2
  );
}

// ===========================================================================
// 3. PnpResult::rmse stays a pure reprojection statistic even when the prior
//    residual at the solution is large (tug-of-war between a strongly
//    informative wide-FOV dataset and a deliberately offset, tight prior).
// ===========================================================================
TEST(CalibPriorTest, PriorRmsExcludesPriorResidual)
{
  const CameraModel gt =
    makeRadtan5Pinhole(500.0F, 500.0F, 320.0F, 240.0F, -0.05F, 0.02F, 0.0F, 0.0F, 0.0F);
  const CameraModel64 gt64 = toCameraModel64(gt);

  std::vector<Eigen::Vector3d> board = makeCheckerboard3D(6, 8, 0.03);
  for (auto &p : board)
  {
    p -= Eigen::Vector3d(0.105, 0.075, 0.0);
  }
  const double deg = constants::kPi / 180.0;
  const std::vector<Eigen::Matrix3d> Rs = {
    Eigen::Matrix3d::Identity(), rotMatX(30.0 * deg), rotMatY(-30.0 * deg),
    rotMatX(25.0 * deg) * rotMatY(-25.0 * deg), rotMatX(-20.0 * deg) * rotMatY(20.0 * deg)};
  const std::vector<Eigen::Vector3d> ts = {
    {0.0, 0.0, 0.35}, {0.0, -0.05, 0.35}, {-0.05, 0.0, 0.40}, {0.05, 0.05, 0.40},
    {-0.05, 0.05, 0.45}};

  std::mt19937 rng(11U);
  std::vector<CalibrationView> views(Rs.size());
  optimizer::ObjectPointSets obj_sets(Rs.size());
  optimizer::ImagePointSets img_sets(Rs.size());
  optimizer::PnpInitialGuess guess;
  for (std::size_t i = 0; i < Rs.size(); ++i)
  {
    ASSERT_TRUE(buildCalibView(gt64, Rs[i], ts[i], board, 0.2, rng, views[i]));
    obj_sets[i] = views[i].world_points;
    img_sets[i] = views[i].image_points;
    // Seed poses at GT (angle-axis of R) — this test targets the objective
    // bookkeeping, not pose recovery.
    const Eigen::AngleAxisd aa(Rs[i]);
    guess.rvecs.push_back(aa.axis() * aa.angle());
    guess.tvecs.push_back(ts[i]);
  }
  guess.camera_model = gt;

  optimizer::PnpSolverOptions opts;
  // Deliberately WRONG reference, very tight: the prior residual at the
  // (data-dominated) solution is ~ tens of sigmas. A contaminated rmse would
  // jump by an order of magnitude; the clean one stays at the noise level.
  opts.principal_point_prior.enabled = true;
  opts.principal_point_prior.reference_x = 320.0 + 50.0;
  opts.principal_point_prior.reference_y = 240.0;
  opts.principal_point_prior.sigma_px = 1.0;

  optimizer::PnpSolver solver;
  optimizer::PnpResult result;
  ASSERT_TRUE(solver.solve(obj_sets, img_sets, guess, result, opts, PnpFlag::NONE));

  // Manual reprojection-only RMS with the recovered model / poses.
  const CameraModel64 rec64 = toCameraModel64(result.camera_model);
  double sq_sum = 0.0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < obj_sets.size(); ++i)
  {
    Eigen::Matrix3d R;
    {
      const double angle = result.rvecs[i].norm();
      R = (angle > 0.0)
            ? Eigen::AngleAxisd(angle, result.rvecs[i] / angle).toRotationMatrix()
            : Eigen::Matrix3d::Identity();
    }
    for (std::size_t j = 0; j < obj_sets[i].size(); ++j)
    {
      const Eigen::Vector3d p_cam = R * obj_sets[i][j] + result.tvecs[i];
      const PixelResult64 pr = rayToPixel64(rec64, p_cam);
      ASSERT_EQ(pr.status, StatusCode::OK);
      const double du = pr.pixel.u - img_sets[i][j].x();
      const double dv = pr.pixel.v - img_sets[i][j].y();
      sq_sum += du * du + dv * dv;
      ++n;
    }
  }
  const double manual_rms = std::sqrt(sq_sum / static_cast<double>(n));

  // rmse is computed in double before the float write-back the manual
  // recompute sees, so allow a small tolerance — far below the ~3 px jump a
  // prior-contaminated rmse would show.
  EXPECT_NEAR(result.rmse, manual_rms, 0.05);
  EXPECT_LT(result.rmse, 1.0);
}

// ===========================================================================
// 4. Invalid prior inputs are rejected atomically.
// ===========================================================================
TEST(CalibPriorTest, PriorInvalidInputRejected)
{
  const std::vector<CalibrationView> views = makeNarrowViews(0.0, 5U);
  ASSERT_FALSE(views.empty());
  const CameraModel seed = makeNarrowSeed();

  auto expectInvalid = [&](double ref_x, double ref_y, double sigma) {
    CalibrationOptions opts = makeNarrowOptions();
    opts.apply_principal_point_prior = true;
    opts.principal_point_prior_reference_x = ref_x;
    opts.principal_point_prior_reference_y = ref_y;
    opts.principal_point_prior_sigma_px = sigma;
    const CalibrationResult r = calibrate(views, seed, kNarrowLocks, opts);
    EXPECT_EQ(r.status, StatusCode::INVALID_INPUT);
  };

  expectInvalid(kNarrowW / 2.0, kNarrowH / 2.0, 0.0);    // sigma == 0
  expectInvalid(kNarrowW / 2.0, kNarrowH / 2.0, -1.0);   // sigma < 0
  expectInvalid(
    kNarrowW / 2.0, kNarrowH / 2.0, std::numeric_limits<double>::quiet_NaN()
  );                                                     // sigma NaN
  expectInvalid(0.0, kNarrowH / 2.0, 10.0);              // ref_x <= 0
  expectInvalid(
    std::numeric_limits<double>::infinity(), kNarrowH / 2.0, 10.0
  );                                                     // ref_x inf
}

// ===========================================================================
// 5. Uncertainty / observability reflect the prior: std(cx/cy) is capped by
//    the prior information and cx/cy never appear underdetermined.
// ===========================================================================
TEST(CalibPriorTest, UncertaintyReflectsPrior)
{
  const std::vector<CalibrationView> views = makeNarrowViews(0.3, 42U);
  ASSERT_FALSE(views.empty());

  const CalibrationResult with =
    calibrate(views, makeNarrowSeed(), kNarrowLocks, withPrior(makeNarrowOptions(), kPriorSigmaPx));
  ASSERT_EQ(with.status, StatusCode::OK);
  ASSERT_TRUE(with.uncertainty_available);

  // Cov = sigma_hat^2 (JtJ + P)^-1 with P >= 1/sigma_prior^2 on cx/cy: even if
  // the data carried zero information, std(cx) <= sigma_hat * sigma_prior, and
  // sigma_hat (pixel noise, ~0.3 px) keeps that well under sigma_prior itself.
  EXPECT_LE(stdOf(with, "cx"), kPriorSigmaPx);
  EXPECT_LE(stdOf(with, "cy"), kPriorSigmaPx);

  const auto &under = with.underdetermined_parameters;
  EXPECT_TRUE(std::find(under.begin(), under.end(), "cx") == under.end());
  EXPECT_TRUE(std::find(under.begin(), under.end(), "cy") == under.end());
}

// ===========================================================================
// 6. FIX_PRINCIPAL_POINTS wins: the prior is ignored on a fixed block.
// ===========================================================================
TEST(CalibPriorTest, PriorIgnoredWhenPrincipalPointFixed)
{
  const std::vector<CalibrationView> views = makeNarrowViews(0.2, 3U);
  ASSERT_FALSE(views.empty());
  const CameraModel seed = makeNarrowSeed();

  CalibrationOptions opts = makeNarrowOptions();
  // Far-off reference with a tight sigma: if the prior were (incorrectly)
  // applied to a fixed block, Ceres would fail to move it and the solve
  // bookkeeping would still show the pull; with a correctly skipped prior the
  // principal point simply stays at the seed.
  opts = withPrior(opts, 1.0);
  opts.principal_point_prior_reference_x = kNarrowW / 2.0 + 80.0;

  const CalibrationResult r =
    calibrate(views, seed, kNarrowLocks | PnpFlag::FIX_PRINCIPAL_POINTS, opts);
  ASSERT_TRUE(r.status == StatusCode::OK || r.status == StatusCode::NON_CONVERGED);
  EXPECT_FLOAT_EQ(r.camera_model.intrinsics.cx, seed.intrinsics.cx);
  EXPECT_FLOAT_EQ(r.camera_model.intrinsics.cy, seed.intrinsics.cy);
}

}  // namespace
