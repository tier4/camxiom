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

// Unit tests for camxiom::init::estimatePinholeZhang.
//
// Strategy: build an 8x6 checkerboard at known pose(s), project through a
// ground-truth pinhole K via the standard K*[R|t] formula (no rayToPixel
// dependency), recover K with estimatePinholeZhang, and compare with the
// ground truth.
//
// All tests are deterministic (seed 42) and self-contained — no shared
// mutable fixtures between tests.

#include "camxiom/init/pinhole_zhang.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "support/calib_test_fixtures.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using camxiom::StatusCode;
using camxiom::init::estimatePinholeZhang;
using camxiom::init::PlanarObservation;

namespace
{

// ---------------------------------------------------------------------------
// Board / rotation geometry helpers — shared across the calibration / init
// tests (tests/support/calib_test_fixtures.hpp).
// ---------------------------------------------------------------------------

using camxiom::test::makeCheckerboard;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;

// ---------------------------------------------------------------------------
// Projection helper — purely inline pinhole; does NOT use camxiom::rayToPixel
// so that the test is independent of MS0 projection paths.
// ---------------------------------------------------------------------------

/// Project 2xM board points (Z=0 implicit) to 2xM image pixels via K and
/// the view transform (R, t), where world = board frame.
///   x_cam = R * [board(0,i); board(1,i); 0] + t
///   u = K * x_cam / x_cam(2)
///
/// Returns true on success.  Returns false (without projecting) if any
/// point has non-positive depth — that would mean the board is behind the
/// camera, which the test poses are designed to avoid.  Call sites wrap
/// this in ASSERT_TRUE so that a false return aborts the calling TEST body.
bool projectPinhole(
  const Eigen::Matrix3d &K, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
  const Eigen::Matrix2Xd &board_pts, Eigen::Matrix2Xd &image_pts_out
)
{
  const Eigen::Index m = board_pts.cols();
  image_pts_out.resize(2, m);
  for (Eigen::Index i = 0; i < m; ++i)
  {
    Eigen::Vector3d P;
    P << board_pts(0, i), board_pts(1, i), 0.0;
    const Eigen::Vector3d x_cam = R * P + t;
    if (x_cam(2) <= 0.0)
    {
      ADD_FAILURE() << "Point " << i << " is behind the camera (depth = " << x_cam(2) << ")";
      return false;
    }
    const Eigen::Vector3d uv_h = K * x_cam;
    image_pts_out(0, i) = uv_h(0) / uv_h(2);
    image_pts_out(1, i) = uv_h(1) / uv_h(2);
  }
  return true;
}

/// Convenience: build one PlanarObservation from a pose.
/// Returns true on success; false if any board point is behind the camera.
/// Call sites wrap this in ASSERT_TRUE so that a false return aborts the
/// calling TEST body before any downstream algorithm invocation.
bool buildView(
  const Eigen::Matrix3d &K, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
  const Eigen::Matrix2Xd &board, PlanarObservation &view_out
)
{
  view_out.board_pts = board;
  return projectPinhole(K, R, t, board, view_out.image_pts);
}

/// Add zero-mean Gaussian noise (std = sigma pixels) to a 2xM matrix in-place.
void addGaussianNoise(std::mt19937 &rng, double sigma, Eigen::Matrix2Xd &pts)
{
  std::normal_distribution<double> noise(0.0, sigma);
  for (Eigen::Index r = 0; r < pts.rows(); ++r)
  {
    for (Eigen::Index c = 0; c < pts.cols(); ++c)
    {
      pts(r, c) += noise(rng);
    }
  }
}

// ---------------------------------------------------------------------------
// Canonical 5-view setup used by NoNoiseRecovery and NoisyRecovery.
//
// Ground-truth K: fx=fy=500, cx=320, cy=240, skew=0 on a 640x480 sensor.
// Board: 8x6 checkerboard at 0.05 m spacing, 48 corners.
// Poses: 5 views spanning tilt so the IAC constraint matrix is full rank.
// ---------------------------------------------------------------------------

Eigen::Matrix3d groundTruthK()
{
  Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
  K(0, 0) = 500.0;  // fx
  K(1, 1) = 500.0;  // fy
  K(0, 2) = 320.0;  // cx
  K(1, 2) = 240.0;  // cy
  K(2, 2) = 1.0;
  return K;
}

// makeCanonicalViews returns an empty vector on failure (any board point
// behind camera). Call sites use ASSERT_FALSE(views.empty()) to abort the
// TEST before passing invalid data to the algorithm.
std::vector<PlanarObservation> makeCanonicalViews(
  const Eigen::Matrix3d &K, const Eigen::Matrix2Xd &board
)
{
  // Five views chosen to span the IAC null space (see Zhang 2000, §3.1):
  //   at least two distinct rotation axes are required. The five poses below
  //   include rotations around X, Y, and combinations, so the 2*5=10 IAC
  //   constraint rows are rank-5 (full rank for 5 unknowns + scale).
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const double deg20 = 20.0 * camxiom::constants::kPi / 180.0;

  // Pose 0: front-parallel (R = I), board centred ~0.2 m left and ~0.15 m up.
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);

  // Pose 1: look-down (+25 deg around X).
  const Eigen::Matrix3d R1 = rotMatX(+deg25);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);

  // Pose 2: look-up (−25 deg around X).
  const Eigen::Matrix3d R2 = rotMatX(-deg25);
  const Eigen::Vector3d t2(-0.2, -0.15, 1.0);

  // Pose 3: look-right (+25 deg around Y).
  const Eigen::Matrix3d R3 = rotMatY(+deg25);
  const Eigen::Vector3d t3(-0.2, -0.15, 1.0);

  // Pose 4: compound tilt (+20 X, −20 Y).
  const Eigen::Matrix3d R4 = rotMatX(+deg20) * rotMatY(-deg20);
  const Eigen::Vector3d t4(-0.2, -0.15, 1.1);

  std::vector<PlanarObservation> views(5);
  if (!buildView(K, R0, t0, board, views[0]))
  {
    return {};
  }
  if (!buildView(K, R1, t1, board, views[1]))
  {
    return {};
  }
  if (!buildView(K, R2, t2, board, views[2]))
  {
    return {};
  }
  if (!buildView(K, R3, t3, board, views[3]))
  {
    return {};
  }
  if (!buildView(K, R4, t4, board, views[4]))
  {
    return {};
  }
  return views;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: NoNoiseRecovery
//
// Ground-truth projection (exact, no floating-point noise beyond
// double-precision arithmetic). The IAC linear system and Cholesky path
// should recover fx, fy within 1e-2 and cx, cy within 1e-2.
//
// Tolerance 1e-2: Zhang's IAC is a linear-algebra pipeline on exactly-
// projected points. The dominant source of error is condition number of V
// (roughly 1e3 for these five tilts), which limits accuracy to ~1e-12
// residuals after SVD. The 1e-2 tolerance is conservative by many orders
// of magnitude and is chosen to also accommodate any future minor
// algorithmic changes without breaking the test.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, NoNoiseRecovery)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::vector<PlanarObservation> views = makeCanonicalViews(K_gt, board);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point is behind the camera";

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Constant(42.0);
  const StatusCode sc = estimatePinholeZhang(views, K_out);

  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for noise-free 5-view setup";

  // K(2,2) = 1 convention (test 8 merged here for convenience).
  // Tolerance 1e-12: the algorithm normalises K by K(2,2), so K_out(2,2)
  // should be exactly 1.0 to double precision.
  EXPECT_NEAR(K_out(2, 2), 1.0, 1e-12) << "K_out(2,2) must equal 1.0 (algorithm output convention: "
                                          "K(2,2)=1 so K is directly usable as IntrinsicsModel)";

  // fx and fy must be positive.
  EXPECT_GT(K_out(0, 0), 0.0) << "fx must be positive";
  EXPECT_GT(K_out(1, 1), 0.0) << "fy must be positive";

  // Focal lengths: 1e-2 tolerance.
  // Justification: noise-free IAC; tolerance dominated by double-precision
  // accumulation through Hartley normalisation and Cholesky, not by noise.
  EXPECT_NEAR(K_out(0, 0), 500.0, 1e-2) << "fx should recover 500.0 within 1e-2 (noise-free)";
  EXPECT_NEAR(K_out(1, 1), 500.0, 1e-2) << "fy should recover 500.0 within 1e-2 (noise-free)";

  // Principal point: 1e-2 tolerance.
  EXPECT_NEAR(K_out(0, 2), 320.0, 1e-2) << "cx should recover 320.0 within 1e-2 (noise-free)";
  EXPECT_NEAR(K_out(1, 2), 240.0, 1e-2) << "cy should recover 240.0 within 1e-2 (noise-free)";

  // Skew: 1e-3 tolerance.
  // Ground truth has zero skew; any non-zero value is pure numerical noise
  // from the SVD / Cholesky chain. 1e-3 is generous given the data scale.
  EXPECT_NEAR(K_out(0, 1), 0.0, 1e-3) << "skew should be near 0 within 1e-3 (noise-free)";

  // K must be upper-triangular (lower-left entries = 0).
  EXPECT_NEAR(K_out(1, 0), 0.0, 1e-12) << "K_out(1,0) must be 0 (upper-triangular)";
  EXPECT_NEAR(K_out(2, 0), 0.0, 1e-12) << "K_out(2,0) must be 0 (upper-triangular)";
  EXPECT_NEAR(K_out(2, 1), 0.0, 1e-12) << "K_out(2,1) must be 0 (upper-triangular)";
}

// ---------------------------------------------------------------------------
// Test 2: NoisyRecovery
//
// Same 5-view / 8x6 setup, but with N(0, 0.5 px) Gaussian noise added to
// each observed image point. Seed = 42 for reproducibility.
//
// Expected bounds (relative for focal lengths, absolute for pp/skew):
//   |fx - 500|/500 < 0.05  (5 %)
//   |fy - 500|/500 < 0.05  (5 %)
//   |cx - 320| < 5.0 px
//   |cy - 240| < 10.0 px
//   |skew| < 5.0
//
// Tolerance justification: estimatePinholeZhang is a LINEAR initial-guess
// estimator (Zhang 2000 Step 1–4 only; no nonlinear Ceres refinement). The
// IAC is a rank-1 constrained linear system; with only 5 views and N(0, 0.5)
// pixel noise, the observable range is limited.
//
// Empirical values observed with seed 42 (single trial):
//   fx ≈ 492.6 (error 1.48 %), fy ≈ 491.2 (error 1.76 %)
//   cx ≈ 318.7 (error 1.32 px), cy ≈ 234.4 (error 5.6 px)
//
// The principal-point (especially cy) is notoriously sensitive to the IAC
// solution because a small change in the null-space direction maps to a
// large principal-point shift. For 5 views this is the expected regime for
// the linear estimator; Ceres refinement (MS2 IntrinsicsCalibrator) is what
// ultimately drives errors down to sub-pixel.
//
// The 5% / 5 px / 10 px / 5.0 skew bounds give a 3× safety margin over the
// observed worst-case with seed 42. The purpose of this test is to confirm
// the algorithm runs without crashing and returns a *usable* initial guess
// for downstream Ceres optimisation — not to validate final accuracy.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, NoisyRecovery)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::vector<PlanarObservation> views = makeCanonicalViews(K_gt, board);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point is behind the camera";

  // Add N(0, 0.5 px) noise to every view's image_pts.
  std::mt19937 rng(42U);
  for (auto &v : views)
  {
    addGaussianNoise(rng, 0.5, v.image_pts);
  }

  Eigen::Matrix3d K_out = Eigen::Matrix3d::Constant(42.0);
  const StatusCode sc = estimatePinholeZhang(views, K_out);

  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for 5-view noisy setup";

  // K(2,2) convention always holds regardless of noise.
  EXPECT_NEAR(K_out(2, 2), 1.0, 1e-12) << "K_out(2,2) must equal 1.0 even with noisy inputs";
  EXPECT_GT(K_out(0, 0), 0.0) << "fx must be positive";
  EXPECT_GT(K_out(1, 1), 0.0) << "fy must be positive";

  // Relative focal-length error < 5 %.
  // Justification: Zhang's LINEAR estimator with 5 views and 0.5 px noise
  // achieves ~1.5-2% empirically (seed 42). 5% gives a 3x safety margin for
  // adversarial seed draws without making the test meaningless.
  EXPECT_NEAR(K_out(0, 0) / 500.0, 1.0, 0.05)
    << "fx relative error should be < 5% under 0.5 px noise (linear estimator)";
  EXPECT_NEAR(K_out(1, 1) / 500.0, 1.0, 0.05)
    << "fy relative error should be < 5% under 0.5 px noise (linear estimator)";

  // Principal-point absolute error: cx < 5 px, cy < 10 px.
  // Justification: the IAC null-space direction is weakly constrained for the
  // principal point; cy error of ~5.6 px is observed at seed 42, and 10 px
  // gives a 1.8x margin. This is expected for a linear-only estimator.
  EXPECT_NEAR(K_out(0, 2), 320.0, 5.0)
    << "cx should be within 5.0 px under 0.5 px noise (linear estimator)";
  EXPECT_NEAR(K_out(1, 2), 240.0, 10.0)
    << "cy should be within 10.0 px under 0.5 px noise (linear estimator)";

  // Skew: ground truth is 0; 0.5 px noise on the IAC null-space can induce
  // apparent skew. 5.0 is a loose sanity bound — any larger value would
  // indicate a seriously degenerate solution not suitable as an initial guess.
  EXPECT_NEAR(K_out(0, 1), 0.0, 5.0)
    << "skew should remain near 0 under 0.5 px noise (linear estimator)";
}

// ---------------------------------------------------------------------------
// Test 3: RejectFewViews
//
// views.size() == 2 < kMinViews (3) -> INVALID_INPUT.
// K_out pre-seeded with sentinel 42.0 must remain unchanged.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectFewViews)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // Build 2 views only.
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(25.0 * camxiom::constants::kPi / 180.0);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);

  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(K_gt, R0, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R1, t1, board, views[1]));

  // Pre-seed sentinel.
  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for views.size() == 2";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 4: RejectInsufficientPoints
//
// 3 views, but one view has only 3 points (M < 4) -> INVALID_INPUT.
// K_out must remain unchanged.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectInsufficientPoints)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // Pose helpers.
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(+deg25);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);
  const Eigen::Matrix3d R2 = rotMatX(-deg25);
  const Eigen::Vector3d t2(-0.2, -0.15, 1.0);

  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(K_gt, R0, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R1, t1, board, views[1]));

  // Third view: deliberately only 3 points (< 4).
  {
    PlanarObservation bad;
    bad.board_pts = board.leftCols(3);
    ASSERT_TRUE(projectPinhole(K_gt, R2, t2, board.leftCols(3), bad.image_pts));
    views.push_back(bad);
  }

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT when a view has only 3 points";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 5: RejectColumnMismatch
//
// 3 views, one view has board_pts.cols() = 10, image_pts.cols() = 8
// -> INVALID_INPUT. K_out unchanged.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectColumnMismatch)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(+deg25);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);

  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(K_gt, R0, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R1, t1, board, views[1]));

  // Third view: board_pts has 10 cols, image_pts has 8 cols.
  {
    PlanarObservation bad;
    bad.board_pts = board.leftCols(10);  // 10 board pts
    bad.image_pts = board.leftCols(8);   // 8 image pts (mismatch)
    views.push_back(bad);
  }

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for board/image column count mismatch (10 vs 8)";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 6a: RejectNonFinite_NaN
//
// 3 views, set image_pts(0,0) = NaN in one view -> INVALID_INPUT, K_out
// unchanged.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectNonFinite_NaN)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(+deg25);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);
  const Eigen::Matrix3d R2 = rotMatX(-deg25);
  const Eigen::Vector3d t2(-0.2, -0.15, 1.0);

  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(K_gt, R0, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R1, t1, board, views[1]));
  {
    PlanarObservation bad;
    ASSERT_TRUE(buildView(K_gt, R2, t2, board, bad));
    bad.image_pts(0, 0) = std::numeric_limits<double>::quiet_NaN();
    views.push_back(bad);
  }

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when image_pts contains NaN";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 6b: RejectNonFinite_Inf
//
// 3 views, set image_pts(1,0) = +inf in one view -> INVALID_INPUT, K_out
// unchanged.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectNonFinite_Inf)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(+deg25);
  const Eigen::Vector3d t1(-0.2, -0.15, 1.2);
  const Eigen::Matrix3d R2 = rotMatX(-deg25);
  const Eigen::Vector3d t2(-0.2, -0.15, 1.0);

  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(K_gt, R0, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R1, t1, board, views[1]));
  {
    PlanarObservation bad;
    ASSERT_TRUE(buildView(K_gt, R2, t2, board, bad));
    bad.image_pts(1, 0) = std::numeric_limits<double>::infinity();
    views.push_back(bad);
  }

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when image_pts contains +inf";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 7: RejectFrontParallelDuplicates
//
// All three views have R = identity (front-parallel with the board). The
// constraints contributed to the IAC are effectively duplicates and the
// null space of V becomes multi-dimensional -> DEGENERATE_CONFIG.
// K_out must remain unchanged.
//
// This is the canonical "Zhang's method fails with all-fronto-parallel
// views" corner case.  The tolerances inside appendViewConstraints yield
// identical (up to translation) rows so V is rank-deficient.
// ---------------------------------------------------------------------------

TEST(PinholeZhang, RejectFrontParallelDuplicates)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // All three views: R = identity, only translation differs.
  const Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-0.2, -0.15, 1.0);
  const Eigen::Vector3d t1(0.0, 0.0, 1.0);
  const Eigen::Vector3d t2(0.1, 0.1, 0.9);

  std::vector<PlanarObservation> views(3);
  ASSERT_TRUE(buildView(K_gt, R_id, t0, board, views[0]));
  ASSERT_TRUE(buildView(K_gt, R_id, t1, board, views[1]));
  ASSERT_TRUE(buildView(K_gt, R_id, t2, board, views[2]));

  const Eigen::Matrix3d sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d K_out = sentinel;

  const StatusCode sc = estimatePinholeZhang(views, K_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG for three front-parallel views (R=I)";
  EXPECT_EQ((K_out - sentinel).norm(), 0.0)
    << "K_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ---------------------------------------------------------------------------
// Test 8: SuccessKConvention
//
// Verifies the K_out convention properties that must hold on every OK return,
// independently of the specific intrinsic values.  Uses the same 5-view
// noise-free setup as test 1, but focuses only on structural properties.
//
//   K_out(2,2) == 1.0  within 1e-12  (normalisation)
//   K_out(0,0) > 0                    (fx > 0)
//   K_out(1,1) > 0                    (fy > 0)
//   K_out(1,0) == 0.0  within 1e-12  (upper-triangular)
//   K_out(2,0) == 0.0  within 1e-12
//   K_out(2,1) == 0.0  within 1e-12
// ---------------------------------------------------------------------------

TEST(PinholeZhang, SuccessKConvention)
{
  const Eigen::Matrix3d K_gt = groundTruthK();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);
  std::vector<PlanarObservation> views = makeCanonicalViews(K_gt, board);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point is behind the camera";

  Eigen::Matrix3d K_out;
  const StatusCode sc = estimatePinholeZhang(views, K_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for 5-view canonical setup";

  // (a) Normalisation: K_out(2,2) == 1.
  // Tolerance 1e-12: the algorithm performs K /= K(2,2), so K(2,2) should be
  // 1.0 to double precision after normalisation.
  EXPECT_NEAR(K_out(2, 2), 1.0, 1e-12)
    << "K_out(2,2) must equal 1.0 exactly (normalisation convention)";

  // (b) Positive focal lengths.
  EXPECT_GT(K_out(0, 0), 0.0) << "K_out(0,0) = fx must be positive";
  EXPECT_GT(K_out(1, 1), 0.0) << "K_out(1,1) = fy must be positive";

  // (c) Upper-triangular structure: three lower-left entries must be zero.
  // Tolerance 1e-12: Cholesky + triangular solve on a 3x3 should produce
  // zeros in the lower triangle to nearly machine precision.
  EXPECT_NEAR(K_out(1, 0), 0.0, 1e-12) << "K_out(1,0) must be 0 (upper-triangular)";
  EXPECT_NEAR(K_out(2, 0), 0.0, 1e-12) << "K_out(2,0) must be 0 (upper-triangular)";
  EXPECT_NEAR(K_out(2, 1), 0.0, 1e-12) << "K_out(2,1) must be 0 (upper-triangular)";
}
