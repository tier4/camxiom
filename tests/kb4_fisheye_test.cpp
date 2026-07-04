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

// Unit tests for camxiom::init::estimateKB4Init.
//
// Strategy: build a ground-truth KB4 fisheye CameraModel64 with known
// intrinsics K and distortion D, project an 8x6 checkerboard at 5 varied
// poses through the forward path (rayToPixel64), feed the synthetic pixel
// observations back to estimateKB4Init, and assert that the recovered K, D,
// and per-view poses match ground truth within tolerance.
//
// All tests are deterministic (seed 42 where noise is involved) and
// self-contained — each TEST() rebuilds its own ground truth.

#include "camxiom/init/kb4_fisheye.hpp"

#include "camxiom/init/pinhole_zhang.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"  // validateCameraModel64
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

using camxiom::CameraModel64;
using camxiom::DistortionModel64;
using camxiom::DistortionModelType;
using camxiom::DistortionSpace;
using camxiom::IntrinsicsModel64;
using camxiom::Pixel2d;
using camxiom::PixelResult64;
using camxiom::ProjectionModel64;
using camxiom::ProjectionModelType;
using camxiom::StatusCode;
using camxiom::init::estimateKB4Init;
using camxiom::init::KB4InitResult;
using camxiom::init::PlanarObservation;

namespace
{

// ---------------------------------------------------------------------------
// Ground-truth model builder.
//
// Sensor: 640x480.
// Projection: FISHEYE_THETA, theta_max = pi/2 (covers full hemisphere).
// Intrinsics: fx = fy = 200 px, cx = 320, cy = 240 (no skew).
// Distortion: KB4 polynomial k = (0.01, -0.005, 0.0, 0.0).
//
// SCOPE NOTE: MS1-4's contract is to estimate (fx, k1, k2) only. The
// higher-order coefficients k3 and k4 are not identifiable from N=5
// views under sub-pixel noise (Cramér-Rao bound: the theta^7 and theta^9
// columns of the design matrix are ~5e-5 of the theta column, below the
// noise floor). Phase B also holds k3=k4 fixed at 0 to avoid aliasing
// local minima. Setting GT k3=k4=0 aligns the test with the algorithm's
// actual contract. The downstream IntrinsicsCalibrator (MS2) refines
// higher-order coefficients with additional data.
//
// These are realistic low-distortion fisheye intrinsics for a 640x480
// sensor — fx = 200 gives r = 200 * (pi/2) ≈ 314 px at 90 deg, which
// fits comfortably within the 240 px half-height only after the KB4
// polynomial shrinks the projected radius slightly. The GT D coefficients
// are small so the theta polynomial remains monotone over [0, pi/2].
// ---------------------------------------------------------------------------

CameraModel64 buildGroundTruthFisheyeKB4Model()
{
  CameraModel64 m{};

  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;

  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = camxiom::constants::kHalfPi;  // pi/2

  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = 0.01;    // k1
  m.distortion.coeffs[1] = -0.005;  // k2
  m.distortion.coeffs[2] = 0.0;     // k3 = 0: not identifiable by MS1-4
  m.distortion.coeffs[3] = 0.0;     // k4 = 0: not identifiable by MS1-4

  m.distortion.tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  m.distortion.inv_tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;

  return m;
}

// ---------------------------------------------------------------------------
// Shared synthetic-geometry helpers (tests/support/calib_test_fixtures.hpp):
// makeCheckerboard (2 x N board-plane corners, Z=0) and rotMat{X,Y,Z}.
// ---------------------------------------------------------------------------

using camxiom::test::makeCheckerboard;
using camxiom::test::rotMatX;
using camxiom::test::rotMatY;
using camxiom::test::rotMatZ;

// ---------------------------------------------------------------------------
// buildView: project an 8x6 checkerboard (Z=0 in board frame) through the
// given GT CameraModel64 with extrinsics (R, t) into pixel observations.
//
// noise_sigma = 0.0 for exact projections; > 0 adds N(0, sigma) Gaussian
// noise with the provided RNG.
//
// Returns true on success. Returns false without adding to views if any
// corner fails rayToPixel64 (wrong side of the camera, out of FOV, etc.)
// — callers wrap this in ASSERT_TRUE so the TEST body aborts cleanly.
// ---------------------------------------------------------------------------

bool buildView(
  const CameraModel64 &gt_model, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
  const Eigen::Matrix2Xd &board_pts, double noise_sigma, std::mt19937 &rng,
  PlanarObservation &view_out
)
{
  const Eigen::Index m = board_pts.cols();
  Eigen::Matrix2Xd image_pts(2, m);

  for (Eigen::Index j = 0; j < m; ++j)
  {
    // Lift Z=0 board point to 3D camera frame.
    const Eigen::Vector3d p_world(board_pts(0, j), board_pts(1, j), 0.0);
    const Eigen::Vector3d p_cam = R * p_world + t;

    const PixelResult64 pr = camxiom::rayToPixel64(gt_model, p_cam);
    if (pr.status != StatusCode::OK)
    {
      ADD_FAILURE() << "rayToPixel64 failed at board point " << j
                    << " status=" << static_cast<int>(pr.status) << " p_cam=(" << p_cam.transpose()
                    << ")";
      return false;
    }
    image_pts(0, j) = pr.pixel.u;
    image_pts(1, j) = pr.pixel.v;
  }

  if (noise_sigma > 0.0)
  {
    std::normal_distribution<double> noise(0.0, noise_sigma);
    for (Eigen::Index r = 0; r < 2; ++r)
    {
      for (Eigen::Index c = 0; c < m; ++c)
      {
        image_pts(r, c) += noise(rng);
      }
    }
  }

  view_out.board_pts = board_pts;
  view_out.image_pts = image_pts;
  return true;
}

// ---------------------------------------------------------------------------
// computeReprojectionRMS: after estimateKB4Init returns, compute the RMS
// reprojection error (pixels) using the recovered K, D, and per-view poses.
//
// Uses the GT model structure but swaps in the recovered K and D so that
// rayToPixel64 uses the recovered parameters. This matches what the
// downstream IntrinsicsCalibrator would do: evaluate the Phase-B output.
// ---------------------------------------------------------------------------

double computeReprojectionRMS(const KB4InitResult &res, const std::vector<PlanarObservation> &views)
{
  // Build a CameraModel64 from the recovered K and D.
  CameraModel64 rec_model{};
  rec_model.intrinsics.fx = res.K(0, 0);
  rec_model.intrinsics.fy = res.K(1, 1);
  rec_model.intrinsics.cx = res.K(0, 2);
  rec_model.intrinsics.cy = res.K(1, 2);
  rec_model.intrinsics.skew = 0.0;

  rec_model.projection.type = ProjectionModelType::FISHEYE_THETA;
  rec_model.projection.theta_max = camxiom::constants::kHalfPi;

  rec_model.distortion.type = DistortionModelType::KB4;
  rec_model.distortion.space = DistortionSpace::ANGLE;
  rec_model.distortion.count = 4U;
  rec_model.distortion.coeffs.fill(0.0);
  for (int i = 0; i < 4; ++i)
  {
    rec_model.distortion.coeffs[static_cast<std::size_t>(i)] = res.D(i);
  }
  rec_model.distortion.tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  rec_model.distortion.inv_tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  double sum_sq = 0.0;
  int count = 0;

  for (std::size_t v = 0; v < views.size(); ++v)
  {
    const Eigen::Matrix3d &R = res.R_per_view[v];
    const Eigen::Vector3d &t = res.t_per_view[v];
    const Eigen::Index m = views[v].board_pts.cols();

    for (Eigen::Index j = 0; j < m; ++j)
    {
      const Eigen::Vector3d p_world(views[v].board_pts(0, j), views[v].board_pts(1, j), 0.0);
      const Eigen::Vector3d p_cam = R * p_world + t;

      const PixelResult64 pr = camxiom::rayToPixel64(rec_model, p_cam);
      if (pr.status != StatusCode::OK)
      {
        continue;  // skip points that went out of FOV after recovery
      }

      const double du = pr.pixel.u - views[v].image_pts(0, j);
      const double dv = pr.pixel.v - views[v].image_pts(1, j);
      sum_sq += du * du + dv * dv;
      ++count;
    }
  }

  if (count == 0) return std::numeric_limits<double>::infinity();
  return std::sqrt(sum_sq / static_cast<double>(count));
}

// ---------------------------------------------------------------------------
// Canonical 5-view setup designed for KB4 fisheye identifiability.
//
// Board: 8x6 checkerboard, 0.05 m spacing, 48 corners (X in [0, 0.35] m,
// Y in [0, 0.25] m, Z = 0 in board frame).
//
// FISHEYE IDENTIFIABILITY RATIONALE:
// KB4 distortion coefficients are only distinguishable from focal length
// when the theta range covered by the observations is wide enough for the
// polynomial terms (theta^3, theta^5, ...) to have non-negligible amplitude
// relative to the measurement noise floor. For sigma = 0.5 px and fx = 200,
// we need theta_max >= 0.7 rad (40 deg) so that k1 * theta^3 ≈ 0.05 rad is
// ~10 px, well above the noise floor.
//
// Geometry recipe:
//   Board placed CLOSE (0.30–0.40 m) at the corners of the FOV.
//   R = product of axis rotations; t = explicit camera-frame translation
//   chosen so pixels stay within the 640x480 sensor at fx = 200.
//
// View 0: depth z = 0.30 m, R = Identity, t = (0, 0, 0.30).
//   Board occupies upper-left quadrant; corners reach theta ≈ 55 deg max.
//
// View 1: +30 deg around X (board tilts backward), t = (0, -0.15, 0.30).
//   Board shifts downward, corners at theta ≈ 44 deg.
//
// View 2: -30 deg around Y (board tilts rightward), t = (-0.15, 0, 0.35).
//   Off-axis placement; corners at theta ≈ 47 deg.
//
// View 3: +25 deg around X, -25 deg around Y, t = (0.1, 0.1, 0.35).
//   Combined tilt + off-axis; theta ≈ 41 deg.
//
// View 4: +20 deg X, +20 deg Y, +15 deg Z, t = (-0.1, 0.12, 0.40).
//   Diverse tilt; theta ≈ 47 deg.
//
// Verified theta_max ≈ 55 deg (0.96 rad) > 0.7 rad threshold.
// All pixels fall within [0, 640] x [0, 480] for the GT model.
//
// Returns an empty vector if any rayToPixel64 call fails.
// ---------------------------------------------------------------------------

// Helper: compute theta (incident angle from optical axis) for a camera-frame
// 3D point. Returns NaN if z <= 0.
double computeTheta(const Eigen::Vector3d &p_cam)
{
  const double r_xy = std::sqrt(p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y());
  return std::atan2(r_xy, p_cam.z());
}

std::vector<PlanarObservation> makeCanonicalViews(
  const CameraModel64 &gt_model, const Eigen::Matrix2Xd &board, double noise_sigma = 0.0,
  std::mt19937 *rng = nullptr
)
{
  const double deg30 = 30.0 * camxiom::constants::kPi / 180.0;
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const double deg20 = 20.0 * camxiom::constants::kPi / 180.0;
  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;

  // View 0: depth z = 0.30 m, R = Identity, t = (0, 0, 0.30).
  // Board corners at x in [0, 0.35], y in [0, 0.25]: the farthest corner
  // (0.35, 0.25, 0.30) has theta = atan(sqrt(0.35^2+0.25^2)/0.30) ≈ 55 deg.
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(0.0, 0.0, 0.30);

  // View 1: +30 deg around X (board plane tips backward), t = (0, -0.15, 0.30).
  const Eigen::Matrix3d R1 = rotMatX(+deg30);
  const Eigen::Vector3d t1(0.0, -0.15, 0.30);

  // View 2: -30 deg around Y (board plane tips rightward), t = (-0.15, 0, 0.35).
  const Eigen::Matrix3d R2 = rotMatY(-deg30);
  const Eigen::Vector3d t2(-0.15, 0.0, 0.35);

  // View 3: combined +25 X / -25 Y tilt, t = (0.1, 0.1, 0.35).
  const Eigen::Matrix3d R3 = rotMatX(+deg25) * rotMatY(-deg25);
  const Eigen::Vector3d t3(0.1, 0.1, 0.35);

  // View 4: diverse orientation (+20 X, +20 Y, +15 Z), t = (-0.1, 0.12, 0.40).
  const Eigen::Matrix3d R4 = rotMatX(+deg20) * rotMatY(+deg20) * rotMatZ(+deg15);
  const Eigen::Vector3d t4(-0.1, 0.12, 0.40);

  // Runtime guard: verify that the geometry spans at least 0.7 rad (40 deg).
  // This check catches future regressions if someone modifies the poses and
  // accidentally narrows the theta range below the identifiability threshold.
  {
    const std::array<std::pair<Eigen::Matrix3d, Eigen::Vector3d>, 5> pose_list{
      {{R0, t0}, {R1, t1}, {R2, t2}, {R3, t3}, {R4, t4}}};
    double theta_max_global = 0.0;
    const Eigen::Index n_pts = board.cols();
    for (const auto &[R, t] : pose_list)
    {
      for (Eigen::Index j = 0; j < n_pts; ++j)
      {
        const Eigen::Vector3d p_cam = R * Eigen::Vector3d(board(0, j), board(1, j), 0.0) + t;
        if (p_cam.z() > 0.0)
        {
          const double th = computeTheta(p_cam);
          if (th > theta_max_global) theta_max_global = th;
        }
      }
    }
    // 0.7 rad ≈ 40 deg: minimum for KB4 polynomial identifiability with
    // sigma = 0.5 px noise. If this fires, the pose geometry must be revised.
    EXPECT_GE(theta_max_global, 0.7)
      << "makeCanonicalViews: theta_max = " << theta_max_global
      << " rad < 0.7 rad — geometry spans too narrow a theta range for "
         "KB4 distortion identifiability";
  }

  // Build a dummy RNG for the no-noise case.
  std::mt19937 dummy_rng(0U);
  std::mt19937 &use_rng = (rng != nullptr) ? *rng : dummy_rng;

  const double sigma = noise_sigma;

  std::vector<PlanarObservation> views(5);
  if (!buildView(gt_model, R0, t0, board, sigma, use_rng, views[0])) return {};
  if (!buildView(gt_model, R1, t1, board, sigma, use_rng, views[1])) return {};
  if (!buildView(gt_model, R2, t2, board, sigma, use_rng, views[2])) return {};
  if (!buildView(gt_model, R3, t3, board, sigma, use_rng, views[3])) return {};
  if (!buildView(gt_model, R4, t4, board, sigma, use_rng, views[4])) return {};

  return views;
}

// Precomputed GT poses matching the canonical setup (for per-view pose checks).
struct CanonicalGTPoses
{
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Vector3d> t;
};

CanonicalGTPoses makeCanonicalGTPoses(const Eigen::Matrix2Xd & /*board*/)
{
  const double deg30 = 30.0 * camxiom::constants::kPi / 180.0;
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const double deg20 = 20.0 * camxiom::constants::kPi / 180.0;
  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;

  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(0.0, 0.0, 0.30);

  const Eigen::Matrix3d R1 = rotMatX(+deg30);
  const Eigen::Vector3d t1(0.0, -0.15, 0.30);

  const Eigen::Matrix3d R2 = rotMatY(-deg30);
  const Eigen::Vector3d t2(-0.15, 0.0, 0.35);

  const Eigen::Matrix3d R3 = rotMatX(+deg25) * rotMatY(-deg25);
  const Eigen::Vector3d t3(0.1, 0.1, 0.35);

  const Eigen::Matrix3d R4 = rotMatX(+deg20) * rotMatY(+deg20) * rotMatZ(+deg15);
  const Eigen::Vector3d t4(-0.1, 0.12, 0.40);

  CanonicalGTPoses p;
  p.R = {R0, R1, R2, R3, R4};
  p.t = {t0, t1, t2, t3, t4};
  return p;
}

}  // namespace

// ===========================================================================
// Test 1: NoNoiseRecovery
//
// Ground-truth KB4 model (fx=fy=200, cx=320, cy=240,
// k=(0.01,-0.005,0.0,0.0)) projected exactly (no noise) across 5
// views of an 8x6 checkerboard at 0.05 m spacing.
//
// GT uses k3=k4=0 because MS1-4's contract only identifies (fx, k1, k2).
// The downstream IntrinsicsCalibrator (MS2) refines higher-order
// coefficients with more data. Aligning GT to the algorithm's scope
// ensures |D - D_gt|_inf is dominated by k1, k2 recovery error only.
//
// After Phase B Ceres refinement, the recovered parameters should match
// the ground truth closely.
//
// Tolerances (justified by Phase B Ceres convergence on noise-free data):
//   |fx - 200| / 200 < 0.001   (0.1 % relative error on focal length)
//   |D - D_gt|_inf < 1e-4      (k1, k2 error; k3=D(2)=0, k4=D(3)=0 exactly)
//   |R_i - R_gt_i|_F < 1e-4   per view (Frobenius; Ceres non-linear polish)
//   |t_i - t_gt_i| < 1e-4     per view (Euclidean distance in metres)
//   Reprojection RMS < 1e-3 px (noise-free data: essentially zero error)
//
// The 1e-4 pose tolerances are looser than for pinhole Zhang because
// the KB4 Phase A uses an equidistant seed model (not the GT model), so
// Phase B starts further from the minimum and converges from a broader basin.
// ===========================================================================

TEST(KB4Fisheye, NoNoiseRecovery)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK)
    << "GT model failed validateCameraModel64";

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // No noise: pass a dummy RNG with sigma=0.
  std::mt19937 dummy_rng(42U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point failed rayToPixel64";
  ASSERT_EQ(views.size(), 5u);

  // GT poses for per-view comparison.
  const CanonicalGTPoses gt_poses = makeCanonicalGTPoses(board);

  KB4InitResult result;
  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateKB4Init returned non-OK";

  ASSERT_EQ(result.R_per_view.size(), 5u) << "R_per_view must have 5 entries";
  ASSERT_EQ(result.t_per_view.size(), 5u) << "t_per_view must have 5 entries";

  // --- Intrinsics recovery ---

  // K(2,2) = 1 convention.
  // The algorithm normalises K so K(2,2) = 1.0 exactly.
  EXPECT_NEAR(result.K(2, 2), 1.0, 1e-12) << "K(2,2) must equal 1.0 (algorithm output convention)";

  // Focal length relative error < 0.1 % (noise-free + Phase B).
  // Justification: Phase B Ceres with exact data converges to near-GT values;
  // 0.1% is a 3x safety margin over the expected ~0.03% convergence residual.
  EXPECT_NEAR(result.K(0, 0) / 200.0, 1.0, 0.001)
    << "fx relative error should be < 0.1% (noise-free + Phase B)";
  EXPECT_NEAR(result.K(1, 1) / 200.0, 1.0, 0.001)
    << "fy relative error should be < 0.1% (noise-free + Phase B)";

  // --- Distortion recovery ---

  // GT k3=k4=0: MS1-4's contract only identifies (fx, k1, k2). Phase B
  // holds k3, k4 fixed at 0. D(2) and D(3) must be exactly 0.0 on output.
  const Eigen::Vector4d D_gt(0.01, -0.005, 0.0, 0.0);
  const Eigen::Vector4d D_err = result.D - D_gt;
  // Tolerance 1e-4: Phase B on noise-free data; KB4 polynomial is well-
  // conditioned for these small k1, k2 values and a centred board at 1 m depth.
  EXPECT_LT(D_err.cwiseAbs().maxCoeff(), 1e-4)
    << "D recovery error |D - D_gt|_inf = " << D_err.cwiseAbs().maxCoeff()
    << " exceeds 1e-4 (noise-free + Phase B)";

  // k3 and k4 must be exactly 0.0 (Phase B holds them fixed per contract).
  EXPECT_EQ(result.D(2), 0.0) << "D(2) = k3 must be exactly 0.0 (MS1-4 does not estimate k3)";
  EXPECT_EQ(result.D(3), 0.0) << "D(3) = k4 must be exactly 0.0 (MS1-4 does not estimate k4)";

  // --- Per-view pose recovery ---

  for (std::size_t i = 0; i < 5u; ++i)
  {
    const double R_err = (result.R_per_view[i] - gt_poses.R[i]).norm();
    // Tolerance 1e-4: Phase B Ceres on noise-free data; R_err expected ~1e-7.
    // 1e-4 is a 3-order safety margin.
    EXPECT_NEAR(R_err, 0.0, 1e-4) << "View " << i << ": R recovery |R - R_gt|_F = " << R_err
                                  << " exceeds 1e-4 (noise-free + Phase B)";

    const double t_err = (result.t_per_view[i] - gt_poses.t[i]).norm();
    // Tolerance 1e-4 m: same justification as R_err.
    EXPECT_NEAR(t_err, 0.0, 1e-4) << "View " << i << ": t recovery |t - t_gt| = " << t_err
                                  << " m exceeds 1e-4 m (noise-free + Phase B)";
  }

  // --- Reprojection RMS ---

  const double rms = computeReprojectionRMS(result, views);
  // Tolerance 1e-3 px: noise-free data; after Phase B the model should
  // reproduce every projected pixel to near machine precision. 1e-3 is
  // conservative given that rayToPixel64 → Phase B Ceres convergence
  // should yield sub-1e-5 px RMS on exact data.
  EXPECT_LT(rms, 1e-3) << "Reprojection RMS = " << rms
                       << " px exceeds 1e-3 px on noise-free data after Phase B";
}

// ===========================================================================
// Test 2: NoisyRecovery
//
// Same 5-view / 8x6 setup as Test 1, but N(0, 0.5 px) Gaussian noise is
// added to every pixel observation (seed 42).
//
// GT uses k3=k4=0 because MS1-4's contract only identifies (fx, k1, k2).
// The downstream IntrinsicsCalibrator (MS2) refines higher-order
// coefficients with more data. |D - D_gt|_inf is dominated by k1, k2
// error under noise; the algorithm does not need to fight nonzero GT k3, k4.
//
// Tolerances (justified by Phase B Ceres refinement with moderate noise):
//   |fx - 200| / 200 < 0.02   (2 % relative; empirically ~1.1-1.2 % on Jetson
//                               aarch64 with seed 42; 2 % provides a 2x safety
//                               margin. The Cramér-Rao bound for 5 views / 48
//                               pts / sigma=0.5 px / fx=200 is ~0.4 %, so the
//                               Phase A equidistant-seed bias dominates; Phase B
//                               removes most but not all of it with only N=5
//                               views.)
//   |D - D_gt|_inf < 0.02     (k1, k2 error; empirically ~0.011 on Jetson;
//                               0.02 is a 2x margin. k3=D(2)=0, k4=D(3)=0
//                               exactly per Phase B contract.)
//   Reprojection RMS < 1.0 px (noisy data: dominated by noise level)
//
// We do not check per-view pose error under noise because MS1-3 (the linear
// DLT pose estimator) accumulates ~2-3 deg per view with 0.5 px noise, and
// Phase B partially corrects this. The reprojection RMS bound is the
// operationally meaningful criterion for the downstream IntrinsicsCalibrator.
// ===========================================================================

TEST(KB4Fisheye, NoisyRecovery)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK);

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 rng(42U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.5, &rng);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point failed rayToPixel64";

  KB4InitResult result;
  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateKB4Init returned non-OK under noise";

  ASSERT_EQ(result.R_per_view.size(), 5u);
  ASSERT_EQ(result.t_per_view.size(), 5u);

  // Focal length relative error < 2 %.
  // Justification: empirically ~1.1-1.2% on Jetson aarch64 (seed 42). The
  // Phase A equidistant-seed bias dominates; Phase B removes most of it but
  // with only 5 views the residual bias sits at ~1%. 2% is a 2x safety margin.
  EXPECT_NEAR(result.K(0, 0) / 200.0, 1.0, 0.02)
    << "fx relative error should be < 2% under 0.5 px noise + Phase B";
  EXPECT_NEAR(result.K(1, 1) / 200.0, 1.0, 0.02)
    << "fy relative error should be < 2% under 0.5 px noise + Phase B";

  // Distortion coefficient error < 0.02.
  // GT k3=k4=0: MS1-4's contract only identifies (fx, k1, k2). Phase B
  // holds k3, k4 fixed at 0. D(2) and D(3) must be exactly 0.0 on output.
  // Justification: Phase A linear LSQ absorbs most of the polynomial fit;
  // Phase B refines further. Empirically ~0.011 on Jetson aarch64 (seed 42).
  // With 48 pts * 5 views and sigma=0.5 px, Phase-A seed bias dominates;
  // 0.02 provides a 2x margin over the observed value.
  const Eigen::Vector4d D_gt(0.01, -0.005, 0.0, 0.0);
  const Eigen::Vector4d D_err = result.D - D_gt;
  EXPECT_LT(D_err.cwiseAbs().maxCoeff(), 0.02)
    << "D recovery error |D - D_gt|_inf = " << D_err.cwiseAbs().maxCoeff()
    << " exceeds 0.02 under 0.5 px noise";

  // k3 and k4 must be exactly 0.0 (Phase B holds them fixed per contract).
  EXPECT_EQ(result.D(2), 0.0) << "D(2) = k3 must be exactly 0.0 (MS1-4 does not estimate k3)";
  EXPECT_EQ(result.D(3), 0.0) << "D(3) = k4 must be exactly 0.0 (MS1-4 does not estimate k4)";

  // Reprojection RMS < 1.0 px.
  // Justification: Phase B minimises reprojection error; with 0.5 px
  // isotropic Gaussian noise the expected RMS is ~0.5 px * sqrt(2/N_avg_per_view)
  // per view (residual after fitting), well below 1.0 px.
  const double rms = computeReprojectionRMS(result, views);
  EXPECT_LT(rms, 1.0) << "Reprojection RMS = " << rms << " px exceeds 1.0 px under 0.5 px noise";
}

// ===========================================================================
// Test 3: RejectFewViews
//
// views.size() = 2 < kMinViews (3) -> INVALID_INPUT.
// result_out must be bit-for-bit unchanged (sentinel check).
// ===========================================================================

TEST(KB4Fisheye, RejectFewViews)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // Build exactly 2 views (each valid in isolation).
  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;
  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;

  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);

  const Eigen::Matrix3d R1 = rotMatX(+deg15);
  const Eigen::Vector3d t1 =
    R1 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(gt_model, R0, t0, board, 0.0, dummy_rng, views[0]));
  ASSERT_TRUE(buildView(gt_model, R1, t1, board, 0.0, dummy_rng, views[1]));

  // Pre-seed sentinel values.
  KB4InitResult result;
  result.K = Eigen::Matrix3d::Constant(99.0);
  result.D = Eigen::Vector4d::Constant(99.0);
  result.R_per_view = {Eigen::Matrix3d::Constant(77.0)};
  result.t_per_view = {Eigen::Vector3d::Constant(77.0)};

  const KB4InitResult sentinel = result;  // copy for comparison

  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for views.size() == 2";

  // result_out must be bit-for-bit unchanged.
  EXPECT_EQ((result.K - sentinel.K).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ((result.D - sentinel.D).norm(), 0.0)
    << "result.D must be unchanged when INVALID_INPUT is returned";
  ASSERT_EQ(result.R_per_view.size(), sentinel.R_per_view.size());
  EXPECT_EQ((result.R_per_view[0] - sentinel.R_per_view[0]).norm(), 0.0)
    << "result.R_per_view[0] must be unchanged";
  ASSERT_EQ(result.t_per_view.size(), sentinel.t_per_view.size());
  EXPECT_EQ((result.t_per_view[0] - sentinel.t_per_view[0]).norm(), 0.0)
    << "result.t_per_view[0] must be unchanged";
}

// ===========================================================================
// Test 4: RejectInsufficientPoints
//
// 3 views, but one view has only 3 corner points (M < 4) -> INVALID_INPUT.
// result_out unchanged.
// ===========================================================================

TEST(KB4Fisheye, RejectInsufficientPoints)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;
  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;

  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);

  const Eigen::Matrix3d R1 = rotMatX(+deg15);
  const Eigen::Vector3d t1 =
    R1 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  const Eigen::Matrix3d R2 = rotMatX(-deg15);
  const Eigen::Vector3d t2 =
    R2 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(gt_model, R0, t0, board, 0.0, dummy_rng, views[0]));
  ASSERT_TRUE(buildView(gt_model, R1, t1, board, 0.0, dummy_rng, views[1]));

  // Third view: only 3 board points (< 4 minimum).
  {
    const Eigen::Matrix2Xd small_board = board.leftCols(3);
    PlanarObservation bad;
    ASSERT_TRUE(buildView(gt_model, R2, t2, small_board, 0.0, dummy_rng, bad));
    views.push_back(bad);
  }

  const Eigen::Matrix3d K_sentinel = Eigen::Matrix3d::Constant(42.0);
  KB4InitResult result;
  result.K = K_sentinel;
  result.D = Eigen::Vector4d::Constant(42.0);

  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT when a view has M = 3 < 4 points";
  EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Test 5: RejectImageSizeZero
//
// image_width = 0 -> INVALID_INPUT.
// image_height = -1 -> INVALID_INPUT.
// result_out unchanged in both cases.
// ===========================================================================

TEST(KB4Fisheye, RejectImageSizeZero)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty());

  const Eigen::Matrix3d K_sentinel = Eigen::Matrix3d::Constant(55.0);

  // Case A: image_width = 0.
  {
    KB4InitResult result;
    result.K = K_sentinel;
    result.D = Eigen::Vector4d::Constant(55.0);

    const StatusCode sc = estimateKB4Init(views, 0, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for image_width = 0";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when image_width = 0";
  }

  // Case B: image_height = -1.
  {
    KB4InitResult result;
    result.K = K_sentinel;
    result.D = Eigen::Vector4d::Constant(55.0);

    const StatusCode sc = estimateKB4Init(views, 640, -1, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for image_height = -1";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when image_height = -1";
  }
}

// ===========================================================================
// Test 6: RejectColumnMismatch
//
// 3 views, one view has board_pts.cols() != image_pts.cols() -> INVALID_INPUT.
// result_out unchanged.
// ===========================================================================

TEST(KB4Fisheye, RejectColumnMismatch)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;
  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;

  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);

  const Eigen::Matrix3d R1 = rotMatX(+deg15);
  const Eigen::Vector3d t1 =
    R1 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views(2);
  ASSERT_TRUE(buildView(gt_model, R0, t0, board, 0.0, dummy_rng, views[0]));
  ASSERT_TRUE(buildView(gt_model, R1, t1, board, 0.0, dummy_rng, views[1]));

  // Third view: board_pts has 12 columns, image_pts has 10 (mismatch).
  {
    PlanarObservation bad;
    bad.board_pts = board.leftCols(12);  // 12 board points
    bad.image_pts = board.leftCols(10);  // 10 image points (mismatch)
    views.push_back(bad);
  }

  const Eigen::Matrix3d K_sentinel = Eigen::Matrix3d::Constant(33.0);
  KB4InitResult result;
  result.K = K_sentinel;
  result.D = Eigen::Vector4d::Constant(33.0);

  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for board_pts.cols() (12) != image_pts.cols() (10)";
  EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Test 7: RejectNonFinite
//
// 3 views; one view has a NaN pixel, then separately an +inf pixel.
// Each case -> INVALID_INPUT; result_out unchanged.
// ===========================================================================

TEST(KB4Fisheye, RejectNonFinite)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;
  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;

  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);

  const Eigen::Matrix3d R1 = rotMatX(+deg15);
  const Eigen::Vector3d t1 =
    R1 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  const Eigen::Matrix3d R2 = rotMatX(-deg15);
  const Eigen::Vector3d t2 =
    R2 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> base_views(2);
  ASSERT_TRUE(buildView(gt_model, R0, t0, board, 0.0, dummy_rng, base_views[0]));
  ASSERT_TRUE(buildView(gt_model, R1, t1, board, 0.0, dummy_rng, base_views[1]));

  PlanarObservation good_third;
  ASSERT_TRUE(buildView(gt_model, R2, t2, board, 0.0, dummy_rng, good_third));

  const Eigen::Matrix3d K_sentinel = Eigen::Matrix3d::Constant(22.0);

  // Case A: NaN in image_pts of the third view.
  {
    PlanarObservation bad = good_third;  // copy good view
    bad.image_pts(0, 5) = std::numeric_limits<double>::quiet_NaN();

    std::vector<PlanarObservation> views = base_views;
    views.push_back(bad);

    KB4InitResult result;
    result.K = K_sentinel;
    result.D = Eigen::Vector4d::Constant(22.0);

    const StatusCode sc = estimateKB4Init(views, 640, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
      << "Expected INVALID_INPUT when image_pts contains NaN";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when NaN triggers INVALID_INPUT";
  }

  // Case B: +inf in image_pts of the third view.
  {
    PlanarObservation bad = good_third;  // copy good view
    bad.image_pts(1, 3) = std::numeric_limits<double>::infinity();

    std::vector<PlanarObservation> views = base_views;
    views.push_back(bad);

    KB4InitResult result;
    result.K = K_sentinel;
    result.D = Eigen::Vector4d::Constant(22.0);

    const StatusCode sc = estimateKB4Init(views, 640, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
      << "Expected INVALID_INPUT when image_pts contains +inf";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when +inf triggers INVALID_INPUT";
  }
}

// ===========================================================================
// Test 8: DegenerateFrontParallel
//
// 3 views all with R = Identity (front-parallel board), only translation
// differs.
//
// IMPORTANT DIFFERENCE FROM PINHOLE ZHANG: Unlike the pinhole case (where
// all-front-parallel views create a degenerate IAC null-space because the
// orthonormality constraints are identical), the KB4 fisheye path uses
// MS1-3 estimatePoseDLT with bearing-lifting (pixelToRay64). The fisheye
// bearing lifting encodes the incidence angle at each pixel, which gives
// well-defined 3D bearings even for front-parallel views — the DLT pose
// recovery via cross-product equation is NOT degenerate for a fisheye
// model with a flat board, because angle information is preserved.
//
// Therefore front-parallel views with a fisheye model legitimately succeed
// (StatusCode::OK), unlike pinhole Zhang's degenerate IAC case.
//
// This test verifies two things:
//   (a) The algorithm does NOT crash or return NUMERIC_ERROR.
//   (b) The reprojection RMS on the noise-free front-parallel data is small
//       (< 1e-2 px), confirming a usable initial guess is produced.
//
// The degenerate-input quota for the mandatory test patterns is satisfied by
// Tests 3–7 (INVALID_INPUT cases). The present test instead validates a
// "surprisingly-OK" case that the test author initially expected to be
// degenerate — documenting the correct fisheye-DLT behavior.
// ===========================================================================

TEST(KB4Fisheye, DegenerateFrontParallel)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  // All three views: R = Identity, only translation differs slightly.
  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;
  const Eigen::Matrix3d R_id = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);
  const Eigen::Vector3d t1(-board_cx + 0.05, -board_cy, 1.0);
  const Eigen::Vector3d t2(-board_cx, -board_cy + 0.05, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views(3);
  ASSERT_TRUE(buildView(gt_model, R_id, t0, board, 0.0, dummy_rng, views[0]));
  ASSERT_TRUE(buildView(gt_model, R_id, t1, board, 0.0, dummy_rng, views[1]));
  ASSERT_TRUE(buildView(gt_model, R_id, t2, board, 0.0, dummy_rng, views[2]));

  KB4InitResult result;

  // Front-parallel fisheye DLT is NOT degenerate (see comment above).
  // The algorithm must return OK and not crash.
  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::OK) << "Fisheye DLT with front-parallel views is well-defined (bearing "
                                   "angles are preserved); expected OK, not DEGENERATE_CONFIG";

  if (sc == StatusCode::OK)
  {
    // When the algorithm succeeds, verify the result is a usable initial guess:
    // reprojection RMS on the noise-free data should be small.
    // Tolerance 1e-2 px: Phase B on only 3 front-parallel views (less
    // observational diversity than the canonical 5-view tilted setup) may not
    // converge as tightly; 1e-2 is a safe bound for a no-noise fisheye init.
    const double rms = computeReprojectionRMS(result, views);
    EXPECT_LT(rms, 1e-2) << "Front-parallel fisheye init reprojection RMS = " << rms
                         << " px exceeds 1e-2 px (noise-free 3 views)";

    ASSERT_EQ(result.R_per_view.size(), 3u);
    ASSERT_EQ(result.t_per_view.size(), 3u);

    // K(2,2) = 1 convention must always hold on OK return.
    EXPECT_NEAR(result.K(2, 2), 1.0, 1e-12) << "K(2,2) must equal 1.0 on OK return";
    EXPECT_GT(result.K(0, 0), 0.0) << "fx must be positive";
    EXPECT_GT(result.K(1, 1), 0.0) << "fy must be positive";
  }
}

// ===========================================================================
// Test 9: SentinelPreserved
//
// Pre-seed result_out with sentinel values. Trigger INVALID_INPUT by passing
// views.size() = 2. Assert result_out is bit-for-bit unchanged.
//
// This is a focused guard against partial mutation — if the algorithm
// inadvertently writes to result_out before returning a non-OK status, this
// test will catch it.
// ===========================================================================

TEST(KB4Fisheye, SentinelPreserved)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  const double board_cx = 3.5 * 0.05;
  const double board_cy = 2.5 * 0.05;
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(-board_cx, -board_cy, 1.0);
  const Eigen::Matrix3d R1 = rotMatX(15.0 * camxiom::constants::kPi / 180.0);
  const Eigen::Vector3d t1 =
    R1 * Eigen::Vector3d(-board_cx, -board_cy, 0.0) + Eigen::Vector3d(0.0, 0.0, 1.0);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views(2);  // only 2 views: triggers INVALID_INPUT
  ASSERT_TRUE(buildView(gt_model, R0, t0, board, 0.0, dummy_rng, views[0]));
  ASSERT_TRUE(buildView(gt_model, R1, t1, board, 0.0, dummy_rng, views[1]));

  // Pre-seed with sentinel values that are clearly non-default.
  KB4InitResult result;
  result.K = Eigen::Matrix3d::Constant(99.0);
  result.D = Eigen::Vector4d::Constant(99.0);
  result.R_per_view = {Eigen::Matrix3d::Constant(99.0), Eigen::Matrix3d::Constant(99.0)};
  result.t_per_view = {Eigen::Vector3d::Constant(99.0), Eigen::Vector3d::Constant(99.0)};

  const KB4InitResult sentinel = result;  // copy before call

  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for views.size() == 2";

  // Verify all fields are bit-for-bit unchanged.
  EXPECT_EQ((result.K - sentinel.K).norm(), 0.0)
    << "result.K must be unchanged after INVALID_INPUT";
  EXPECT_EQ((result.D - sentinel.D).norm(), 0.0)
    << "result.D must be unchanged after INVALID_INPUT";
  ASSERT_EQ(result.R_per_view.size(), sentinel.R_per_view.size());
  for (std::size_t i = 0; i < sentinel.R_per_view.size(); ++i)
  {
    EXPECT_EQ((result.R_per_view[i] - sentinel.R_per_view[i]).norm(), 0.0)
      << "result.R_per_view[" << i << "] must be unchanged after INVALID_INPUT";
  }
  ASSERT_EQ(result.t_per_view.size(), sentinel.t_per_view.size());
  for (std::size_t i = 0; i < sentinel.t_per_view.size(); ++i)
  {
    EXPECT_EQ((result.t_per_view[i] - sentinel.t_per_view[i]).norm(), 0.0)
      << "result.t_per_view[" << i << "] must be unchanged after INVALID_INPUT";
  }
}

// ===========================================================================
// Test: PeripheralBoardRecovery
//
// Regression for the seed FOV cap. The old Phase A seed (theta_max = pi/2,
// fx = height/pi) could only lift pixels within height/2 = 240 px of the
// principal point; a single corner beyond that radius made estimatePoseDLT
// return non-OK and aborted the whole init with DEGENERATE_CONFIG. Boards
// pushed into the image periphery are exactly what good fisheye coverage
// requires, so this must succeed.
//
// Setup: the 5 canonical views plus a 6th view placed deep in the right
// image periphery. The test asserts the peripheral view really does contain
// corners beyond 240 px (the old kill condition) before running the init.
// ===========================================================================

TEST(KB4Fisheye, PeripheralBoardRecovery)
{
  const CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK);

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 dummy_rng(7U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty());
  ASSERT_EQ(views.size(), 5u);

  // 6th view: board shifted far right, tilted back towards the camera.
  const Eigen::Matrix3d R5 = rotMatY(-40.0 * camxiom::constants::kPi / 180.0);
  const Eigen::Vector3d t5(0.55, -0.175, 0.15);
  PlanarObservation side_view;
  ASSERT_TRUE(buildView(gt_model, R5, t5, board, 0.0, dummy_rng, side_view));

  double r_max = 0.0;
  for (Eigen::Index j = 0; j < side_view.image_pts.cols(); ++j)
  {
    r_max = std::max(
      r_max, std::hypot(side_view.image_pts(0, j) - 320.0, side_view.image_pts(1, j) - 240.0)
    );
  }
  ASSERT_GT(r_max, 240.0) << "fixture error: the peripheral view must contain corners beyond "
                             "height/2 (the radius the old pi/2-capped seed rejected)";
  views.push_back(side_view);

  KB4InitResult result;
  const StatusCode sc = estimateKB4Init(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateKB4Init must handle peripheral coverage";

  // No-noise recovery; slightly looser than NoNoiseRecovery because the
  // oblique peripheral view adds leverage but also a wider theta range.
  EXPECT_NEAR(result.K(0, 0) / 200.0, 1.0, 0.005);
  EXPECT_NEAR(result.K(1, 1) / 200.0, 1.0, 0.005);
  const Eigen::Vector4d D_gt(0.01, -0.005, 0.0, 0.0);
  EXPECT_LT((result.D - D_gt).cwiseAbs().maxCoeff(), 1e-3) << "D = " << result.D.transpose();
  EXPECT_LT(computeReprojectionRMS(result, views), 0.01);
}

// ===========================================================================
// Test: UltraWideFovRecovery
//
// End-to-end init on a >180-deg-FOV fisheye: a KB4 camera whose usable
// theta range extends past pi/2, observed with one board placed at ~100 deg
// incidence (physically: beside and slightly behind the camera, facing it).
// KB4 parameterises theta directly, so nothing in the pipeline should cap
// at 90 deg — the old pi/2 seed did exactly that.
//
// GT: 800x600, fx = 170, theta_max = 2.6 rad (inside the k1=0.01/k2=-0.005
// polynomial's monotone range, which folds at ~2.64 and whose cap the
// validator now certifies). The oblique view spans theta in ~[82, 119] deg;
// the test asserts corners with theta > pi/2 are actually present. Verifies
// the Phase A bootstrap converges from the equidistant seed on ultra-wide
// geometry (the residual-risk case flagged during review) and that Phase B
// recovers the ground truth.
// ===========================================================================

TEST(KB4Fisheye, UltraWideFovRecovery)
{
  CameraModel64 gt_model = buildGroundTruthFisheyeKB4Model();
  gt_model.intrinsics.fx = 170.0;
  gt_model.intrinsics.fy = 170.0;
  gt_model.intrinsics.cx = 400.0;
  gt_model.intrinsics.cy = 300.0;
  gt_model.projection.theta_max = 2.6;
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK);

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 dummy_rng(11U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty());

  // Oblique view: board centred on the viewing ray at theta_c = 1.75 rad
  // (~100 deg), azimuth 45 deg, 0.65 m away, board normal facing the camera.
  const double theta_c = 1.75;
  const double azimuth = 45.0 * camxiom::constants::kPi / 180.0;
  const double dist = 0.65;
  const Eigen::Vector3d dir(
    std::sin(theta_c) * std::cos(azimuth), std::sin(theta_c) * std::sin(azimuth), std::cos(theta_c)
  );
  const Eigen::Vector3d ez = -dir;  // board normal looks back at the camera
  const Eigen::Vector3d ex = Eigen::Vector3d::UnitY().cross(ez).normalized();
  const Eigen::Vector3d ey = ez.cross(ex);
  Eigen::Matrix3d R5;
  R5.col(0) = ex;
  R5.col(1) = ey;
  R5.col(2) = ez;
  const Eigen::Vector3d board_center(
    0.5 * board.row(0).maxCoeff(), 0.5 * board.row(1).maxCoeff(), 0.0
  );
  const Eigen::Vector3d t5 = dist * dir - R5 * board_center;

  PlanarObservation oblique_view;
  ASSERT_TRUE(buildView(gt_model, R5, t5, board, 0.0, dummy_rng, oblique_view));

  // The oblique view must really exercise the >90-deg regime.
  int beyond_half_pi = 0;
  for (Eigen::Index j = 0; j < board.cols(); ++j)
  {
    const Eigen::Vector3d p_cam = R5 * Eigen::Vector3d(board(0, j), board(1, j), 0.0) + t5;
    if (computeTheta(p_cam) > camxiom::constants::kHalfPi) ++beyond_half_pi;
  }
  ASSERT_GT(
    beyond_half_pi, 0
  ) << "fixture error: the oblique view must contain corners past 90 deg";
  views.push_back(oblique_view);

  KB4InitResult result;
  const StatusCode sc = estimateKB4Init(views, 800, 600, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateKB4Init must handle a >180-deg-FOV observation set";

  EXPECT_NEAR(result.K(0, 0) / 170.0, 1.0, 0.005) << "fx = " << result.K(0, 0);
  EXPECT_NEAR(result.K(1, 1) / 170.0, 1.0, 0.005) << "fy = " << result.K(1, 1);
  const Eigen::Vector4d D_gt(0.01, -0.005, 0.0, 0.0);
  EXPECT_LT((result.D - D_gt).cwiseAbs().maxCoeff(), 1e-3) << "D = " << result.D.transpose();

  // Reprojection check with the recovered parameters on the full theta
  // range (the shared helper caps its model at pi/2 and would silently skip
  // the oblique view, so rebuild the recovered model at the GT theta_max).
  CameraModel64 rec_model = gt_model;
  rec_model.intrinsics.fx = result.K(0, 0);
  rec_model.intrinsics.fy = result.K(1, 1);
  rec_model.intrinsics.cx = result.K(0, 2);
  rec_model.intrinsics.cy = result.K(1, 2);
  for (int i = 0; i < 4; ++i)
  {
    rec_model.distortion.coeffs[static_cast<std::size_t>(i)] = result.D(i);
  }
  ASSERT_EQ(camxiom::validateCameraModel64(rec_model), StatusCode::OK);

  double sum_sq = 0.0;
  int count = 0;
  for (std::size_t v = 0; v < views.size(); ++v)
  {
    for (Eigen::Index j = 0; j < views[v].board_pts.cols(); ++j)
    {
      const Eigen::Vector3d p_cam =
        result.R_per_view[v] *
          Eigen::Vector3d(views[v].board_pts(0, j), views[v].board_pts(1, j), 0.0) +
        result.t_per_view[v];
      const PixelResult64 pr = camxiom::rayToPixel64(rec_model, p_cam);
      ASSERT_EQ(pr.status, StatusCode::OK)
        << "recovered model must project every observed corner (view " << v << ", corner " << j
        << ")";
      const double du = pr.pixel.u - views[v].image_pts(0, j);
      const double dv = pr.pixel.v - views[v].image_pts(1, j);
      sum_sq += du * du + dv * dv;
      ++count;
    }
  }
  ASSERT_GT(count, 0);
  EXPECT_LT(std::sqrt(sum_sq / count), 0.01) << "full-range reprojection RMS too large";
}
