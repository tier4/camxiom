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

// Unit tests for camxiom::init::estimateMEIInit.
//
// Strategy: build a ground-truth MEI (Mei 2007) omnidirectional CameraModel64
// with known intrinsics K and xi, project an 8x6 checkerboard at 5 varied
// poses through the forward path (rayToPixel64), feed the synthetic pixel
// observations back to estimateMEIInit, and assert that the recovered K, xi,
// and per-view poses match ground truth within tolerance.
//
// All tests are deterministic (seed 42 where noise is involved) and
// self-contained — each TEST() rebuilds its own ground truth.

#include "camxiom/init/mei_omni.hpp"

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
using camxiom::init::estimateMEIInit;
using camxiom::init::MEIInitResult;
using camxiom::init::PlanarObservation;

namespace
{

// ---------------------------------------------------------------------------
// Ground-truth model builder.
//
// Sensor: 640x480.
// Projection: OMNIDIRECTIONAL (Mei 2007 unified-sphere model).
// Intrinsics: fx = fy = 200 px, cx = 320, cy = 240 (no skew).
// xi: 1.0  <-- LOCKED AT THE PHASE A SEED VALUE.
//
// MS1-5 CONTRACT: estimateMEIInit's Phase B runs PnpSolver with
// FIX_PROJECTION_PARAMS, which keeps xi fixed at the Phase A seed of 1.0.
// Joint refinement of (K, xi, poses) in the linear-init-then-Ceres pipeline
// is unreliable — xi tends to stay near the seed or jump to non-physical
// values. MS2's full IntrinsicsCalibrator refines xi with more data and
// multi-seed restarts. Therefore, the GT model uses xi = 1.0 so that
// the recovered xi can be asserted with EXPECT_EQ (bit-exact, not numerical).
//
// With fx = 200 and xi = 1.0 (parabolic-mirror special case), the forward
// projection radius for a ray at incidence angle theta is:
//   r ~ fx * sin(theta) / (cos(theta) + 1.0)
// At theta = 0.8 rad (~46 deg): r ~ 200 * 0.72 / (0.70 + 1.0) ~ 85 px,
// well within the 240 px half-height for the 640x480 sensor.
//
// Distortion is held at NONE (MEI's xi captures the fisheye behaviour;
// plane-space radial distortion is deferred to MS2).
// ---------------------------------------------------------------------------

CameraModel64 buildGroundTruthMEIModel()
{
  CameraModel64 m{};

  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;

  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  // xi = 1.0: matches Phase A seed that Phase B holds fixed (FIX_PROJECTION_PARAMS).
  // See MS1-5 contract in mei_omni.hpp docstring.
  m.projection.xi = 1.0;
  // theta_max: pi/2 is the default and covers the geometry below.
  m.projection.theta_max = 1.5707963267948966;

  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.coeffs.fill(0.0);
  m.distortion.count = 0U;
  m.distortion.is_rational = false;
  m.distortion.has_thin_prism = false;
  m.distortion.has_tilt = false;
  m.distortion.tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  m.distortion.inv_tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

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
// Helper: compute theta (incident angle from optical axis) for a camera-frame
// 3D point. Returns NaN if z <= 0 (behind camera).
// ---------------------------------------------------------------------------

double computeTheta(const Eigen::Vector3d &p_cam)
{
  const double r_xy = std::sqrt(p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y());
  return std::atan2(r_xy, p_cam.z());
}

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
// computeReprojectionRMS: after estimateMEIInit returns, compute the RMS
// reprojection error (pixels) using the recovered K, xi, and per-view poses.
//
// Builds a CameraModel64 with OMNIDIRECTIONAL projection and the recovered
// (fx, fy, cx, cy, xi), then projects each board point through the recovered
// extrinsics and computes the pixel distance to the observed image point.
// ---------------------------------------------------------------------------

double computeReprojectionRMS(const MEIInitResult &res, const std::vector<PlanarObservation> &views)
{
  // Build a CameraModel64 from the recovered K and xi.
  CameraModel64 rec_model{};
  rec_model.intrinsics.fx = res.K(0, 0);
  rec_model.intrinsics.fy = res.K(1, 1);
  rec_model.intrinsics.cx = res.K(0, 2);
  rec_model.intrinsics.cy = res.K(1, 2);
  rec_model.intrinsics.skew = 0.0;

  rec_model.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  rec_model.projection.xi = res.xi;
  rec_model.projection.theta_max = 1.5707963267948966;

  rec_model.distortion.type = DistortionModelType::NONE;
  rec_model.distortion.space = DistortionSpace::NONE;
  rec_model.distortion.coeffs.fill(0.0);
  rec_model.distortion.count = 0U;
  rec_model.distortion.is_rational = false;
  rec_model.distortion.has_thin_prism = false;
  rec_model.distortion.has_tilt = false;
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
// Canonical 5-view setup designed for MEI omnidirectional identifiability.
//
// Board: 8x6 checkerboard, 0.05 m spacing, 48 corners (X in [0, 0.35] m,
// Y in [0, 0.25] m, Z = 0 in board frame).
//
// MEI IDENTIFIABILITY RATIONALE:
// The xi parameter is identifiable only when the observations span a wide
// range of incidence angles theta. For sigma = 0.5 px and fx = 200, we
// need theta_max >= 0.7 rad (40 deg) so that the curvature of the
// omnidirectional projection function is clearly visible above the noise
// floor. This mirrors the same threshold established for KB4 in MS1-4.
//
// The board is placed CLOSE (0.30–0.40 m) at varied tilts to push the
// board corners into wide-angle regions. For the GT model (xi = 0.8,
// fx = 200), all board points project to valid pixels on the 640x480
// sensor at these distances.
//
// View 0: R=I, t=(0, 0, 0.30).
// View 1: Rx(+30°), t=(0, -0.15, 0.30).
// View 2: Ry(-30°), t=(-0.15, 0, 0.35).
// View 3: Rx(+25°)·Ry(-25°), t=(0.10, 0.10, 0.35).
// View 4: Rx(+20°)·Ry(+20°)·Rz(+15°), t=(-0.10, 0.12, 0.40).
//
// A runtime assertion confirms theta_max > 0.7 rad across all views.
// Returns an empty vector if any rayToPixel64 call fails.
// ---------------------------------------------------------------------------

std::vector<PlanarObservation> makeCanonicalViews(
  const CameraModel64 &gt_model, const Eigen::Matrix2Xd &board, double noise_sigma = 0.0,
  std::mt19937 *rng = nullptr
)
{
  const double deg30 = 30.0 * camxiom::constants::kPi / 180.0;
  const double deg25 = 25.0 * camxiom::constants::kPi / 180.0;
  const double deg20 = 20.0 * camxiom::constants::kPi / 180.0;
  const double deg15 = 15.0 * camxiom::constants::kPi / 180.0;

  // View 0: frontal view, board at 0.30 m.
  const Eigen::Matrix3d R0 = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d t0(0.0, 0.0, 0.30);

  // View 1: +30 deg around X (board tips backward), t = (0, -0.15, 0.30).
  const Eigen::Matrix3d R1 = rotMatX(+deg30);
  const Eigen::Vector3d t1(0.0, -0.15, 0.30);

  // View 2: -30 deg around Y (board tips rightward), t = (-0.15, 0, 0.35).
  const Eigen::Matrix3d R2 = rotMatY(-deg30);
  const Eigen::Vector3d t2(-0.15, 0.0, 0.35);

  // View 3: combined +25 X / -25 Y tilt, t = (0.10, 0.10, 0.35).
  const Eigen::Matrix3d R3 = rotMatX(+deg25) * rotMatY(-deg25);
  const Eigen::Vector3d t3(0.1, 0.1, 0.35);

  // View 4: diverse orientation (+20 X, +20 Y, +15 Z), t = (-0.10, 0.12, 0.40).
  const Eigen::Matrix3d R4 = rotMatX(+deg20) * rotMatY(+deg20) * rotMatZ(+deg15);
  const Eigen::Vector3d t4(-0.1, 0.12, 0.40);

  // Runtime guard: verify that the geometry spans at least 0.7 rad.
  // This check catches regressions if someone narrows the theta range below
  // the MEI xi identifiability threshold.
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
    // 0.7 rad ≈ 40 deg: minimum for MEI xi identifiability with sigma=0.5 px
    // noise. If this fires, the pose geometry must be revised.
    EXPECT_GE(theta_max_global, 0.7)
      << "makeCanonicalViews: theta_max = " << theta_max_global
      << " rad < 0.7 rad — geometry spans too narrow a theta range for "
         "MEI xi identifiability";
  }

  // Build a dummy RNG for the no-noise case.
  std::mt19937 dummy_rng(0U);
  std::mt19937 &use_rng = (rng != nullptr) ? *rng : dummy_rng;

  std::vector<PlanarObservation> views(5);
  if (!buildView(gt_model, R0, t0, board, noise_sigma, use_rng, views[0])) return {};
  if (!buildView(gt_model, R1, t1, board, noise_sigma, use_rng, views[1])) return {};
  if (!buildView(gt_model, R2, t2, board, noise_sigma, use_rng, views[2])) return {};
  if (!buildView(gt_model, R3, t3, board, noise_sigma, use_rng, views[3])) return {};
  if (!buildView(gt_model, R4, t4, board, noise_sigma, use_rng, views[4])) return {};

  return views;
}

// Precomputed GT poses matching the canonical setup (for per-view pose checks).
struct CanonicalGTPoses
{
  std::vector<Eigen::Matrix3d> R;
  std::vector<Eigen::Vector3d> t;
};

CanonicalGTPoses makeCanonicalGTPoses()
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
// Ground-truth MEI model (fx=fy=200, cx=320, cy=240, xi=1.0) projected
// exactly (no noise) across 5 views of an 8x6 checkerboard at 0.05 m
// spacing.
//
// MS1-5 CONTRACT: estimateMEIInit only identifies (K, poses); xi is held
// fixed at the Phase A seed of 1.0 (FIX_PROJECTION_PARAMS in Phase B).
// The GT model is therefore built with xi = 1.0 so that the test can use
// EXPECT_EQ(result.xi, 1.0) — a bit-exact check, not a numerical tolerance.
// MS2's full IntrinsicsCalibrator is responsible for refining xi.
//
// Tolerances (justified by Phase B Ceres convergence on noise-free data):
//   |fx - 200| / 200 < 0.001  (0.1 % relative error on focal length;
//                               Phase B with exact data + xi=GT converges
//                               to near-GT)
//   xi == 1.0 exactly          (Phase B locks xi at the Phase A seed; this
//                               is a bit-exact assertion, not a tolerance)
//   Reprojection RMS < 1e-3 px (noise-free data + GT xi: near machine
//                               precision after Phase B Ceres)
//   R per view: |R^T R - I|_F < 1e-12  (Phase A Procrustes + Phase B
//                               enforce orthonormality)
//   t per view: t.z() > 0     (board must be in front of the camera for
//                               all 5 canonical views)
// ===========================================================================

TEST(MEIOmni, NoNoiseRecovery)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK)
    << "GT model failed validateCameraModel64";

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 dummy_rng(42U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point failed rayToPixel64";
  ASSERT_EQ(views.size(), 5u);

  const CanonicalGTPoses gt_poses = makeCanonicalGTPoses();

  MEIInitResult result;
  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateMEIInit returned non-OK";

  ASSERT_EQ(result.R_per_view.size(), 5u) << "R_per_view must have 5 entries";
  ASSERT_EQ(result.t_per_view.size(), 5u) << "t_per_view must have 5 entries";

  // --- Intrinsics recovery ---

  // K(2,2) = 1 convention: the algorithm normalises K so K(2,2) = 1.0 exactly.
  EXPECT_NEAR(result.K(2, 2), 1.0, 1e-12) << "K(2,2) must equal 1.0 (algorithm output convention)";

  // Focal length relative error < 0.1 % (noise-free + Phase B with xi=GT).
  // Justification: Phase B Ceres with exact data and xi fixed at GT value
  // converges to near-GT K; 0.1% is a 3x safety margin.
  EXPECT_NEAR(result.K(0, 0) / 200.0, 1.0, 0.001)
    << "fx relative error should be < 0.1% (noise-free + Phase B, xi locked); "
       "actual fx = "
    << result.K(0, 0);
  EXPECT_NEAR(result.K(1, 1) / 200.0, 1.0, 0.001)
    << "fy relative error should be < 0.1% (noise-free + Phase B, xi locked); "
       "actual fy = "
    << result.K(1, 1);

  // --- xi lock assertion ---

  // MS1-5 CONTRACT: xi is locked at the Phase A seed (1.0) by Phase B's
  // FIX_PROJECTION_PARAMS flag. This is a bit-exact assertion — the value
  // should come back as exactly 1.0. MS2 is responsible for refining xi.
  EXPECT_EQ(
    result.xi, 1.0
  ) << "xi must equal exactly 1.0 (Phase A seed locked by FIX_PROJECTION_PARAMS); "
       "actual xi = "
    << result.xi;

  // --- Per-view pose recovery ---

  for (std::size_t i = 0; i < 5u; ++i)
  {
    // Orthonormality: R^T R should be close to I (Procrustes step enforces this).
    const Eigen::Matrix3d RtR = result.R_per_view[i].transpose() * result.R_per_view[i];
    const double ortho_err = (RtR - Eigen::Matrix3d::Identity()).norm();
    // Tolerance 1e-12: Procrustes + Phase B on noise-free data should give
    // near-machine-precision orthonormality.
    EXPECT_LT(ortho_err, 1e-12) << "View " << i << ": |R^T R - I|_F = " << ortho_err
                                << " exceeds 1e-12";

    // det(R) must be close to +1 (proper rotation, not reflection).
    const double det_R = result.R_per_view[i].determinant();
    EXPECT_GT(det_R, 0.99) << "View " << i << ": det(R) = " << det_R
                           << " is not close to +1 (expected a proper rotation)";

    // t.z() > 0: board must be in front of the camera for canonical poses.
    EXPECT_GT(result.t_per_view[i].z(), 0.0)
      << "View " << i << ": t.z() = " << result.t_per_view[i].z()
      << " <= 0 (board should be in front of the camera)";
  }

  // --- Reprojection RMS ---

  const double rms = computeReprojectionRMS(result, views);
  // Tolerance 1e-3 px: noise-free data; after Phase B the model should
  // reproduce every projected pixel to near machine precision.
  EXPECT_LT(rms, 1e-3) << "Reprojection RMS = " << rms
                       << " px exceeds 1e-3 px on noise-free data after Phase B";
}

// ===========================================================================
// Test 2: NoisyRecovery
//
// Same 5-view / 8x6 setup as Test 1, but N(0, 0.5 px) Gaussian noise is
// added to every pixel observation (seed 42). GT model uses xi = 1.0.
//
// MS1-5 CONTRACT: xi is locked at 1.0 (Phase A seed, FIX_PROJECTION_PARAMS).
// Since GT xi = 1.0, the recovered model should be self-consistent and
// produce low reprojection error despite noise in K.
//
// Tolerances (justified by Phase B Ceres refinement with moderate noise):
//   |fx - 200| / 200 < 0.02   (2 % relative; Phase A equidistant-seed bias
//                               is partially corrected by Phase B with xi=GT;
//                               N=5 views, sigma=0.5 px expected ~1-2 %. 2%
//                               is a 2x safety margin.)
//   xi == 1.0 exactly          (bit-exact: Phase B keeps xi at the seed)
//   Reprojection RMS < 1.0 px (noisy data: dominated by 0.5 px input noise;
//                               with GT xi=1.0 the model fits cleanly)
// ===========================================================================

TEST(MEIOmni, NoisyRecovery)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
  ASSERT_EQ(camxiom::validateCameraModel64(gt_model), StatusCode::OK);

  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 rng(42U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.5, &rng);
  ASSERT_FALSE(views.empty()) << "Canonical view setup: a board point failed rayToPixel64";

  MEIInitResult result;
  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::OK) << "estimateMEIInit returned non-OK under noise";

  ASSERT_EQ(result.R_per_view.size(), 5u);
  ASSERT_EQ(result.t_per_view.size(), 5u);

  // Focal length relative error < 2 %.
  // Justification: Phase A equidistant-seed bias is partially resolved by Phase B
  // (xi is already at GT value of 1.0, so no xi-induced bias); empirically
  // ~1-2 % expected for N=5 views, sigma=0.5 px. 2% is a 2x margin.
  EXPECT_NEAR(result.K(0, 0) / 200.0, 1.0, 0.02)
    << "fx relative error should be < 2% under 0.5 px noise + Phase B (xi locked); "
       "actual fx = "
    << result.K(0, 0);
  EXPECT_NEAR(result.K(1, 1) / 200.0, 1.0, 0.02)
    << "fy relative error should be < 2% under 0.5 px noise + Phase B (xi locked); "
       "actual fy = "
    << result.K(1, 1);

  // MS1-5 CONTRACT: xi locked at 1.0 by Phase B FIX_PROJECTION_PARAMS.
  // Bit-exact assertion — no tolerance needed.
  EXPECT_EQ(
    result.xi, 1.0
  ) << "xi must equal exactly 1.0 (Phase A seed locked by FIX_PROJECTION_PARAMS); "
       "actual xi = "
    << result.xi;

  // Reprojection RMS < 1.0 px.
  // Justification: Phase B minimises reprojection error; with 0.5 px isotropic
  // Gaussian noise and GT xi = 1.0 the model fits cleanly. Expected RMS ~0.5 px,
  // well below 1.0 px.
  const double rms = computeReprojectionRMS(result, views);
  EXPECT_LT(rms, 1.0) << "Reprojection RMS = " << rms << " px exceeds 1.0 px under 0.5 px noise";
}

// ===========================================================================
// Test 3: RejectFewViews
//
// views.size() = 2 < kMinViews (3) -> INVALID_INPUT.
// result_out must be bit-for-bit unchanged (sentinel check).
// ===========================================================================

TEST(MEIOmni, RejectFewViews)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
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
  MEIInitResult result;
  result.K = Eigen::Matrix3d::Constant(99.0);
  result.xi = 99.0;
  result.R_per_view = {Eigen::Matrix3d::Constant(77.0)};
  result.t_per_view = {Eigen::Vector3d::Constant(77.0)};

  const MEIInitResult sentinel = result;  // copy for comparison

  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for views.size() == 2 (< kMinViews = 3)";

  // result_out must be bit-for-bit unchanged.
  EXPECT_EQ((result.K - sentinel.K).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ(result.xi, sentinel.xi) << "result.xi must be unchanged when INVALID_INPUT is returned";
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

TEST(MEIOmni, RejectInsufficientPoints)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
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
  MEIInitResult result;
  result.K = K_sentinel;
  result.xi = 42.0;

  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT when a view has M = 3 < 4 points";
  EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ(result.xi, 42.0) << "result.xi must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Test 5: RejectImageSizeZero
//
// image_width = 0 -> INVALID_INPUT.
// image_height = -1 -> INVALID_INPUT.
// result_out unchanged in both cases.
// ===========================================================================

TEST(MEIOmni, RejectImageSizeZero)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
  const Eigen::Matrix2Xd board = makeCheckerboard(8, 6, 0.05);

  std::mt19937 dummy_rng(0U);
  std::vector<PlanarObservation> views = makeCanonicalViews(gt_model, board, 0.0, &dummy_rng);
  ASSERT_FALSE(views.empty());

  const Eigen::Matrix3d K_sentinel = Eigen::Matrix3d::Constant(55.0);

  // Case A: image_width = 0.
  {
    MEIInitResult result;
    result.K = K_sentinel;
    result.xi = 55.0;

    const StatusCode sc = estimateMEIInit(views, 0, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for image_width = 0";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when image_width = 0";
    EXPECT_EQ(result.xi, 55.0) << "result.xi must be unchanged when image_width = 0";
  }

  // Case B: image_height = -1.
  {
    MEIInitResult result;
    result.K = K_sentinel;
    result.xi = 55.0;

    const StatusCode sc = estimateMEIInit(views, 640, -1, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for image_height = -1";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when image_height = -1";
    EXPECT_EQ(result.xi, 55.0) << "result.xi must be unchanged when image_height = -1";
  }
}

// ===========================================================================
// Test 6: RejectColumnMismatch
//
// 3 views, one view has board_pts.cols() != image_pts.cols() -> INVALID_INPUT.
// result_out unchanged.
// ===========================================================================

TEST(MEIOmni, RejectColumnMismatch)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
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
  MEIInitResult result;
  result.K = K_sentinel;
  result.xi = 33.0;

  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for board_pts.cols() (12) != image_pts.cols() (10)";
  EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
    << "result.K must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ(result.xi, 33.0) << "result.xi must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Test 7: RejectNonFinite
//
// 3 views; one view has a NaN pixel (Case A), then separately an +inf pixel
// (Case B). Each case -> INVALID_INPUT; result_out unchanged.
// ===========================================================================

TEST(MEIOmni, RejectNonFinite)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
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

    MEIInitResult result;
    result.K = K_sentinel;
    result.xi = 22.0;

    const StatusCode sc = estimateMEIInit(views, 640, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
      << "Expected INVALID_INPUT when image_pts contains NaN";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when NaN triggers INVALID_INPUT";
    EXPECT_EQ(result.xi, 22.0) << "result.xi must be unchanged when NaN triggers INVALID_INPUT";
  }

  // Case B: +inf in image_pts of the third view.
  {
    PlanarObservation bad = good_third;  // copy good view
    bad.image_pts(1, 3) = std::numeric_limits<double>::infinity();

    std::vector<PlanarObservation> views = base_views;
    views.push_back(bad);

    MEIInitResult result;
    result.K = K_sentinel;
    result.xi = 22.0;

    const StatusCode sc = estimateMEIInit(views, 640, 480, result);
    EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
      << "Expected INVALID_INPUT when image_pts contains +inf";
    EXPECT_EQ((result.K - K_sentinel).norm(), 0.0)
      << "result.K must be unchanged when +inf triggers INVALID_INPUT";
    EXPECT_EQ(result.xi, 22.0) << "result.xi must be unchanged when +inf triggers INVALID_INPUT";
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

TEST(MEIOmni, SentinelPreserved)
{
  const CameraModel64 gt_model = buildGroundTruthMEIModel();
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
  MEIInitResult result;
  result.K = Eigen::Matrix3d::Constant(99.0);
  result.xi = 99.0;
  result.R_per_view = {Eigen::Matrix3d::Constant(99.0), Eigen::Matrix3d::Constant(99.0)};
  result.t_per_view = {Eigen::Vector3d::Constant(99.0), Eigen::Vector3d::Constant(99.0)};

  const MEIInitResult sentinel = result;  // copy before call

  const StatusCode sc = estimateMEIInit(views, 640, 480, result);
  ASSERT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for views.size() == 2 (< kMinViews = 3)";

  // Verify all fields are bit-for-bit unchanged.
  EXPECT_EQ((result.K - sentinel.K).norm(), 0.0)
    << "result.K must be unchanged after INVALID_INPUT";
  EXPECT_EQ(result.xi, sentinel.xi) << "result.xi must be unchanged after INVALID_INPUT";
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
