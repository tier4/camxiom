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

// Unit tests for camxiom::init::estimatePoseDLT.
//
// Strategy: build a known ground-truth pose (R_gt, t_gt), project 3D world
// points through the camera model via rayToPixel64 to get synthetic pixel
// observations, call estimatePoseDLT, and assert the recovered pose matches
// the ground truth within tolerance.
//
// The cross-product DLT formulation is model-agnostic: the pixels are first
// lifted to unit bearings via pixelToRay64, so the downstream linear algebra
// is the same regardless of projection model.  The all-5-models smoke test
// exercises this model independence.

#include "camxiom/init/dlt_pnp.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"  // validateCameraModel64
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>
#include <string>

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
using camxiom::init::estimatePoseDLT;
using camxiom::init::estimatePoseRefined;

namespace
{

// ---------------------------------------------------------------------------
// Camera model builders (double-precision, mirroring projection_smoke_test.cpp)
// ---------------------------------------------------------------------------

/// Pinhole camera, no distortion.
/// fx = fy = 500, cx = 320, cy = 240 (realistic for a 640x480 sensor).
CameraModel64 makePinhole64()
{
  CameraModel64 m;
  m.intrinsics.fx = 500.0;
  m.intrinsics.fy = 500.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = camxiom::constants::kPi / 2.0;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

/// Fisheye (equidistant KB, zero higher-order coefficients).
/// Mirrors the float model in projection_smoke_test.cpp.
CameraModel64 makeFisheye64()
{
  CameraModel64 m;
  m.intrinsics.fx = 280.0;
  m.intrinsics.fy = 280.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = camxiom::constants::kPi - 1e-4;
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

/// Omnidirectional (Mei), xi = 1.
CameraModel64 makeOmni64()
{
  CameraModel64 m;
  m.intrinsics.fx = 400.0;
  m.intrinsics.fy = 400.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = camxiom::constants::kPi;
  m.projection.xi = 1.0;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

/// Double Sphere, xi = 0.5, alpha = 0.5.
CameraModel64 makeDoubleSphere64()
{
  CameraModel64 m;
  m.intrinsics.fx = 350.0;
  m.intrinsics.fy = 350.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = camxiom::constants::kPi;
  m.projection.xi = 0.5;
  m.projection.alpha = 0.5;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

/// EUCM, alpha = 0.6, beta = 1.1.
CameraModel64 makeEucm64()
{
  CameraModel64 m;
  m.intrinsics.fx = 350.0;
  m.intrinsics.fy = 350.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = camxiom::constants::kPi;
  m.projection.alpha = 0.6;
  m.projection.beta = 1.1;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

// ---------------------------------------------------------------------------
// Ground-truth pose generator
// ---------------------------------------------------------------------------

/// Returns the fixed ground-truth pose used in most tests.
/// R_gt = rotation of 0.3 rad around (1,1,1).normalized(), t_gt = (0.1,-0.2,1.5).
/// This places a small checkerboard at ~1.5 m depth, clearly in front of camera.
void makeGroundTruthPose(Eigen::Matrix3d &R_gt, Eigen::Vector3d &t_gt)
{
  const Eigen::AngleAxisd aa(0.3, Eigen::Vector3d(1.0, 1.0, 1.0).normalized());
  R_gt = aa.toRotationMatrix();
  t_gt = Eigen::Vector3d(0.1, -0.2, 1.5);
}

// ---------------------------------------------------------------------------
// Synthetic data generators
// ---------------------------------------------------------------------------

/// Generate N random 3D points in [-half_range, half_range]^3 (camera frame:
/// box in front of camera for these tests).
/// Seeded with the provided RNG.
Eigen::Matrix3Xd makeRandomWorld(std::mt19937 &rng, Eigen::Index n, double half_range = 0.2)
{
  std::uniform_real_distribution<double> dist(-half_range, half_range);
  Eigen::Matrix3Xd pts(3, n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    pts(0, i) = dist(rng);
    pts(1, i) = dist(rng);
    pts(2, i) = dist(rng);
  }
  return pts;
}

/// Generate a planar 8x6 checkerboard in the board (Z=0) plane with
/// 0.05 m spacing, centred at (0,0,0).
/// Returns a 3 x 48 matrix.
Eigen::Matrix3Xd makeCheckerboard()
{
  constexpr int rows = 6;
  constexpr int cols = 8;
  constexpr double spacing = 0.05;
  const double ox = -(cols - 1) * spacing / 2.0;
  const double oy = -(rows - 1) * spacing / 2.0;

  Eigen::Matrix3Xd pts(3, rows * cols);
  Eigen::Index k = 0;
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      pts(0, k) = ox + c * spacing;
      pts(1, k) = oy + r * spacing;
      pts(2, k) = 0.0;
      ++k;
    }
  }
  return pts;
}

/// Project world_pts to pixels via rayToPixel64 using R_gt, t_gt.
/// Returns false (and fills a diagnostic string) if any projection fails.
bool projectWorldToPixels(
  const CameraModel64 &model, const Eigen::Matrix3Xd &world_pts, const Eigen::Matrix3d &R_gt,
  const Eigen::Vector3d &t_gt, Eigen::Matrix2Xd &pixels_out, std::string &diag
)
{
  const Eigen::Index n = world_pts.cols();
  pixels_out.resize(2, n);

  for (Eigen::Index i = 0; i < n; ++i)
  {
    const Eigen::Vector3d p_cam = R_gt * world_pts.col(i) + t_gt;
    const PixelResult64 pr = camxiom::rayToPixel64(model, p_cam);
    if (pr.status != StatusCode::OK)
    {
      diag = "rayToPixel64 failed at column " + std::to_string(i) +
             " status=" + std::to_string(static_cast<int>(pr.status));
      return false;
    }
    pixels_out(0, i) = pr.pixel.u;
    pixels_out(1, i) = pr.pixel.v;
  }
  return true;
}

/// Add zero-mean Gaussian noise to a 2xN pixel matrix.
void addPixelNoise(std::mt19937 &rng, double sigma, Eigen::Matrix2Xd &pixels)
{
  std::normal_distribution<double> noise(0.0, sigma);
  for (Eigen::Index r = 0; r < pixels.rows(); ++r)
  {
    for (Eigen::Index c = 0; c < pixels.cols(); ++c)
    {
      pixels(r, c) += noise(rng);
    }
  }
}

/// Compute the geodesic angle (radians) between two rotation matrices.
/// angle(R1, R2) = ||log(R1^T R2)||  (Frobenius of log -> half-angle * 2)
double rotationAngle(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2)
{
  const Eigen::AngleAxisd aa(R1.transpose() * R2);
  return std::abs(aa.angle());
}

}  // namespace

// ===========================================================================
// Scenario 1: Pinhole noise-free recovery
//
// 10 random points in [-0.2, 0.2]^3.  K = diag(500,500), cx=320, cy=240,
// no distortion.  Ground-truth R/t are projected exactly via rayToPixel64,
// then estimatePoseDLT must recover them.
//
// Tolerance: |R - R_gt|_F < 1e-6, |t - t_gt| < 1e-6.
// Justification: exact DLT on Hartley-normalised data; no noise, so residual
// is purely floating-point arithmetic (~1e-13).  1e-6 gives a 7-order margin.
// ===========================================================================

TEST(DltPnp, PinholeNoNoiseRecovery)
{
  std::mt19937 rng(42U);

  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 10);

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for pinhole noise-free";

  // Rotation recovery: Frobenius distance to R_gt.
  // 1e-6: no noise, Hartley-normalised DLT; floating-point error is ~1e-12
  // on the design matrix, growing to ~1e-8 after Procrustes; 1e-6 is safe.
  EXPECT_NEAR((R_out - R_gt).norm(), 0.0, 1e-6) << "Pinhole noise-free: R recovery error too large";

  // Translation recovery: Euclidean distance.
  EXPECT_NEAR((t_out - t_gt).norm(), 0.0, 1e-6) << "Pinhole noise-free: t recovery error too large";

  // Post-condition: R_out must be a proper rotation.
  // |R^T R - I|_F < 1e-12 and det(R) > 0.99.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "R_out is not orthogonal";
  EXPECT_GT(R_out.determinant(), 0.99) << "R_out determinant is not close to +1";

  // Forward-projection round-trip for the first point: the pixel obtained
  // by projecting R_gt*X + t_gt should match the original pixel to ~1e-6 px
  // (noise-free, so both estimates are from the same ground truth).
  {
    const Eigen::Vector3d p_cam = R_out * world_pts.col(0) + t_out;
    const PixelResult64 pr = camxiom::rayToPixel64(model, p_cam);
    ASSERT_EQ(pr.status, StatusCode::OK) << "Round-trip rayToPixel64 failed";
    EXPECT_NEAR(pr.pixel.u, pixels(0, 0), 1e-6) << "Round-trip pixel u mismatch";
    EXPECT_NEAR(pr.pixel.v, pixels(1, 0), 1e-6) << "Round-trip pixel v mismatch";
  }
}

// ===========================================================================
// Scenario 2: Pinhole sub-pixel noise recovery
//
// N=30 random points, same model, add N(0, 0.5 px) Gaussian noise.
//
// Tolerance: angle(R, R_gt) < 0.01745 rad (1 deg), |t - t_gt|/|t_gt| < 0.05.
// Justification: 0.5 px at f=500 is ~1e-3 rad per bearing; with N=30
// overdetermination the DLT residual drops by ~1/sqrt(30) to ~2e-4 rad;
// 1 deg = 0.0175 rad is an 87x safety margin.
// Note: N=10 was insufficient (measured ~2 deg at 0.5 px; the linear DLT
// is less noise-robust than iterative PnP). N=30 is the minimum for which
// empirical testing confirmed sub-1-degree recovery at 0.5 px noise.
// ===========================================================================

TEST(DltPnp, PinholeSubPixelNoiseRecovery)
{
  std::mt19937 rng(42U);

  const CameraModel64 model = makePinhole64();
  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 30);  // N=30 for noise robustness

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  // Add N(0, 0.5 px) noise.
  addPixelNoise(rng, 0.5, pixels);

  Eigen::Matrix3d R_out;
  Eigen::Vector3d t_out;
  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for pinhole noisy case";

  // Rotation error < 1 degree (0.01745 rad): 0.5 px at f=500 gives ~1e-3 rad
  // per bearing; N=30 overdetermination reduces to ~2e-4 rad; 1 deg is 87x
  // safety margin.
  const double angle_err = rotationAngle(R_out, R_gt);
  EXPECT_LT(angle_err, 0.01745) << "Rotation error " << angle_err << " rad exceeds 1 degree";

  // Translation relative error < 5%: linear DLT on bearing rays; depth
  // uncertainty scales with bearing noise / baseline.
  const double t_rel_err = (t_out - t_gt).norm() / t_gt.norm();
  EXPECT_LT(t_rel_err, 0.05) << "Relative translation error " << t_rel_err << " exceeds 5%";

  // R_out still proper rotation.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
  EXPECT_GT(R_out.determinant(), 0.99);
}

// ===========================================================================
// Scenario 3: Planar checkerboard — noise-free recovery via the 9-DOF path.
//
// 8x6 board (48 points), 0.05 m spacing, centred at (0,0,0) in board frame
// (all Z=0 in world frame).  Same pinhole model and ground-truth pose.
//
// The algorithm detects planarity via eigenvalue ratio of the world-point
// covariance (all Z=0 => smallest eigenvalue ~ machine zero, ratio << 1e-6)
// and dispatches to the 9-DOF DLT path that avoids the rank deficiency of
// the 12-DOF formulation on Z=0 data.
//
// Tolerance: |R - R_gt|_F < 1e-6 and |t - t_gt| < 1e-6.
// Justification: exact data, Hartley-normalised 9-DOF DLT; floating-point
// residual is comparable to the non-planar case (~1e-12 on the design
// matrix).  1e-6 gives a 6-order margin.
// ===========================================================================

TEST(DltPnp, PlanarCheckerboardNoNoise)
{
  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeCheckerboard();  // 3 x 48, all Z=0

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Planar board (Z=0): expected OK via the 9-DOF planar DLT path";

  // Rotation recovery: Frobenius distance to R_gt.
  // 1e-6: no noise, 9-DOF Hartley-normalised DLT on 48 points.
  EXPECT_NEAR((R_out - R_gt).norm(), 0.0, 1e-6)
    << "Planar board no-noise: R recovery error too large";

  // Translation recovery: Euclidean distance.
  EXPECT_NEAR((t_out - t_gt).norm(), 0.0, 1e-6)
    << "Planar board no-noise: t recovery error too large";

  // Post-condition: R_out must be a proper rotation.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "R_out is not orthogonal";
  EXPECT_GT(R_out.determinant(), 0.99) << "R_out determinant is not close to +1";

  // Forward-projection round-trip for the first point: pixel obtained from
  // R_out*X + t_out must match the synthetic pixel within 1e-6 px.
  {
    const Eigen::Vector3d p_cam = R_out * world_pts.col(0) + t_out;
    const PixelResult64 pr = camxiom::rayToPixel64(model, p_cam);
    ASSERT_EQ(pr.status, StatusCode::OK) << "Round-trip rayToPixel64 failed";
    EXPECT_NEAR(pr.pixel.u, pixels(0, 0), 1e-6) << "Round-trip pixel u mismatch";
    EXPECT_NEAR(pr.pixel.v, pixels(1, 0), 1e-6) << "Round-trip pixel v mismatch";
  }
}

// ===========================================================================
// Scenario 3a: Planar checkerboard — sub-pixel noise recovery via 9-DOF path.
//
// Companion to PinholeSubPixelNoiseRecovery (Scenario 2). Calibration boards
// are the primary operational case for the 9-DOF planar path, so noise
// sensitivity here matters as much as for the 12-DOF path.
//
// Same K (diag(500,500), cx=320, cy=240), R_gt (AngleAxis 0.3 rad around
// (1,1,1).normalized()), t_gt (0.1, -0.2, 1.5) as PlanarCheckerboardNoNoise.
// 8x6 board at Z=0, 0.05 m spacing (48 corners), same as the noise-free case.
// N(0, 0.5 px) Gaussian noise added to image_pts via std::mt19937(42U).
//
// Tolerance: angle(R, R_gt) < 0.0873 rad (5 degrees), |t - t_gt|/|t_gt| < 0.05.
// Justification:
// (a) Single-view planar PnP has HIGHER rotation sensitivity than general 3D
//     PnP because rotation about in-board-plane axes is weakly observable: all
//     world points share Z=0, so the depth-rank along the board normal direction
//     is zero and those rotation components are reconstructed indirectly from
//     perspective foreshortening alone — which is poorly conditioned.
// (b) Empirical error at seed 42 is ~0.046 rad (2.6 deg); the 5-deg bound
//     (0.0873 rad) provides roughly a 2x safety margin over the measured value.
// (c) MS1-3 outputs are fed directly into MS2 Ceres nonlinear refinement.
//     2-3 deg initial rotation error is well within the convergence basin of
//     Levenberg-Marquardt (literature on planar PnP noise bounds confirms
//     this is the expected regime for DLT-based linear initialization).
// ===========================================================================

TEST(DltPnp, PlanarCheckerboardSubPixelNoiseRecovery)
{
  std::mt19937 rng(42U);

  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  // All 48 corners of the 8x6 board (Z=0 in world frame).
  const Eigen::Matrix3Xd world_pts = makeCheckerboard();  // 3 x 48, all Z=0

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  // Add N(0, 0.5 px) noise.
  addPixelNoise(rng, 0.5, pixels);

  Eigen::Matrix3d R_out;
  Eigen::Vector3d t_out;
  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Planar board noisy (9-DOF path): expected OK";

  // Rotation error < 5 degrees (0.0873 rad): single-view planar PnP is
  // geometrically weakly observable for rotations about in-board-plane axes
  // (all Z=0, so those components are constrained only via foreshortening).
  // Empirical error at seed 42 is ~0.046 rad (2.6 deg); 0.0873 rad is a
  // ~2x safety margin. This is the intended initial-guess regime for MS2
  // Ceres refinement, which converges from 2-3 deg without difficulty.
  const double angle_err = rotationAngle(R_out, R_gt);
  EXPECT_LT(angle_err, 0.0873) << "Planar board noisy: rotation error " << angle_err
                               << " rad exceeds 5 degrees (0.0873 rad)";

  // Relative translation error < 5%: same reasoning as Scenario 2.
  const double t_rel_err = (t_out - t_gt).norm() / t_gt.norm();
  EXPECT_LT(t_rel_err, 0.05) << "Planar board noisy: relative translation error " << t_rel_err
                             << " exceeds 5%";

  // R_out must still be a proper rotation.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "R_out is not orthogonal after noisy planar recovery";
  EXPECT_GT(R_out.determinant(), 0.99)
    << "R_out determinant is not close to +1 after noisy planar recovery";
}

// ===========================================================================
// Scenario 3d: Grazing incidence — bearings straddling 90 degrees (rz ~ 0),
// routine for >180-deg-FOV fisheyes with targets beside the camera. The old
// fixed (row0, row1) pair of [ray]_x degenerates there: row0 -> (0, 0, ry)
// and row1 -> (0, 0, -rx) become parallel and each point's constraint
// collapses towards rank 1, amplifying noise. The dominant-component row
// selection keeps the pair well-conditioned for every direction. Exercised
// for both the 12-DOF (non-planar cloud) and 9-DOF (planar board) paths.
// ===========================================================================

TEST(DltPnp, GrazingIncidenceRecovery)
{
  const CameraModel64 model = makeFisheye64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  // Build targets directly in the CAMERA frame beside the camera (bearing
  // z-components in roughly [-0.13, 0.23]) and map them back to world
  // coordinates, so the pixel observations really exercise rz ~ 0.
  const auto buildWorld = [&](bool planar, Eigen::Matrix3Xd &world_pts_out) {
    std::vector<Eigen::Vector3d> cam_pts;
    for (int ix = 0; ix < 4; ++ix)
    {
      for (int iy = 0; iy < 3; ++iy)
      {
        const double x = 0.55 + 0.05 * ix;
        const double y = -0.15 + 0.15 * iy;
        double z = -0.08 + 0.05 * ix + 0.02 * iy;
        if (!planar)
        {
          z += 0.04 * ((ix + iy) % 2);  // break coplanarity -> 12-DOF path
        }
        cam_pts.emplace_back(x, y, z);
      }
    }
    world_pts_out.resize(3, static_cast<Eigen::Index>(cam_pts.size()));
    for (std::size_t j = 0; j < cam_pts.size(); ++j)
    {
      world_pts_out.col(static_cast<Eigen::Index>(j)) = R_gt.transpose() * (cam_pts[j] - t_gt);
    }
  };

  for (const bool planar : {false, true})
  {
    Eigen::Matrix3Xd world_pts;
    buildWorld(planar, world_pts);

    Eigen::Matrix2Xd pixels;
    std::string diag;
    ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
      << "planar=" << planar << ": " << diag;

    Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
    Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);
    const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
    ASSERT_EQ(sc, StatusCode::OK) << "planar=" << planar;

    EXPECT_NEAR((R_out - R_gt).norm(), 0.0, 1e-6)
      << "planar=" << planar << ": R recovery error too large";
    EXPECT_NEAR((t_out - t_gt).norm(), 0.0, 1e-6)
      << "planar=" << planar << ": t recovery error too large";
  }
}

// ===========================================================================
// Scenario 3c: Tilted coplanar points — the 9-DOF path must handle planes
// that are NOT z = 0 in the world frame.
//
// Planarity detection uses covariance eigenvalues and is rotation-invariant,
// so board corners expressed in an arbitrary world frame (e.g. a fiducial
// map in vehicle coordinates) still dispatch to the 9-DOF path. That path
// used to drop the world z coordinate outright — Hartley normalisation only
// shifts and scales, it does not rotate the plane to z = 0 — so a tilted
// plane returned a systematically wrong pose with status OK. The fix solves
// in the plane's covariance eigenbasis and folds the basis change back into
// the recovered rotation.
//
// Construction: the SAME physical setup as PlanarCheckerboardNoNoise, only
// re-expressed in a world frame rotated 45 deg about Y and offset by
// (5, -2, 3). The pixels are identical, so the expected pose is known in
// closed form and the tolerances match the z = 0 planar test (1e-6).
// ===========================================================================

TEST(DltPnp, TiltedCoplanarPointsNoNoise)
{
  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt_board;
  Eigen::Vector3d t_gt_board;
  makeGroundTruthPose(R_gt_board, t_gt_board);

  const Eigen::Matrix3Xd board_pts = makeCheckerboard();  // 3 x 48, all Z=0

  // World frame: p_world = R_tilt * p_board + offset.
  const Eigen::Matrix3d R_tilt =
    Eigen::AngleAxisd(camxiom::constants::kPi / 4.0, Eigen::Vector3d(0.0, 1.0, 0.0))
      .toRotationMatrix();
  const Eigen::Vector3d offset(5.0, -2.0, 3.0);
  Eigen::Matrix3Xd world_pts = R_tilt * board_pts;
  world_pts.colwise() += offset;

  // p_cam = R_gt_board * p_board + t_gt_board
  //       = (R_gt_board * R_tilt^T) * p_world
  //         + (t_gt_board - R_gt_board * R_tilt^T * offset).
  const Eigen::Matrix3d R_expect = R_gt_board * R_tilt.transpose();
  const Eigen::Vector3d t_expect = t_gt_board - R_expect * offset;

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, board_pts, R_gt_board, t_gt_board, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Tilted coplanar set: expected OK via the 9-DOF planar DLT path";

  EXPECT_NEAR((R_out - R_expect).norm(), 0.0, 1e-6)
    << "Tilted coplanar set: R recovery error too large";
  EXPECT_NEAR((t_out - t_expect).norm(), 0.0, 1e-6)
    << "Tilted coplanar set: t recovery error too large";

  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "R_out is not orthogonal";
  EXPECT_GT(R_out.determinant(), 0.99) << "R_out determinant is not close to +1";
}

// ===========================================================================
// Scenario 3b: Near-planar input — verifies dispatch to 12-DOF path.
//
// World points are an 8x6 checkerboard with Z coordinates drawn from
// N(0, 0.001 m) (std = 1 mm).  With XY spread ~0.35 m x 0.25 m, the XY
// eigenvalues of the covariance are ~0.01–0.03 m^2.  The Z eigenvalue is
// ~(0.001)^2 = 1e-6 m^2, giving a ratio ~1e-6 / 0.01 = 1e-4 — well above
// kPlanarRatioThreshold (1e-6).  The planarity detector therefore does NOT
// classify these points as planar and the 12-DOF path is taken.
//
// Tolerance: |R - R_gt|_F < 1e-5, |t - t_gt| < 1e-4.
// Justification: Z variation of 1 mm is small relative to the 0.05 m spacing;
// the 12-DOF path is slightly less well-conditioned than the 9-DOF path for
// near-planar data (sigma(10)/sigma(0) is small but above the degeneracy
// threshold of 1e-12), so we allow a slightly looser tolerance than the exact
// planar case.  No noise on pixels, so residual comes purely from
// near-planarity conditioning; 1e-5/1e-4 gives a safe margin.
//
// --- Dispatch-verification commentary ---
//
// (1) This test verifies that the 12-DOF code path PRODUCES A CORRECT RESULT
//     on near-planar input. It is a happy-path correctness test for the
//     12-DOF formulation when given near-degenerate (but above-threshold)
//     geometry.
//
// (2) STRICT dispatch validation is provided by PlanarCheckerboardNoNoise:
//     exactly-planar input (wz=0 everywhere) is RANK-DEFICIENT in the 12-DOF
//     formulation because the entire third column of the design matrix
//     (wz-dependent entries) vanishes, causing sigma(10)/sigma(0) < 1e-12 and
//     a DEGENERATE_CONFIG return if the 12-DOF path were mistakenly taken.
//     Since PlanarCheckerboardNoNoise returns StatusCode::OK with low pose
//     error, dispatch to the 9-DOF path on exactly-planar input is provably
//     correct — an inverted dispatch would produce DEGENERATE_CONFIG, not OK.
//
// (3) The inline eigenvalue assertion in this test is a fixture sanity check:
//     it confirms that this test's input has a planarity ratio ABOVE the
//     internal kPlanarRatioThreshold (1e-6), i.e. these points ARE classified
//     as non-planar by the planarity detector. Combined with (2), this
//     triangulates dispatch correctness without any test-only API hooks.
// ===========================================================================

TEST(DltPnp, NearPlanarRoutesTo12DoF)
{
  // Seed chosen to produce well-conditioned near-planar data.
  std::mt19937 rng(123U);
  std::normal_distribution<double> z_noise(0.0, 0.001);  // 1 mm std in Z

  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  // Build an 8x6 checkerboard with small Z perturbations.
  constexpr int rows = 6;
  constexpr int cols = 8;
  constexpr double spacing = 0.05;
  const double ox = -(cols - 1) * spacing / 2.0;
  const double oy = -(rows - 1) * spacing / 2.0;

  constexpr int n_pts = rows * cols;
  Eigen::Matrix3Xd world_pts(3, n_pts);
  Eigen::Index k = 0;
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      world_pts(0, k) = ox + c * spacing;
      world_pts(1, k) = oy + r * spacing;
      world_pts(2, k) = z_noise(rng);  // N(0, 0.001 m): small but non-zero
      ++k;
    }
  }

  // Verify the planarity ratio is above the threshold (-> 12-DOF path).
  // The covariance eigenvalue ratio for Z-spread << XY-spread must be
  // larger than kPlanarRatioThreshold = 1e-6.
  {
    const Eigen::Vector3d centroid = world_pts.rowwise().mean();
    const Eigen::Matrix3Xd shifted = world_pts.colwise() - centroid;
    const Eigen::Matrix3d cov = (shifted * shifted.transpose()) / static_cast<double>(n_pts);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    ASSERT_EQ(eig.info(), Eigen::Success);
    const Eigen::Vector3d evals = eig.eigenvalues();
    ASSERT_GT(evals(2), 0.0) << "Largest eigenvalue must be positive";
    const double ratio = evals(0) / evals(2);
    // Ratio ~1e-4 (Z-spread^2 / XY-spread^2), well above 1e-6 threshold.
    EXPECT_GT(ratio, 1e-6) << "Near-planar data should NOT trigger the planar path; ratio="
                           << ratio;
  }

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Near-planar (12-DOF path): expected OK";

  // 1e-5: no pixel noise; slight loss of conditioning from near-planarity
  // compared to the fully 3D case (Scenario 1), but still well within 1e-5.
  EXPECT_NEAR((R_out - R_gt).norm(), 0.0, 1e-5) << "Near-planar 12-DOF: R recovery error too large";

  // 1e-4: translation is slightly more sensitive to near-planar conditioning.
  EXPECT_NEAR((t_out - t_gt).norm(), 0.0, 1e-4) << "Near-planar 12-DOF: t recovery error too large";

  // R must still be a proper rotation.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "R_out is not orthogonal";
  EXPECT_GT(R_out.determinant(), 0.99) << "R_out determinant is not close to +1";
}

// ===========================================================================
// Scenario 3c: Collinear world points — planarity detector fires, then the
// 9-DOF path trips its own rank check.
//
// 10 world points along the X-axis: (i, 0, 0) for i = 0..9.
// Covariance: only the X eigenvalue is non-zero; ratio Y/X = 0 < 1e-6 ->
// is_planar = true -> 9-DOF path is taken.  In the 9-DOF design matrix, wy=0
// for all points, so columns 3, 4, 5 (the r2·wy / t2·wy terms) are all zero.
// This reduces the matrix rank to ≤ 6 (of 9 needed), causing sigma(7)/sigma(0)
// < 1e-12 -> DEGENERATE_CONFIG.
//
// Expected: StatusCode::DEGENERATE_CONFIG (not a crash).
// ===========================================================================

TEST(DltPnp, CollinearWorldPointsDegenerate)
{
  const CameraModel64 model = makePinhole64();
  ASSERT_EQ(camxiom::validateCameraModel64(model), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  // 10 world points along the X-axis, 0.1 m spacing. Y=Z=0 for all.
  constexpr int n_col = 10;
  Eigen::Matrix3Xd world_pts(3, n_col);
  for (int i = 0; i < n_col; ++i)
  {
    world_pts(0, i) = static_cast<double>(i) * 0.1 - 0.45;  // centred
    world_pts(1, i) = 0.0;
    world_pts(2, i) = 0.0;
  }

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(88.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(88.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Collinear world points: planarity detector fires (Y/Z spread = 0), "
       "then 9-DOF path trips rank check (wy=0 zeroes out r2 columns)";

  // Out-params must be unchanged on non-OK return.
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when DEGENERATE_CONFIG is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ===========================================================================
// Scenario 4: All-5-models smoke test, noise-free (N=20)
//
// For each of pinhole / fisheye-equidistant / omni-MEI / double_sphere / EUCM,
// build the model, project 20 random 3D points, recover pose, assert match.
//
// Tolerance: |R - R_gt|_F < 1e-4, |t - t_gt| < 1e-4.
// Justification: the DLT linear system is the same for all models once pixels
// are lifted to bearings.  Tighter tolerance than 1e-6 is not warranted here
// because pixelToRay64 inverse solvers for non-pinhole models have finite
// Newton convergence (typically ~1e-12 residual after 15 iterations), but the
// tolerance through the full chain (rayToPixel -> pixelToRay -> DLT) lands
// around 1e-9 for well-conditioned cases.  1e-4 gives a comfortable 5-order
// margin against any per-model inverse solver variation.
//
// If a model's pixelToRay64 converges less tightly (an MS0 numerical limit,
// not an MS1-3 bug), the per-model tolerance may be relaxed with a comment.
// ===========================================================================

struct ModelTestCase
{
  std::string name;
  CameraModel64 model;
};

class DltPnpAllModels : public ::testing::TestWithParam<ModelTestCase>
{
};

TEST_P(DltPnpAllModels, NoNoiseRecovery)
{
  const ModelTestCase &tc = GetParam();
  ASSERT_EQ(camxiom::validateCameraModel64(tc.model), StatusCode::OK)
    << "Model " << tc.name << " failed validateCameraModel64";

  std::mt19937 rng(42U);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 20);

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(tc.model, world_pts, R_gt, t_gt, pixels, diag))
    << "Model " << tc.name << ": projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);

  const StatusCode sc = estimatePoseDLT(tc.model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Model " << tc.name << ": estimatePoseDLT returned non-OK";

  // Tolerance 1e-4 across all models:
  // - Pinhole: expect ~1e-9 (near machine epsilon through DLT)
  // - Fisheye/Omni/DS/EUCM: pixelToRay64 inverse converges to ~1e-12
  //   residual; end-to-end chain stays well below 1e-4.
  // If a non-pinhole model requires a looser tolerance it would be split into
  // a separate TEST() with an explanatory comment (none needed so far).
  const double R_err = (R_out - R_gt).norm();
  EXPECT_NEAR(R_err, 0.0, 1e-4) << "Model " << tc.name << ": R recovery error " << R_err;

  const double t_err = (t_out - t_gt).norm();
  EXPECT_NEAR(t_err, 0.0, 1e-4) << "Model " << tc.name << ": t recovery error " << t_err;

  // Proper rotation.
  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12)
    << "Model " << tc.name << ": R_out is not orthogonal";
  EXPECT_GT(R_out.determinant(), 0.99)
    << "Model " << tc.name << ": R_out determinant is not close to +1";

  // Round-trip for first point.
  {
    const Eigen::Vector3d p_cam = R_out * world_pts.col(0) + t_out;
    const PixelResult64 pr = camxiom::rayToPixel64(tc.model, p_cam);
    ASSERT_EQ(pr.status, StatusCode::OK)
      << "Model " << tc.name << ": round-trip rayToPixel64 failed";
    // 1e-6 px tolerance: round-trip should reproduce the original pixel
    // (noise-free) to near machine precision.
    EXPECT_NEAR(pr.pixel.u, pixels(0, 0), 1e-6)
      << "Model " << tc.name << ": round-trip pixel u mismatch";
    EXPECT_NEAR(pr.pixel.v, pixels(1, 0), 1e-6)
      << "Model " << tc.name << ": round-trip pixel v mismatch";
  }
}

INSTANTIATE_TEST_SUITE_P(
  AllModelsNoNoise, DltPnpAllModels,
  ::testing::Values(
    ModelTestCase{"Pinhole", makePinhole64()}, ModelTestCase{"FisheyeKB4", makeFisheye64()},
    ModelTestCase{"OmniMEI", makeOmni64()}, ModelTestCase{"DoubleSphere", makeDoubleSphere64()},
    ModelTestCase{"EUCM", makeEucm64()}
  ),
  [](const ::testing::TestParamInfo<ModelTestCase> &info) { return info.param.name; }
);

// ===========================================================================
// Scenario 5a: N < 6 -> INVALID_INPUT; R_out / t_out unchanged.
// ===========================================================================

TEST(DltPnp, RejectTooFewPoints)
{
  const CameraModel64 model = makePinhole64();
  std::mt19937 rng(42U);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 5);  // N = 5 < 6

  Eigen::Matrix2Xd pixels(2, 5);
  pixels.setRandom();

  // Sentinel values: must survive unchanged on non-OK return.
  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(77.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(77.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for N=5 (< 6)";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Scenario 5b: Column mismatch -> INVALID_INPUT.
// world_pts has 8 columns, image_pts has 7 columns.
// ===========================================================================

TEST(DltPnp, RejectColumnMismatch)
{
  const CameraModel64 model = makePinhole64();
  std::mt19937 rng(42U);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 8);  // 8 columns
  Eigen::Matrix2Xd pixels(2, 7);                               // 7 columns
  pixels.setZero();

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(55.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(55.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for mismatched column counts (8 vs 7)";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Scenario 5c: NaN in world_pts -> INVALID_INPUT.
// ===========================================================================

TEST(DltPnp, RejectNaNInWorldPts)
{
  const CameraModel64 model = makePinhole64();
  std::mt19937 rng(42U);

  Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 8);
  world_pts(1, 3) = std::numeric_limits<double>::quiet_NaN();

  Eigen::Matrix2Xd pixels(2, 8);
  pixels.setZero();

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(55.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(55.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when world_pts contains NaN";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Scenario 5d: Inf in image_pts -> INVALID_INPUT.
// ===========================================================================

TEST(DltPnp, RejectInfInImagePts)
{
  const CameraModel64 model = makePinhole64();
  std::mt19937 rng(42U);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 8);
  Eigen::Matrix2Xd pixels(2, 8);
  pixels.setZero();
  pixels(0, 2) = std::numeric_limits<double>::infinity();

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(55.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(55.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when image_pts contains +inf";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when INVALID_INPUT is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when INVALID_INPUT is returned";
}

// ===========================================================================
// Scenario 5e: All world_pts identical -> DEGENERATE_CONFIG.
// The Hartley normalisation step finds mean_dist == 0 and returns
// DEGENERATE_CONFIG before the SVD is even attempted.
// ===========================================================================

TEST(DltPnp, RejectCoincidentWorldPts)
{
  const CameraModel64 model = makePinhole64();

  // All world points at the same location.
  constexpr Eigen::Index n = 8;
  Eigen::Matrix3Xd world_pts(3, n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    world_pts.col(i) = Eigen::Vector3d(0.1, 0.2, 0.3);
  }

  // Use arbitrary but finite pixel coordinates (content doesn't matter
  // since the degeneracy is in world_pts, detected before pixelToRay64).
  Eigen::Matrix2Xd pixels(2, n);
  pixels.setZero();

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(33.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(33.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG for all-coincident world points";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when DEGENERATE_CONFIG is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ===========================================================================
// Scenario 5f: pixelToRay64 failure -> DEGENERATE_CONFIG.
//
// Use a CameraModel64 with type UNKNOWN.  validateCameraModel64 would reject
// it, but estimatePoseDLT calls pixelToRay64 directly which returns
// INVALID_MODEL for UNKNOWN type.  The algorithm must surface this as
// DEGENERATE_CONFIG (any non-OK pixelToRay64 result is treated that way).
// ===========================================================================

TEST(DltPnp, RejectPixelToRayFailure)
{
  // Build an UNKNOWN-type model64 — pixelToRay64 will return INVALID_MODEL.
  CameraModel64 bad_model;
  bad_model.projection.type = ProjectionModelType::UNKNOWN;
  // Fill minimally plausible intrinsics (doesn't matter; dispatch fails first).
  bad_model.intrinsics.fx = 500.0;
  bad_model.intrinsics.fy = 500.0;
  bad_model.intrinsics.cx = 320.0;
  bad_model.intrinsics.cy = 240.0;

  std::mt19937 rng(42U);
  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 8);

  // Arbitrary valid-looking pixels (anything works; the model dispatch fails).
  Eigen::Matrix2Xd pixels(2, 8);
  pixels.row(0).setConstant(320.0);
  pixels.row(1).setConstant(240.0);

  Eigen::Matrix3d R_sentinel = Eigen::Matrix3d::Constant(11.0);
  Eigen::Vector3d t_sentinel = Eigen::Vector3d::Constant(11.0);
  Eigen::Matrix3d R_out = R_sentinel;
  Eigen::Vector3d t_out = t_sentinel;

  const StatusCode sc = estimatePoseDLT(bad_model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG when pixelToRay64 returns non-OK";
  EXPECT_EQ((R_out - R_sentinel).norm(), 0.0)
    << "R_out must be unchanged when DEGENERATE_CONFIG is returned";
  EXPECT_EQ((t_out - t_sentinel).norm(), 0.0)
    << "t_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ===========================================================================
// Extra: N = exactly 6 (minimum) noise-free — boundary of the valid domain.
//
// The 2*6 = 12 design matrix is exactly square (12x12), leaving a 1D null
// space only if sigma(11) is truly zero.  With exact data the algorithm must
// return OK and recover the pose.
//
// Tolerance: same as Scenario 1 (1e-6).
// ===========================================================================

TEST(DltPnp, MinimumSixPointsNoNoise)
{
  std::mt19937 rng(42U);

  const CameraModel64 model = makePinhole64();
  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 6);  // exactly N=6

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag))
    << "Projection failed: " << diag;

  Eigen::Matrix3d R_out;
  Eigen::Vector3d t_out;
  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for exactly N=6 noise-free points";

  // 1e-6: as for Scenario 1 — exact data, Hartley-normalised DLT.
  EXPECT_NEAR((R_out - R_gt).norm(), 0.0, 1e-6) << "N=6 noise-free: R recovery error too large";
  EXPECT_NEAR((t_out - t_gt).norm(), 0.0, 1e-6) << "N=6 noise-free: t recovery error too large";

  EXPECT_NEAR((R_out.transpose() * R_out - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
  EXPECT_GT(R_out.determinant(), 0.99);
}

// ===========================================================================
// Extra: N = 5 specifically (one below minimum) — INVALID_INPUT.
// Distinct from RejectTooFewPoints (which uses N=5 implicitly); this
// documents the exact boundary.
// ===========================================================================

TEST(DltPnp, ExactlyFivePointsIsInvalid)
{
  const CameraModel64 model = makePinhole64();
  std::mt19937 rng(99U);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);

  const Eigen::Matrix3Xd world_pts = makeRandomWorld(rng, 5);

  Eigen::Matrix2Xd pixels;
  std::string diag;
  // Project with a *valid* model so we rule out projection issues.
  const bool proj_ok = projectWorldToPixels(model, world_pts, R_gt, t_gt, pixels, diag);
  // N=5 is valid for the projector, so projection should succeed.
  ASSERT_TRUE(proj_ok) << "Projection failed: " << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_out = Eigen::Vector3d::Zero();

  const StatusCode sc = estimatePoseDLT(model, world_pts, pixels, R_out, t_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "N=5 must be INVALID_INPUT (minimum is 6)";
}

// ===========================================================================
// estimatePoseRefined: DLT initialisation + non-linear pose-only refinement
// (the OpenCV-free analogue of cv::solvePnP(SOLVEPNP_ITERATIVE)).
//
// estimatePoseRefined takes a float CameraModel (what the solver and callers
// hold); the helpers below build float models and widen them via
// toCameraModel64 for projection / the raw-DLT baseline so both paths use the
// same numbers.
// ===========================================================================

namespace
{

/// Float pinhole, K = diag(500,500), cx=320, cy=240, no distortion.
camxiom::CameraModel makePinholeFloat()
{
  camxiom::CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi / 2.0);
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

/// Float fisheye (equidistant KB, zero higher-order coefficients).
camxiom::CameraModel makeFisheyeFloat()
{
  camxiom::CameraModel m;
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

/// Global reprojection RMS (px) of a pose against observed pixels, under the
/// given model. Infinity if any point fails to project.
double reprojRms(
  const CameraModel64 &model, const Eigen::Matrix3Xd &world, const Eigen::Matrix2Xd &obs,
  const Eigen::Matrix3d &R, const Eigen::Vector3d &t
)
{
  double sum_sq = 0.0;
  const Eigen::Index n = world.cols();
  for (Eigen::Index i = 0; i < n; ++i)
  {
    const Eigen::Vector3d p_cam = R * world.col(i) + t;
    const PixelResult64 pr = camxiom::rayToPixel64(model, p_cam);
    if (pr.status != StatusCode::OK)
    {
      return std::numeric_limits<double>::infinity();
    }
    const double du = pr.pixel.u - obs(0, i);
    const double dv = pr.pixel.v - obs(1, i);
    sum_sq += du * du + dv * dv;
  }
  return std::sqrt(sum_sq / static_cast<double>(n));
}

}  // namespace

// Noise-free planar board: refinement converges to the exact ground truth.
TEST(PoseRefined, PinholePlanarNoNoiseRecovery)
{
  const camxiom::CameraModel mf = makePinholeFloat();
  const CameraModel64 m64 = camxiom::toCameraModel64(mf);
  ASSERT_EQ(camxiom::validateCameraModel64(m64), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);
  const Eigen::Matrix3Xd world = makeCheckerboard();

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(m64, world, R_gt, t_gt, pixels, diag)) << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);
  const StatusCode sc = estimatePoseRefined(mf, world, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK);

  // No noise: the geometric optimum is the exact pose. 1e-4 leaves a wide
  // margin over the float-model widening + Gauss-Newton convergence floor.
  EXPECT_LT(rotationAngle(R_out, R_gt), 1e-4);
  EXPECT_LT((t_out - t_gt).norm(), 1e-4);
}

// Noisy planar board: refinement minimises the geometric reprojection error
// from the DLT seed, so it is never worse and -- because the linear DLT pose
// is biased on a noisy planar board -- strictly better.
TEST(PoseRefined, PlanarNoiseRefineImprovesReproj)
{
  std::mt19937 rng(7U);
  const camxiom::CameraModel mf = makePinholeFloat();
  const CameraModel64 m64 = camxiom::toCameraModel64(mf);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);
  const Eigen::Matrix3Xd world = makeCheckerboard();

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(m64, world, R_gt, t_gt, pixels, diag)) << diag;
  addPixelNoise(rng, 0.5, pixels);

  Eigen::Matrix3d R_dlt;
  Eigen::Vector3d t_dlt;
  ASSERT_EQ(estimatePoseDLT(m64, world, pixels, R_dlt, t_dlt), StatusCode::OK);

  Eigen::Matrix3d R_ref;
  Eigen::Vector3d t_ref;
  ASSERT_EQ(estimatePoseRefined(mf, world, pixels, R_ref, t_ref), StatusCode::OK);

  const double rms_dlt = reprojRms(m64, world, pixels, R_dlt, t_dlt);
  const double rms_ref = reprojRms(m64, world, pixels, R_ref, t_ref);

  // Never worse (epsilon absorbs floating-point noise).
  EXPECT_LE(rms_ref, rms_dlt + 1e-6)
    << "refined RMS must not exceed DLT (dlt=" << rms_dlt << " ref=" << rms_ref << ")";
  // And strictly better on a noisy planar board, where the algebraic DLT pose
  // does not minimise the geometric residual.
  EXPECT_LT(rms_ref, rms_dlt) << "refined RMS should improve on DLT (dlt=" << rms_dlt
                              << " ref=" << rms_ref << ")";
}

// Model-agnostic: refinement works for a non-pinhole projection.
TEST(PoseRefined, FisheyeNoNoiseRecovery)
{
  const camxiom::CameraModel mf = makeFisheyeFloat();
  const CameraModel64 m64 = camxiom::toCameraModel64(mf);
  ASSERT_EQ(camxiom::validateCameraModel64(m64), StatusCode::OK);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);
  const Eigen::Matrix3Xd world = makeCheckerboard();

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(m64, world, R_gt, t_gt, pixels, diag)) << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);
  const StatusCode sc = estimatePoseRefined(mf, world, pixels, R_out, t_out);
  ASSERT_EQ(sc, StatusCode::OK);

  EXPECT_LT(rotationAngle(R_out, R_gt), 1e-3);
  EXPECT_LT((t_out - t_gt).norm(), 1e-3);
}

// Fewer than 6 correspondences: the DLT prerequisite fails, the status is
// propagated verbatim and the outputs are left untouched.
TEST(PoseRefined, TooFewPointsPropagatesAndLeavesOutputs)
{
  const camxiom::CameraModel mf = makePinholeFloat();
  const CameraModel64 m64 = camxiom::toCameraModel64(mf);

  Eigen::Matrix3d R_gt;
  Eigen::Vector3d t_gt;
  makeGroundTruthPose(R_gt, t_gt);
  const Eigen::Matrix3Xd world = makeCheckerboard().leftCols(5);

  Eigen::Matrix2Xd pixels;
  std::string diag;
  ASSERT_TRUE(projectWorldToPixels(m64, world, R_gt, t_gt, pixels, diag)) << diag;

  Eigen::Matrix3d R_out = Eigen::Matrix3d::Constant(99.0);
  Eigen::Vector3d t_out = Eigen::Vector3d::Constant(99.0);
  const StatusCode sc = estimatePoseRefined(mf, world, pixels, R_out, t_out);

  EXPECT_EQ(sc, StatusCode::INVALID_INPUT);
  EXPECT_TRUE((R_out.array() == 99.0).all());
  EXPECT_TRUE((t_out.array() == 99.0).all());
}
