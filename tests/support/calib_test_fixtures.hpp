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

#ifndef CAMXIOM_TESTS_SUPPORT_CALIB_TEST_FIXTURES_HPP
#define CAMXIOM_TESTS_SUPPORT_CALIB_TEST_FIXTURES_HPP

// Shared synthetic-data helpers for the camxiom calibration / init tests (#8).
//
// These generators were previously copy-pasted verbatim across the model init
// and calibration test files (kb4 / mei / double_sphere / eucm / pinhole_zhang
// / pinhole_opencv / calib_intrinsics / integration). They are consolidated
// here unchanged (identical geometry and numerics) so there is a single source
// of truth; each test includes this header instead of redefining them.
//
// All helpers live in namespace camxiom::test. They are header-only `inline`
// definitions, so including this header in multiple test translation units is
// ODR-safe. Every consumer of buildCalibView is already a calibration test
// (Ceres-gated), so pulling in the (Ceres-free) calib header here is harmless.

#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

namespace camxiom::test
{

// ---------------------------------------------------------------------------
// Rotation matrix helpers (right-hand convention, angle in radians).
// ---------------------------------------------------------------------------

inline Eigen::Matrix3d rotMatX(double angle)
{
  const double ca = std::cos(angle);
  const double sa = std::sin(angle);
  Eigen::Matrix3d R;
  R << 1.0, 0.0, 0.0, 0.0, ca, -sa, 0.0, sa, ca;
  return R;
}

inline Eigen::Matrix3d rotMatY(double angle)
{
  const double ca = std::cos(angle);
  const double sa = std::sin(angle);
  Eigen::Matrix3d R;
  R << ca, 0.0, sa, 0.0, 1.0, 0.0, -sa, 0.0, ca;
  return R;
}

inline Eigen::Matrix3d rotMatZ(double angle)
{
  const double ca = std::cos(angle);
  const double sa = std::sin(angle);
  Eigen::Matrix3d R;
  R << ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0;
  return R;
}

// ---------------------------------------------------------------------------
// makeCheckerboard: 2 x (rows*cols) board-plane corner coordinates in
// row-major order (X increases left-to-right, Y top-to-bottom). Z = 0 is
// implicit for all points; origin at the (0,0) corner.
// ---------------------------------------------------------------------------

inline Eigen::Matrix2Xd makeCheckerboard(int rows, int cols, double spacing)
{
  const int n = rows * cols;
  Eigen::Matrix2Xd pts(2, n);
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      const int idx = r * cols + c;
      pts(0, idx) = c * spacing;
      pts(1, idx) = r * spacing;
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// makeCheckerboard3D: (rows*cols) 3D world points with Z = 0, row-major.
// The Vector3d counterpart of makeCheckerboard, used by the calibration tests.
// ---------------------------------------------------------------------------

inline std::vector<Eigen::Vector3d> makeCheckerboard3D(int rows, int cols, double spacing)
{
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      pts.emplace_back(c * spacing, r * spacing, 0.0);
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// buildCalibView: project Z=0 world points through a CameraModel64 with pose
// (R, t) and store in a CalibrationView. Optionally adds Gaussian noise.
// Returns false (and records a gtest failure) if any rayToPixel64 fails.
// ---------------------------------------------------------------------------

inline bool buildCalibView(
  const CameraModel64 &gt_model, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
  const std::vector<Eigen::Vector3d> &world_pts, double noise_sigma, std::mt19937 &rng,
  calib::CalibrationView &view_out
)
{
  const std::size_t m = world_pts.size();
  std::vector<Eigen::Vector2d> image_pts;
  image_pts.reserve(m);

  for (std::size_t j = 0; j < m; ++j)
  {
    const Eigen::Vector3d p_cam = R * world_pts[j] + t;
    const PixelResult64 pr = rayToPixel64(gt_model, p_cam);
    if (pr.status != StatusCode::OK)
    {
      ADD_FAILURE() << "rayToPixel64 failed at world point " << j
                    << " status=" << static_cast<int>(pr.status) << " p_cam=(" << p_cam.transpose()
                    << ")";
      return false;
    }
    image_pts.emplace_back(pr.pixel.u, pr.pixel.v);
  }

  if (noise_sigma > 0.0)
  {
    std::normal_distribution<double> noise(0.0, noise_sigma);
    for (auto &px : image_pts)
    {
      px.x() += noise(rng);
      px.y() += noise(rng);
    }
  }

  view_out.world_points = world_pts;
  view_out.image_points = image_pts;
  return true;
}

}  // namespace camxiom::test

#endif  // CAMXIOM_TESTS_SUPPORT_CALIB_TEST_FIXTURES_HPP
