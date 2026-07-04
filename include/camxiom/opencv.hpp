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

#ifndef CAMXIOM__OPENCV_HPP
#define CAMXIOM__OPENCV_HPP

// Optional OpenCV interoperability layer.
//
// This header and its translation units (src/opencv/*.cpp) are only compiled
// into the camxiom library when OpenCV is found at build time (the CMake build
// defines CAMXIOM_HAS_OPENCV — a PUBLIC definition, so it propagates to CMake
// consumers through the exported targets — and adds the sources). The guard
// below therefore requires BOTH the configure-time definition and the OpenCV
// headers: __has_include alone would expose declarations with no compiled
// implementation whenever the library was built without OpenCV on a machine
// that has the OpenCV headers, turning every use into a link error. Without
// either, the header harmlessly expands to nothing. Non-CMake consumers of an
// OpenCV-enabled build must define CAMXIOM_HAS_OPENCV themselves.
//
// The API mirrors OpenCV's cv::* / cv::fisheye::* / cv::omnidir::* helpers so
// it can act as a drop-in replacement, but every model is handled by the unified
// camxiom core (no OpenCV math is used for projection/undistortion).

#if defined(CAMXIOM_HAS_OPENCV) && __has_include(<opencv2/core.hpp>)

#include "camxiom/remap.hpp"
#include "camxiom/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace camxiom::opencv
{

using camxiom::CameraModel;
using camxiom::ImageRemapOptions;
using camxiom::ImageRemapResult;
using camxiom::ImageSize;
using camxiom::RectifiedOutputModelOptions;
using camxiom::RectifyRemapResult;
using camxiom::RemapResult;
using camxiom::SolverOptions;

// ---------------------------------------------------------------------------
// Intrinsics initialization from planar calibration patterns
// ---------------------------------------------------------------------------

/// Estimate initial pinhole intrinsics from multiple views of a planar pattern.
/// Equivalent to cv::initCameraMatrix2D: uses homography decomposition to
/// estimate focal length, with principal point at image center.
/// Returns a distortion-free pinhole CameraModel.
/// @return true on success (>= 1 view, valid homography).
[[nodiscard]] bool initCameraMatrix2D(
  const std::vector<std::vector<cv::Point3f>> &object_points_per_view,
  const std::vector<std::vector<cv::Point2f>> &image_points_per_view, const cv::Size &image_size,
  CameraModel &camera_model_out
);

// ---------------------------------------------------------------------------
// cv::Mat remap map generation
// ---------------------------------------------------------------------------

RemapResult buildRemapMapCV(
  const CameraModel &src_model, const CameraModel &dst_model, int width, int height, cv::Mat &map1,
  cv::Mat &map2, const SolverOptions &solver_options = SolverOptions{}
);

RemapResult buildUndistortRemapMapCV(
  const CameraModel &src_model, int width, int height, cv::Mat &map1, cv::Mat &map2,
  const SolverOptions &solver_options = SolverOptions{}
);

/// Build rectification remap maps (pinhole output, no black borders).
RectifyRemapResult buildRectifyRemapMapCV(
  const CameraModel &src_model, ImageSize src_size, const RectifiedOutputModelOptions &options,
  cv::Mat &map1, cv::Mat &map2
);

// ---------------------------------------------------------------------------
// Remap cache — compute once, apply many times
// ---------------------------------------------------------------------------

/// Thread safety: one writer, then many readers. The build*() methods and
/// clear() mutate the cached maps and must not run concurrently with anything
/// else on the same instance; once a build has returned, concurrent apply()
/// calls (const) targeting DISTINCT output Mats are safe. (Details:
/// docs/design/thread-safety.md.)
class RemapCache
{
public:
  RemapCache() = default;

  /// Build and cache the remap maps.
  RemapResult build(
    const CameraModel &src_model, const CameraModel &dst_model, int width, int height,
    const SolverOptions &solver_options = SolverOptions{}
  );

  /// Build and cache undistortion remap maps (same-projection, distortion only removed).
  RemapResult buildUndistort(
    const CameraModel &src_model, int width, int height,
    const SolverOptions &solver_options = SolverOptions{}
  );

  /// Build and cache rectification remap maps (pinhole output).
  RectifyRemapResult buildRectify(
    const CameraModel &src_model, ImageSize src_size,
    const RectifiedOutputModelOptions &options = RectifiedOutputModelOptions{}
  );

  /// Apply the cached remap to an image.
  [[nodiscard]] bool apply(
    const cv::Mat &src, cv::Mat &dst, int interpolation = 1, int border_mode = 0,
    const cv::Scalar &border_value = cv::Scalar()
  ) const;

  [[nodiscard]] bool isValid() const;
  int width() const;
  int height() const;
  const cv::Mat &map1() const;
  const cv::Mat &map2() const;
  void clear();

private:
  cv::Mat map1_{};
  cv::Mat map2_{};
  int width_{0};
  int height_{0};
  bool valid_{false};
};

// ---------------------------------------------------------------------------
// Convenience: one-shot image remap / undistort / rectify
// ---------------------------------------------------------------------------

/// Same-projection undistort: removes distortion but keeps the projection model.
[[nodiscard]] bool undistortImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

/// Remap an image from src_model to an explicit dst_model.
[[nodiscard]] bool remapImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, const CameraModel &dst_model,
  int interpolation = 1, const SolverOptions &solver_options = SolverOptions{}
);

/// Rectify an image to a pinhole perspective output.
[[nodiscard]] bool rectifyImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model,
  const RectifiedOutputModelOptions &options = RectifiedOutputModelOptions{},
  CameraModel *output_model_out = nullptr, int interpolation = 1
);

/// Distort (re-distort) an image: remap from the (typically undistorted)
/// src_model into the distorted dst_model. Parameter order matches
/// remapImage (src_model before dst_model); both parameters are CameraModel,
/// so a swapped call compiles but silently builds the inverse mapping.
[[nodiscard]] bool distortImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, const CameraModel &dst_model,
  int interpolation = 1, const SolverOptions &solver_options = SolverOptions{}
);

// ---------------------------------------------------------------------------
// Point projection / undistortion — CameraModel-based (batch SIMD/OpenMP)
// ---------------------------------------------------------------------------

/// Project 3D object points to 2D image points using extrinsics (rvec, tvec).
/// @return Number of successfully projected points, or -1 on invalid input/model.
[[nodiscard]] int projectPoints(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec,
  const cv::Vec3d &tvec, std::vector<cv::Point2f> &image_points_out
);

/// Undistort pixel coordinates to normalized image coordinates.
/// @return Number of successfully undistorted points, or -1 on invalid input/model.
[[nodiscard]] int undistortPoints(
  const CameraModel &model, const std::vector<cv::Point2f> &src_points,
  std::vector<cv::Point2f> &dst_points_out, const cv::Mat &R = cv::Mat(),
  const cv::Mat &P = cv::Mat(), const SolverOptions &opts = SolverOptions{}
);

/// Apply distortion to normalized image coordinates.
/// @return Number of successfully distorted points, or -1 on invalid input/model.
[[nodiscard]] int distortPoints(
  const CameraModel &model, const std::vector<cv::Point2f> &src_points,
  std::vector<cv::Point2f> &dst_points_out
);

// ---------------------------------------------------------------------------
// OpenCV-compatible APIs — pinhole namespace
// ---------------------------------------------------------------------------

namespace pinhole
{

[[nodiscard]] int projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const cv::Mat &R = cv::Mat(), const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &R = cv::Mat(),
  const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs
);

[[nodiscard]] int distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs
);

[[nodiscard]] bool undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

[[nodiscard]] bool distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

}  // namespace pinhole

// ---------------------------------------------------------------------------
// OpenCV-compatible APIs — fisheye namespace
// ---------------------------------------------------------------------------

namespace fisheye
{

[[nodiscard]] int projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const cv::Mat &R = cv::Mat(), const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &R = cv::Mat(),
  const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs
);

[[nodiscard]] int distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs
);

[[nodiscard]] bool undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

[[nodiscard]] bool distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

}  // namespace fisheye

// ---------------------------------------------------------------------------
// OpenCV-compatible APIs — omnidirectional namespace
// ---------------------------------------------------------------------------

namespace omnidirectional
{

[[nodiscard]] int projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, double xi,
  std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, double xi,
  std::vector<cv::Point2f> &image_points
);

[[nodiscard]] int undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, double xi, const cv::Mat &R = cv::Mat(), const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, double xi, const cv::Mat &R = cv::Mat(),
  const cv::Mat &P = cv::Mat()
);

[[nodiscard]] int distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, double xi
);

[[nodiscard]] int distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, double xi
);

[[nodiscard]] bool undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  double xi, const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

[[nodiscard]] bool distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  double xi, const cv::Mat &new_camera_matrix = cv::Mat(), int interpolation = 1,
  const SolverOptions &solver_options = SolverOptions{}
);

}  // namespace omnidirectional

}  // namespace camxiom::opencv

#endif  // CAMXIOM_HAS_OPENCV && __has_include(<opencv2/core.hpp>)

#endif  // CAMXIOM__OPENCV_HPP
