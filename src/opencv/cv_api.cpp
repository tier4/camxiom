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

#include "camxiom/opencv.hpp"

#if __has_include(<opencv2/core.hpp>)

#include "camxiom/model.hpp"
#include "opencv/internal.hpp"

#include <vector>

namespace camgeom_opencv = camxiom::opencv;

namespace camxiom::opencv
{

// ---------------------------------------------------------------------------
// pinhole namespace
// ---------------------------------------------------------------------------

int pinhole::projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point3f> pts;
  if (!extractPoints3f(object_points, pts))
  {
    return -1;
  }
  cv::Vec3d r;
  if (!extractVec3d(rvec, r))
  {
    return -1;
  }
  cv::Vec3d t;
  if (!extractVec3d(tvec, t))
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, pts, r, t, image_points);
}

int pinhole::projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, object_points, rvec, tvec, image_points);
}

int pinhole::undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const cv::Mat &R, const cv::Mat &P
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src, dst_points, R, P);
}

int pinhole::undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &R, const cv::Mat &P
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src_points, dst_points, R, P);
}

int pinhole::distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src, dst_points);
}

int pinhole::distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs
)
{
  CameraModel model{};
  if (buildPinholeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src_points, dst_points);
}

bool pinhole::undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix, const int interpolation, const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildPinholeModelWithNewK(camera_matrix, dist_coeffs, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::remapImage(
    src, dst, distorted, undistorted, interpolation, solver_options
  );
}

bool pinhole::distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix, const int interpolation, const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildPinholeModelWithNewK(camera_matrix, dist_coeffs, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::distortImage(
    src, dst, undistorted, distorted, interpolation, solver_options
  );
}

// ---------------------------------------------------------------------------
// fisheye namespace
// ---------------------------------------------------------------------------

int fisheye::projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point3f> pts;
  if (!extractPoints3f(object_points, pts))
  {
    return -1;
  }
  cv::Vec3d r;
  if (!extractVec3d(rvec, r))
  {
    return -1;
  }
  cv::Vec3d t;
  if (!extractVec3d(tvec, t))
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, pts, r, t, image_points);
}

int fisheye::projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, object_points, rvec, tvec, image_points);
}

int fisheye::undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const cv::Mat &R, const cv::Mat &P
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src, dst_points, R, P);
}

int fisheye::undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &R, const cv::Mat &P
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src_points, dst_points, R, P);
}

int fisheye::distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src, dst_points);
}

int fisheye::distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs
)
{
  CameraModel model{};
  if (buildFisheyeModel(camera_matrix, dist_coeffs, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src_points, dst_points);
}

bool fisheye::undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix, const int interpolation, const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildFisheyeModelWithNewK(camera_matrix, dist_coeffs, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::remapImage(
    src, dst, distorted, undistorted, interpolation, solver_options
  );
}

bool fisheye::distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const cv::Mat &new_camera_matrix, const int interpolation, const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildFisheyeModelWithNewK(camera_matrix, dist_coeffs, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::distortImage(
    src, dst, undistorted, distorted, interpolation, solver_options
  );
}

// ---------------------------------------------------------------------------
// omnidirectional namespace
// ---------------------------------------------------------------------------

int omnidirectional::projectPoints(
  const cv::Mat &object_points, const cv::Mat &rvec, const cv::Mat &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi,
  std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point3f> pts;
  if (!extractPoints3f(object_points, pts))
  {
    return -1;
  }
  cv::Vec3d r;
  if (!extractVec3d(rvec, r))
  {
    return -1;
  }
  cv::Vec3d t;
  if (!extractVec3d(tvec, t))
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, pts, r, t, image_points);
}

int omnidirectional::projectPoints(
  const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi,
  std::vector<cv::Point2f> &image_points
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::projectPoints(model, object_points, rvec, tvec, image_points);
}

int omnidirectional::undistortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const double xi, const cv::Mat &R, const cv::Mat &P
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src, dst_points, R, P);
}

int omnidirectional::undistortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi, const cv::Mat &R,
  const cv::Mat &P
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::undistortPoints(model, src_points, dst_points, R, P);
}

int omnidirectional::distortPoints(
  const cv::Mat &src_points, std::vector<cv::Point2f> &dst_points, const cv::Mat &camera_matrix,
  const cv::Mat &dist_coeffs, const double xi
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  std::vector<cv::Point2f> src;
  if (!extractPoints2f(src_points, src))
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src, dst_points);
}

int omnidirectional::distortPoints(
  const std::vector<cv::Point2f> &src_points, std::vector<cv::Point2f> &dst_points,
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi
)
{
  CameraModel model{};
  if (buildOmniModel(camera_matrix, dist_coeffs, xi, model) != StatusCode::OK)
  {
    return -1;
  }
  return camgeom_opencv::distortPoints(model, src_points, dst_points);
}

bool omnidirectional::undistortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const double xi, const cv::Mat &new_camera_matrix, const int interpolation,
  const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildOmniModelWithNewK(camera_matrix, dist_coeffs, xi, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::remapImage(
    src, dst, distorted, undistorted, interpolation, solver_options
  );
}

bool omnidirectional::distortImage(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
  const double xi, const cv::Mat &new_camera_matrix, const int interpolation,
  const SolverOptions &solver_options
)
{
  CameraModel distorted{};
  CameraModel undistorted{};
  if (buildOmniModelWithNewK(camera_matrix, dist_coeffs, xi, new_camera_matrix, distorted, undistorted) != StatusCode::OK)
  {
    return false;
  }
  return camgeom_opencv::distortImage(
    src, dst, undistorted, distorted, interpolation, solver_options
  );
}

}  // namespace camxiom::opencv

#endif  // __has_include(<opencv2/core.hpp>)
