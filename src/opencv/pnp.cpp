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

#include "camxiom/opencv/pnp.hpp"

#if __has_include(<opencv2/core.hpp>)

#include "camxiom/opencv.hpp"
#include "camxiom/types.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <iostream>
#include <limits>

namespace camgeom_opencv = camxiom::opencv;

namespace camxiom::opencv
{
namespace camgeom = ::camxiom;

bool solvePnP(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points,
  const std::vector<cv::Point2f> &image_points, cv::Vec3d &rvec_out, cv::Vec3d &tvec_out,
  const SolvePnPConfig &config, const SolverOptions &solver_options
)
{
  if (object_points.size() < 4 || object_points.size() != image_points.size())
  {
    return false;
  }

  std::vector<cv::Point2f> normalized;
  const int undist_ok =
    undistortPoints(model, image_points, normalized, cv::Mat(), cv::Mat(), solver_options);
  if (undist_ok < 0 || normalized.size() != image_points.size())
  {
    std::cerr << "[camxiom::opencv::solvePnP] undistortPoints failed"
              << " projection=" << camgeom::toString(model.projection.type)
              << " distortion=" << camgeom::toString(model.distortion.type)
              << " points=" << image_points.size() << " undist_ok=" << undist_ok
              << " normalized_size=" << normalized.size() << std::endl;
    return false;
  }

  std::vector<cv::Point3f> filtered_object_points;
  std::vector<cv::Point2f> filtered_normalized_points;
  filtered_object_points.reserve(object_points.size());
  filtered_normalized_points.reserve(normalized.size());
  for (std::size_t i = 0; i < normalized.size(); ++i)
  {
    const cv::Point2f &point = normalized[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y))
    {
      continue;
    }
    filtered_object_points.push_back(object_points[i]);
    filtered_normalized_points.push_back(point);
  }

  if (filtered_object_points.size() < 4)
  {
    std::cerr << "[camxiom::opencv::solvePnP] insufficient finite normalized points after filtering"
              << " projection=" << camgeom::toString(model.projection.type)
              << " distortion=" << camgeom::toString(model.distortion.type)
              << " points=" << object_points.size()
              << " finite_points=" << filtered_object_points.size() << std::endl;
    return false;
  }

  static const cv::Mat identity_K =
    (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  static const cv::Mat zero_D;

  bool solved = cv::solvePnP(
    filtered_object_points, filtered_normalized_points, identity_K, zero_D, rvec_out, tvec_out,
    false, config.method
  );

  if (!solved && config.use_fallback && config.fallback_method != config.method)
  {
    solved = cv::solvePnP(
      filtered_object_points, filtered_normalized_points, identity_K, zero_D, rvec_out, tvec_out,
      false, config.fallback_method
    );
  }

  if (!solved)
  {
    std::cerr << "[camxiom::opencv::solvePnP] cv::solvePnP failed"
              << " projection=" << camgeom::toString(model.projection.type)
              << " distortion=" << camgeom::toString(model.distortion.type)
              << " points=" << object_points.size()
              << " solve_points=" << filtered_object_points.size() << " method=" << config.method
              << std::endl;
  }

  return solved;
}

double reprojectRmse(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points,
  const std::vector<cv::Point2f> &image_points, const cv::Vec3d &rvec, const cv::Vec3d &tvec
)
{
  if (object_points.empty() || object_points.size() != image_points.size())
  {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<cv::Point2f> projected;
  const int proj_ok = projectPoints(model, object_points, rvec, tvec, projected);
  if (proj_ok < 0 || projected.size() != image_points.size())
  {
    return std::numeric_limits<double>::infinity();
  }

  // projectPoints writes (NaN, NaN) for points that fail to project (e.g.
  // outside the model FOV). Skip those and average over the valid points
  // only -- otherwise a single out-of-FOV point turns the whole RMSE into
  // NaN and "partially out-of-FOV pose" becomes indistinguishable from a
  // completely broken pose (same policy as the core computeReprojErrors).
  double sum_sq = 0.0;
  std::size_t valid = 0;
  for (std::size_t i = 0; i < image_points.size(); ++i)
  {
    const double px = static_cast<double>(projected[i].x);
    const double py = static_cast<double>(projected[i].y);
    if (!std::isfinite(px) || !std::isfinite(py))
    {
      continue;
    }
    const double dx = px - static_cast<double>(image_points[i].x);
    const double dy = py - static_cast<double>(image_points[i].y);
    sum_sq += dx * dx + dy * dy;
    ++valid;
  }

  if (valid == 0)
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(sum_sq / static_cast<double>(valid));
}

}  // namespace camxiom::opencv

#endif  // __has_include(<opencv2/core.hpp>)
