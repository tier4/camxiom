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

#include "camxiom/batch.hpp"
#include "camxiom/model.hpp"
#include "detail/internal.hpp"  // validateCameraModelQuery
#include "opencv/internal.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace camxiom::opencv
{

int projectPoints(
  const CameraModel &model, const std::vector<cv::Point3f> &object_points, const cv::Vec3d &rvec,
  const cv::Vec3d &tvec, std::vector<cv::Point2f> &image_points_out
)
{
  const StatusCode validation = camgeom::detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    return -1;
  }

  const int n = static_cast<int>(object_points.size());
  if (n == 0)
  {
    image_points_out.clear();
    return 0;
  }

  cv::Matx33d R;
  cv::Rodrigues(rvec, R);

  std::vector<float> rays_xyz(static_cast<std::size_t>(n) * 3);
  for (int i = 0; i < n; ++i)
  {
    const cv::Point3f &pt = object_points[static_cast<std::size_t>(i)];
    const std::size_t base = static_cast<std::size_t>(i) * 3;
    rays_xyz[base + 0] =
      static_cast<float>(R(0, 0) * pt.x + R(0, 1) * pt.y + R(0, 2) * pt.z + tvec[0]);
    rays_xyz[base + 1] =
      static_cast<float>(R(1, 0) * pt.x + R(1, 1) * pt.y + R(1, 2) * pt.z + tvec[1]);
    rays_xyz[base + 2] =
      static_cast<float>(R(2, 0) * pt.x + R(2, 1) * pt.y + R(2, 2) * pt.z + tvec[2]);
  }

  std::vector<float> u_out(static_cast<std::size_t>(n));
  std::vector<float> v_out(static_cast<std::size_t>(n));
  std::vector<StatusCode> statuses(static_cast<std::size_t>(n));
  const int valid_count = camgeom::rayToPixelBatch(
    model, rays_xyz.data(), n, u_out.data(), v_out.data(), statuses.data()
  );

  image_points_out.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    if (statuses[static_cast<std::size_t>(i)] == StatusCode::OK)
    {
      image_points_out[static_cast<std::size_t>(i)] =
        cv::Point2f(u_out[static_cast<std::size_t>(i)], v_out[static_cast<std::size_t>(i)]);
    }
    else
    {
      image_points_out[static_cast<std::size_t>(i)] = cv::Point2f(kNaN, kNaN);
    }
  }
  return valid_count;
}

int undistortPoints(
  const CameraModel &model, const std::vector<cv::Point2f> &src_points,
  std::vector<cv::Point2f> &dst_points_out, const cv::Mat &R_mat, const cv::Mat &P_mat,
  const SolverOptions &opts
)
{
  const StatusCode validation = camgeom::detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    return -1;
  }

  const int n = static_cast<int>(src_points.size());
  if (n == 0)
  {
    dst_points_out.clear();
    return 0;
  }

  std::vector<float> u_in(static_cast<std::size_t>(n));
  std::vector<float> v_in(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    u_in[static_cast<std::size_t>(i)] = src_points[static_cast<std::size_t>(i)].x;
    v_in[static_cast<std::size_t>(i)] = src_points[static_cast<std::size_t>(i)].y;
  }

  std::vector<float> dirs_xyz(static_cast<std::size_t>(n) * 3);
  std::vector<StatusCode> statuses(static_cast<std::size_t>(n));
  const int valid_count = camgeom::pixelToRayBatch(
    model, u_in.data(), v_in.data(), n, dirs_xyz.data(), statuses.data(), opts
  );
  (void)valid_count;

  cv::Matx33d R = cv::Matx33d::eye();
  if (!parseRectificationMatrix(R_mat, R))
  {
    dst_points_out.clear();
    return -1;
  }
  bool has_P = false;
  double pfx = 1.0;
  double pfy = 1.0;
  double pcx = 0.0;
  double pcy = 0.0;
  if (!parseProjectionMatrix(P_mat, has_P, pfx, pfy, pcx, pcy))
  {
    dst_points_out.clear();
    return -1;
  }

  int valid_count_out = 0;
  dst_points_out.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    const std::size_t idx = static_cast<std::size_t>(i);
    if (statuses[idx] != StatusCode::OK)
    {
      dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
      continue;
    }

    const std::size_t base = idx * 3;
    const double dz = static_cast<double>(dirs_xyz[base + 2]);
    if (!std::isfinite(dz) || std::abs(dz) <= kDivisionEpsilon)
    {
      dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
      continue;
    }
    const double iz = 1.0 / dz;
    double xn = static_cast<double>(dirs_xyz[base + 0]) * iz;
    double yn = static_cast<double>(dirs_xyz[base + 1]) * iz;
    if (!std::isfinite(xn) || !std::isfinite(yn))
    {
      dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
      continue;
    }

    if (!R_mat.empty())
    {
      const double xr = R(0, 0) * xn + R(0, 1) * yn + R(0, 2);
      const double yr = R(1, 0) * xn + R(1, 1) * yn + R(1, 2);
      const double zr = R(2, 0) * xn + R(2, 1) * yn + R(2, 2);
      if (!std::isfinite(zr) || std::abs(zr) <= kDivisionEpsilon)
      {
        dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
        continue;
      }
      const double izr = 1.0 / zr;
      xn = xr * izr;
      yn = yr * izr;
      if (!std::isfinite(xn) || !std::isfinite(yn))
      {
        dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
        continue;
      }
    }

    if (has_P)
    {
      xn = pfx * xn + pcx;
      yn = pfy * yn + pcy;
      if (!std::isfinite(xn) || !std::isfinite(yn))
      {
        dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
        continue;
      }
    }

    dst_points_out[idx] = cv::Point2f(static_cast<float>(xn), static_cast<float>(yn));
    ++valid_count_out;
  }
  return valid_count_out;
}

int distortPoints(
  const CameraModel &model, const std::vector<cv::Point2f> &src_points,
  std::vector<cv::Point2f> &dst_points_out
)
{
  const StatusCode validation = camgeom::detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    return -1;
  }

  const int n = static_cast<int>(src_points.size());
  if (n == 0)
  {
    dst_points_out.clear();
    return 0;
  }

  std::vector<float> rays_xyz(static_cast<std::size_t>(n) * 3);
  for (int i = 0; i < n; ++i)
  {
    const std::size_t base = static_cast<std::size_t>(i) * 3;
    rays_xyz[base + 0] = src_points[static_cast<std::size_t>(i)].x;
    rays_xyz[base + 1] = src_points[static_cast<std::size_t>(i)].y;
    rays_xyz[base + 2] = 1.0f;
  }

  std::vector<float> u_out(static_cast<std::size_t>(n));
  std::vector<float> v_out(static_cast<std::size_t>(n));
  std::vector<StatusCode> statuses(static_cast<std::size_t>(n));
  const int valid_count = camgeom::rayToPixelBatch(
    model, rays_xyz.data(), n, u_out.data(), v_out.data(), statuses.data()
  );

  dst_points_out.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    const std::size_t idx = static_cast<std::size_t>(i);
    if (statuses[idx] == StatusCode::OK)
    {
      dst_points_out[idx] = cv::Point2f(u_out[idx], v_out[idx]);
    }
    else
    {
      dst_points_out[idx] = cv::Point2f(kNaN, kNaN);
    }
  }
  return valid_count;
}

bool initCameraMatrix2D(
  const std::vector<std::vector<cv::Point3f>> &object_points_per_view,
  const std::vector<std::vector<cv::Point2f>> &image_points_per_view, const cv::Size &image_size,
  CameraModel &camera_model_out
)
{
  camera_model_out = CameraModel{};

  if (object_points_per_view.empty() ||
      object_points_per_view.size() != image_points_per_view.size() ||
      image_size.width <= 0 || image_size.height <= 0)
  {
    return false;
  }

  for (std::size_t i = 0; i < object_points_per_view.size(); ++i)
  {
    if (object_points_per_view[i].size() < 4 ||
        object_points_per_view[i].size() != image_points_per_view[i].size())
    {
      return false;
    }
  }

  cv::Mat K;
  try
  {
    K = cv::initCameraMatrix2D(object_points_per_view, image_points_per_view, image_size);
  }
  catch (const cv::Exception &)
  {
    return false;
  }

  if (K.empty() || K.rows != 3 || K.cols != 3)
  {
    return false;
  }

  cv::Mat K64;
  K.convertTo(K64, CV_64F);
  const double fx = K64.at<double>(0, 0);
  const double fy = K64.at<double>(1, 1);
  const double cx = K64.at<double>(0, 2);
  const double cy = K64.at<double>(1, 2);

  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy) || fx <= 0.0 || fy <= 0.0)
  {
    return false;
  }

  camera_model_out.intrinsics.fx = static_cast<float>(fx);
  camera_model_out.intrinsics.fy = static_cast<float>(fy);
  camera_model_out.intrinsics.cx = static_cast<float>(cx);
  camera_model_out.intrinsics.cy = static_cast<float>(cy);
  camera_model_out.projection.type = camgeom::ProjectionModelType::PINHOLE;
  camera_model_out.distortion.type = camgeom::DistortionModelType::NONE;
  camera_model_out.distortion.count = 0;

  return camgeom::validateCameraModel(camera_model_out) == camgeom::StatusCode::OK;
}

}  // namespace camxiom::opencv

#endif  // __has_include(<opencv2/core.hpp>)
