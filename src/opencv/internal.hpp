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

#ifndef CAMXIOM__SRC__OPENCV__INTERNAL_HPP
#define CAMXIOM__SRC__OPENCV__INTERNAL_HPP

#if __has_include(<opencv2/core.hpp>)

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/opencv.hpp"
#include "camxiom/types.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace camxiom::opencv
{
namespace camgeom = ::camxiom;

using camgeom::CameraModel;
using camgeom::StatusCode;

inline constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
inline constexpr double kDivisionEpsilon = 1e-12;

inline bool extractK(const cv::Mat &camera_matrix, std::array<double, 9> &K_out)
{
  // channels() must be checked explicitly: convertTo preserves the channel
  // count and Mat::at's own type check is CV_DbgAssert (compiled out in
  // Release), so a CV_64FC3 input would be read with the wrong stride and
  // silently produce garbage intrinsics.
  if (camera_matrix.empty() || camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.channels() != 1)
  {
    return false;
  }
  cv::Mat K64;
  camera_matrix.convertTo(K64, CV_64F);
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      K_out[static_cast<std::size_t>(r * 3 + c)] = K64.at<double>(r, c);
    }
  }
  return true;
}

inline std::vector<double> extractD(const cv::Mat &dist_coeffs)
{
  if (dist_coeffs.empty())
  {
    return {};
  }
  cv::Mat d64;
  dist_coeffs.convertTo(d64, CV_64F);
  d64 = d64.reshape(1, 1);
  std::vector<double> D(static_cast<std::size_t>(d64.cols));
  for (int i = 0; i < d64.cols; ++i)
  {
    D[static_cast<std::size_t>(i)] = d64.at<double>(0, i);
  }
  return D;
}

inline bool extractPoints3f(const cv::Mat &points, std::vector<cv::Point3f> &result_out)
{
  result_out.clear();
  if (points.empty())
  {
    return true;
  }

  cv::Mat pts = points;
  if (points.channels() == 3)
  {
    // A row-count-changing reshape throws cv::Exception on non-continuous
    // views (ROI/submat); go through a continuous clone so the callers'
    // "reject invalid input" contract holds instead of an exception
    // escaping. The reshaped header keeps the clone's buffer alive via
    // refcounting.
    const cv::Mat contiguous = points.isContinuous() ? points : points.clone();
    pts = contiguous.reshape(1, static_cast<int>(contiguous.total()));
  }
  else if (points.channels() != 1)
  {
    return false;
  }
  if (pts.cols != 3)
  {
    return false;
  }

  cv::Mat pts32;
  pts.convertTo(pts32, CV_32F);
  result_out.resize(static_cast<std::size_t>(pts32.rows));
  for (int i = 0; i < pts32.rows; ++i)
  {
    const float x = pts32.at<float>(i, 0);
    const float y = pts32.at<float>(i, 1);
    const float z = pts32.at<float>(i, 2);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
    {
      result_out.clear();
      return false;
    }
    result_out[static_cast<std::size_t>(i)] = cv::Point3f(x, y, z);
  }
  return true;
}

inline bool extractPoints2f(const cv::Mat &points, std::vector<cv::Point2f> &result_out)
{
  result_out.clear();
  if (points.empty())
  {
    return true;
  }

  cv::Mat pts = points;
  if (points.channels() == 2)
  {
    // See extractPoints3f: clone non-continuous views before reshaping.
    const cv::Mat contiguous = points.isContinuous() ? points : points.clone();
    pts = contiguous.reshape(1, static_cast<int>(contiguous.total()));
  }
  else if (points.channels() != 1)
  {
    return false;
  }
  if (pts.cols != 2)
  {
    return false;
  }

  cv::Mat pts32;
  pts.convertTo(pts32, CV_32F);
  result_out.resize(static_cast<std::size_t>(pts32.rows));
  for (int i = 0; i < pts32.rows; ++i)
  {
    const float x = pts32.at<float>(i, 0);
    const float y = pts32.at<float>(i, 1);
    if (!std::isfinite(x) || !std::isfinite(y))
    {
      result_out.clear();
      return false;
    }
    result_out[static_cast<std::size_t>(i)] = cv::Point2f(x, y);
  }
  return true;
}

inline bool extractVec3d(const cv::Mat &vec, cv::Vec3d &out)
{
  if (vec.empty() || vec.total() != 3)
  {
    return false;
  }
  cv::Mat v64;
  vec.convertTo(v64, CV_64F);
  const cv::Mat flattened = v64.reshape(1, 1);
  const double x = flattened.at<double>(0, 0);
  const double y = flattened.at<double>(0, 1);
  const double z = flattened.at<double>(0, 2);
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
  {
    return false;
  }
  out = cv::Vec3d(x, y, z);
  return true;
}

inline bool parseRectificationMatrix(const cv::Mat &R_mat, cv::Matx33d &R_out)
{
  if (R_mat.empty())
  {
    R_out = cv::Matx33d::eye();
    return true;
  }
  // Same single-channel requirement as extractK (Release-mode at<double>
  // does not type-check).
  if (R_mat.rows != 3 || R_mat.cols != 3 || R_mat.channels() != 1)
  {
    return false;
  }
  cv::Mat R64;
  R_mat.convertTo(R64, CV_64F);
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      const double value = R64.at<double>(row, col);
      if (!std::isfinite(value))
      {
        return false;
      }
      R_out(row, col) = value;
    }
  }
  return true;
}

inline bool parseProjectionMatrix(
  const cv::Mat &P_mat, bool &has_projection, double &pfx, double &pfy, double &pcx, double &pcy
)
{
  if (P_mat.empty())
  {
    has_projection = false;
    pfx = 1.0;
    pfy = 1.0;
    pcx = 0.0;
    pcy = 0.0;
    return true;
  }
  // Same single-channel requirement as extractK (Release-mode at<double>
  // does not type-check).
  if (P_mat.rows != 3 || (P_mat.cols != 3 && P_mat.cols != 4) || P_mat.channels() != 1)
  {
    return false;
  }
  cv::Mat P64;
  P_mat.convertTo(P64, CV_64F);
  const double fx = P64.at<double>(0, 0);
  const double fy = P64.at<double>(1, 1);
  const double cx = P64.at<double>(0, 2);
  const double cy = P64.at<double>(1, 2);
  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) || !std::isfinite(cy))
  {
    return false;
  }
  has_projection = true;
  pfx = fx;
  pfy = fy;
  pcx = cx;
  pcy = cy;
  return true;
}

inline camgeom::PinholeCompatProfile choosePinholeProfile(const std::size_t d_size)
{
  using camgeom::PinholeCompatProfile;
  switch (d_size)
  {
    case 0:
      return PinholeCompatProfile::CANONICAL;
    case 4:
      return PinholeCompatProfile::OPENCV_CALIB3D_D4;
    case 5:
      return PinholeCompatProfile::OPENCV_CALIB3D_D5;
    case 8:
      return PinholeCompatProfile::OPENCV_CALIB3D_D8;
    case 12:
      return PinholeCompatProfile::OPENCV_CALIB3D_D12;
    case 14:
      return PinholeCompatProfile::OPENCV_CALIB3D_D14;
    default:
      return PinholeCompatProfile::CANONICAL;
  }
}

inline StatusCode buildPinholeModel(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, CameraModel &model_out
)
{
  std::array<double, 9> K{};
  if (!extractK(camera_matrix, K))
  {
    return StatusCode::INVALID_INPUT;
  }
  const std::vector<double> D = extractD(dist_coeffs);
  camgeom::PinholeExternalModel ext;
  ext.K = K;
  ext.D = D;
  return camgeom::importPinholeModel(ext, choosePinholeProfile(D.size()), model_out);
}

inline StatusCode buildFisheyeModel(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, CameraModel &model_out
)
{
  std::array<double, 9> K{};
  if (!extractK(camera_matrix, K))
  {
    return StatusCode::INVALID_INPUT;
  }
  const std::vector<double> D = extractD(dist_coeffs);
  camgeom::FisheyeExternalModel ext;
  ext.K = K;
  ext.D = D;
  return camgeom::importFisheyeModel(
    ext, camgeom::FisheyeCompatProfile::OPENCV_FISHEYE_D4, model_out
  );
}

inline StatusCode buildPinholeModelWithNewK(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &new_camera_matrix,
  CameraModel &distorted_out, CameraModel &undistorted_out
)
{
  const StatusCode status = buildPinholeModel(camera_matrix, dist_coeffs, distorted_out);
  if (status != StatusCode::OK)
  {
    return status;
  }
  if (!new_camera_matrix.empty())
  {
    CameraModel new_k_model{};
    const StatusCode nk_status = buildPinholeModel(new_camera_matrix, cv::Mat(), new_k_model);
    if (nk_status != StatusCode::OK)
    {
      return nk_status;
    }
    undistorted_out = new_k_model;
  }
  else
  {
    undistorted_out = camgeom::makeDistortionFree(distorted_out);
  }
  return StatusCode::OK;
}

inline StatusCode buildFisheyeModelWithNewK(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const cv::Mat &new_camera_matrix,
  CameraModel &distorted_out, CameraModel &undistorted_out
)
{
  const StatusCode status = buildFisheyeModel(camera_matrix, dist_coeffs, distorted_out);
  if (status != StatusCode::OK)
  {
    return status;
  }
  if (!new_camera_matrix.empty())
  {
    CameraModel new_k_model{};
    const StatusCode nk_status = buildFisheyeModel(new_camera_matrix, cv::Mat(), new_k_model);
    if (nk_status != StatusCode::OK)
    {
      return nk_status;
    }
    undistorted_out = new_k_model;
  }
  else
  {
    undistorted_out = camgeom::makeDistortionFree(distorted_out);
  }
  return StatusCode::OK;
}

inline camgeom::OmnidirectionalCompatProfile chooseOmniProfile(const std::size_t d_size)
{
  using camgeom::OmnidirectionalCompatProfile;
  switch (d_size)
  {
    case 0:
      return OmnidirectionalCompatProfile::CANONICAL;
    case 4:
      return OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4;
    case 5:
      return OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D5;
    case 8:
      return OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D8;
    default:
      return OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4;
  }
}

inline StatusCode buildOmniModel(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi, CameraModel &model_out
)
{
  std::array<double, 9> K{};
  if (!extractK(camera_matrix, K))
  {
    return StatusCode::INVALID_INPUT;
  }
  const std::vector<double> D = extractD(dist_coeffs);
  camgeom::OmnidirectionalExternalModel ext;
  ext.K = K;
  ext.D = D;
  ext.xi = xi;
  return camgeom::importOmnidirectionalModel(ext, chooseOmniProfile(D.size()), model_out);
}

inline StatusCode buildOmniModelWithNewK(
  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double xi,
  const cv::Mat &new_camera_matrix, CameraModel &distorted_out, CameraModel &undistorted_out
)
{
  const StatusCode status = buildOmniModel(camera_matrix, dist_coeffs, xi, distorted_out);
  if (status != StatusCode::OK)
  {
    return status;
  }
  if (!new_camera_matrix.empty())
  {
    CameraModel new_k_model{};
    const StatusCode nk_status = buildOmniModel(new_camera_matrix, cv::Mat(), xi, new_k_model);
    if (nk_status != StatusCode::OK)
    {
      return nk_status;
    }
    undistorted_out = new_k_model;
  }
  else
  {
    undistorted_out = camgeom::makeDistortionFree(distorted_out);
  }
  return StatusCode::OK;
}

}  // namespace camxiom::opencv

#endif  // __has_include(<opencv2/core.hpp>)

#endif  // CAMXIOM__SRC__OPENCV__INTERNAL_HPP
