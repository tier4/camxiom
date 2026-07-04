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

#include "camxiom/remap.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace camxiom::opencv
{
namespace camgeom = ::camxiom;

namespace
{

// cv::remap cannot operate in place: when src and dst share a buffer it
// reads pixels it has already overwritten and silently corrupts the output.
// Detour through a temporary whenever the caller passed the same allocation
// (or overlapping views of it) for both.
void remapSafe(
  const cv::Mat &src, cv::Mat &dst, const cv::Mat &map1, const cv::Mat &map2,
  const int interpolation, const int border_mode, const cv::Scalar &border_value
)
{
  if (!src.empty() && src.datastart == dst.datastart)
  {
    cv::Mat tmp;
    cv::remap(src, tmp, map1, map2, interpolation, border_mode, border_value);
    dst = tmp;
    return;
  }
  cv::remap(src, dst, map1, map2, interpolation, border_mode, border_value);
}

bool computeTotalPixelCountChecked(const int width, const int height, int &total_out)
{
  if (width <= 0 || height <= 0)
  {
    return false;
  }
  const long long total_ll = static_cast<long long>(width) * static_cast<long long>(height);
  if (total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    return false;
  }
  total_out = static_cast<int>(total_ll);
  return true;
}

RemapResult fillMapsFromRaw(
  const float *map_x_raw, const float *map_y_raw, const int width, const int height, cv::Mat &map1,
  cv::Mat &map2, const RemapResult &raw_result
)
{
  cv::Mat map_x_f32(height, width, CV_32FC1, const_cast<float *>(map_x_raw));
  cv::Mat map_y_f32(height, width, CV_32FC1, const_cast<float *>(map_y_raw));
  cv::convertMaps(map_x_f32, map_y_f32, map1, map2, CV_16SC2);
  return raw_result;
}

}  // namespace

RemapResult buildRemapMapCV(
  const CameraModel &src_model, const CameraModel &dst_model, const int width, const int height,
  cv::Mat &map1, cv::Mat &map2, const SolverOptions &solver_options
)
{
  int total = 0;
  if (!computeTotalPixelCountChecked(width, height, total))
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    return RemapResult{camgeom::StatusCode::INVALID_INPUT, 0, 0};
  }

  std::vector<float> map_x_raw(static_cast<std::size_t>(total));
  std::vector<float> map_y_raw(static_cast<std::size_t>(total));

  const RemapResult result = camgeom::buildRemapMap(
    src_model, dst_model, width, height, map_x_raw.data(), map_y_raw.data(), solver_options
  );
  if (result.status != camgeom::StatusCode::OK)
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    return result;
  }

  return fillMapsFromRaw(map_x_raw.data(), map_y_raw.data(), width, height, map1, map2, result);
}

RemapResult buildUndistortRemapMapCV(
  const CameraModel &src_model, const int width, const int height, cv::Mat &map1, cv::Mat &map2,
  const SolverOptions &solver_options
)
{
  int total = 0;
  if (!computeTotalPixelCountChecked(width, height, total))
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    return RemapResult{camgeom::StatusCode::INVALID_INPUT, 0, 0};
  }

  std::vector<float> map_x_raw(static_cast<std::size_t>(total));
  std::vector<float> map_y_raw(static_cast<std::size_t>(total));

  const RemapResult result = camgeom::buildUndistortRemapMap(
    src_model, width, height, map_x_raw.data(), map_y_raw.data(), solver_options
  );
  if (result.status != camgeom::StatusCode::OK)
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    return result;
  }

  return fillMapsFromRaw(map_x_raw.data(), map_y_raw.data(), width, height, map1, map2, result);
}

// ---------------------------------------------------------------------------
// RemapCache
// ---------------------------------------------------------------------------

RemapResult RemapCache::build(
  const CameraModel &src_model, const CameraModel &dst_model, const int width, const int height,
  const SolverOptions &solver_options
)
{
  clear();
  const RemapResult result =
    buildRemapMapCV(src_model, dst_model, width, height, map1_, map2_, solver_options);
  width_ = (result.status == camgeom::StatusCode::OK) ? width : 0;
  height_ = (result.status == camgeom::StatusCode::OK) ? height : 0;
  valid_ = (result.status == camgeom::StatusCode::OK) && (result.valid_count > 0);
  return result;
}

RemapResult RemapCache::buildUndistort(
  const CameraModel &src_model, const int width, const int height,
  const SolverOptions &solver_options
)
{
  clear();
  const RemapResult result =
    buildUndistortRemapMapCV(src_model, width, height, map1_, map2_, solver_options);
  width_ = (result.status == camgeom::StatusCode::OK) ? width : 0;
  height_ = (result.status == camgeom::StatusCode::OK) ? height : 0;
  valid_ = (result.status == camgeom::StatusCode::OK) && (result.valid_count > 0);
  return result;
}

bool RemapCache::apply(
  const cv::Mat &src, cv::Mat &dst, const int interpolation, const int border_mode,
  const cv::Scalar &border_value
) const
{
  if (!valid_ || map1_.empty() || map2_.empty())
  {
    return false;
  }
  remapSafe(src, dst, map1_, map2_, interpolation, border_mode, border_value);
  return true;
}

bool RemapCache::isValid() const { return valid_; }
int RemapCache::width() const { return width_; }
int RemapCache::height() const { return height_; }
const cv::Mat &RemapCache::map1() const { return map1_; }
const cv::Mat &RemapCache::map2() const { return map2_; }

void RemapCache::clear()
{
  map1_ = cv::Mat{};
  map2_ = cv::Mat{};
  width_ = 0;
  height_ = 0;
  valid_ = false;
}

// ---------------------------------------------------------------------------
// buildRectifyRemapMapCV
// ---------------------------------------------------------------------------

RectifyRemapResult buildRectifyRemapMapCV(
  const CameraModel &src_model, const ImageSize src_size,
  const RectifiedOutputModelOptions &options, cv::Mat &map1, cv::Mat &map2
)
{
  const ImageSize out_size = options.output_size.isValid() ? options.output_size : src_size;

  int total = 0;
  if (!computeTotalPixelCountChecked(out_size.width, out_size.height, total))
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    RectifyRemapResult fail{};
    fail.remap_result.status = camgeom::StatusCode::INVALID_INPUT;
    return fail;
  }

  std::vector<float> map_x_raw(static_cast<std::size_t>(total));
  std::vector<float> map_y_raw(static_cast<std::size_t>(total));

  RectifyRemapResult result =
    camgeom::buildRectifyRemapMap(src_model, src_size, options, map_x_raw.data(), map_y_raw.data());

  if (result.remap_result.status != camgeom::StatusCode::OK)
  {
    map1 = cv::Mat{};
    map2 = cv::Mat{};
    return result;
  }

  cv::Mat map_x_f32(out_size.height, out_size.width, CV_32FC1, map_x_raw.data());
  cv::Mat map_y_f32(out_size.height, out_size.width, CV_32FC1, map_y_raw.data());
  cv::convertMaps(map_x_f32, map_y_f32, map1, map2, CV_16SC2);
  return result;
}

RectifyRemapResult RemapCache::buildRectify(
  const CameraModel &src_model, const ImageSize src_size, const RectifiedOutputModelOptions &options
)
{
  clear();
  RectifyRemapResult result = buildRectifyRemapMapCV(src_model, src_size, options, map1_, map2_);
  const ImageSize out_size = result.output_size;
  width_ = (result.remap_result.status == camgeom::StatusCode::OK) ? out_size.width : 0;
  height_ = (result.remap_result.status == camgeom::StatusCode::OK) ? out_size.height : 0;
  valid_ = (result.remap_result.status == camgeom::StatusCode::OK) &&
           (result.remap_result.source_in_bounds_count > 0);
  return result;
}

// ---------------------------------------------------------------------------
// undistortImage / remapImage / rectifyImage / distortImage
// ---------------------------------------------------------------------------

bool undistortImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, const int interpolation,
  const SolverOptions &solver_options
)
{
  if (src.empty())
  {
    return false;
  }

  cv::Mat map1;
  cv::Mat map2;
  const RemapResult result =
    buildUndistortRemapMapCV(src_model, src.cols, src.rows, map1, map2, solver_options);

  if (result.status != camgeom::StatusCode::OK || result.valid_count == 0)
  {
    return false;
  }

  remapSafe(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

bool remapImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, const CameraModel &dst_model,
  const int interpolation, const SolverOptions &solver_options
)
{
  if (src.empty())
  {
    return false;
  }

  cv::Mat map1;
  cv::Mat map2;
  const RemapResult result =
    buildRemapMapCV(src_model, dst_model, src.cols, src.rows, map1, map2, solver_options);

  if (result.status != camgeom::StatusCode::OK || result.valid_count == 0)
  {
    return false;
  }

  remapSafe(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

bool rectifyImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model,
  const RectifiedOutputModelOptions &options, CameraModel *output_model_out, const int interpolation
)
{
  if (src.empty())
  {
    return false;
  }

  const ImageSize src_size{src.cols, src.rows};
  RectifiedOutputModelOptions opts = options;
  if (!opts.output_size.isValid())
  {
    opts.output_size = src_size;
  }

  cv::Mat map1;
  cv::Mat map2;
  const RectifyRemapResult result = buildRectifyRemapMapCV(src_model, src_size, opts, map1, map2);

  if (result.remap_result.status != camgeom::StatusCode::OK || result.remap_result.source_in_bounds_count == 0)
  {
    return false;
  }

  if (output_model_out != nullptr)
  {
    *output_model_out = result.output_model;
  }

  remapSafe(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

bool distortImage(
  const cv::Mat &src, cv::Mat &dst, const CameraModel &src_model, const CameraModel &dst_model,
  const int interpolation, const SolverOptions &solver_options
)
{
  if (src.empty())
  {
    return false;
  }

  cv::Mat map1;
  cv::Mat map2;
  const RemapResult result =
    buildRemapMapCV(src_model, dst_model, src.cols, src.rows, map1, map2, solver_options);

  if (result.status != camgeom::StatusCode::OK || result.valid_count == 0)
  {
    return false;
  }

  remapSafe(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT, cv::Scalar());
  return true;
}

}  // namespace camxiom::opencv

#endif  // __has_include(<opencv2/core.hpp>)
