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

#include "camxiom/remap.hpp"

#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/validated_model.hpp"
#include "detail/internal.hpp"
#include "remap_rectify.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

namespace camxiom
{

RemapResult buildRemapMap(
  const CameraModel &src_model, const CameraModel &dst_model, const int width, const int height,
  float *map_x, float *map_y, const SolverOptions &solver_options
)
{
  if (width <= 0 || height <= 0 || map_x == nullptr || map_y == nullptr)
  {
    return RemapResult{StatusCode::INVALID_INPUT, 0, 0};
  }
  const long long total_ll = static_cast<long long>(width) * static_cast<long long>(height);
  if (total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    return RemapResult{StatusCode::INVALID_INPUT, 0, 0};
  }
  const int total = static_cast<int>(total_ll);

  const StatusCode src_valid = validateCameraModel(src_model);
  const StatusCode dst_valid = validateCameraModel(dst_model);
  if (src_valid != StatusCode::OK || dst_valid != StatusCode::OK)
  {
    for (int i = 0; i < total; ++i)
    {
      map_x[i] = -1.0f;
      map_y[i] = -1.0f;
    }
    const StatusCode failure_status = (src_valid != StatusCode::OK) ? src_valid : dst_valid;
    return RemapResult{failure_status, 0, total};
  }
  int valid_count = 0;

#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int v = 0; v < height; ++v)
  {
    const int row_offset = v * width;
    for (int u = 0; u < width; ++u)
    {
      const int idx = row_offset + u;

      const RayResult ray_result =
        pixelToRay(dst_model, Pixel2{static_cast<float>(u), static_cast<float>(v)}, solver_options);

      if (ray_result.status != StatusCode::OK)
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
        continue;
      }

      const PixelResult pixel_result = rayToPixel(src_model, ray_result.ray.direction);

      if (pixel_result.status != StatusCode::OK)
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
        continue;
      }

      map_x[idx] = pixel_result.pixel.u;
      map_y[idx] = pixel_result.pixel.v;
      ++valid_count;
    }
  }

  return RemapResult{StatusCode::OK, valid_count, total};
}

RemapResult buildUndistortRemapMap(
  const CameraModel &src_model, const int width, const int height, float *map_x, float *map_y,
  const SolverOptions &solver_options
)
{
  const CameraModel dst_model = makeDistortionFree(src_model);
  return buildRemapMap(src_model, dst_model, width, height, map_x, map_y, solver_options);
}

// ===========================================================================
// Image-aware remap
// ===========================================================================

namespace
{

// Coarse grid coverage estimation
void estimateCoverage(
  const CameraModel &src_model, const ImageSize &src_size, const CameraModel &output_model,
  const ImageSize &output_size, const Eigen::Matrix3f &src_from_output_rotation, int &out_in_bounds,
  int &out_total
)
{
  constexpr int kGridW = 64;
  constexpr int kGridH = 36;
  const float w = static_cast<float>(output_size.width);
  const float h = static_cast<float>(output_size.height);
  const float sw = static_cast<float>(src_size.width);
  const float sh = static_cast<float>(src_size.height);

  int in_bounds = 0;
  int total = 0;
  for (int gy = 0; gy <= kGridH; ++gy)
  {
    const float v =
      (kGridH > 0) ? (h - 1.0f) * static_cast<float>(gy) / static_cast<float>(kGridH) : 0.0f;
    for (int gx = 0; gx <= kGridW; ++gx)
    {
      const float u =
        (kGridW > 0) ? (w - 1.0f) * static_cast<float>(gx) / static_cast<float>(kGridW) : 0.0f;
      ++total;

      const RayResult ray = pixelToRay(output_model, Pixel2{u, v});
      if (ray.status != StatusCode::OK) continue;

      const Eigen::Vector3f rotated = src_from_output_rotation * ray.ray.direction;
      const PixelResult px = rayToPixel(src_model, rotated);
      if (px.status != StatusCode::OK) continue;

      if (px.pixel.u >= 0.0f && px.pixel.v >= 0.0f && px.pixel.u < sw && px.pixel.v < sh)
      {
        ++in_bounds;
      }
    }
  }
  out_in_bounds = in_bounds;
  out_total = total;
}

float computeFovDeg(
  const CameraModel &model, float u0, float v0, float u1, float v1,
  const SolverOptions &solver_options = SolverOptions{}
)
{
  const RayResult r0 = pixelToRay(model, Pixel2{u0, v0}, solver_options);
  const RayResult r1 = pixelToRay(model, Pixel2{u1, v1}, solver_options);
  if (r0.status != StatusCode::OK || r1.status != StatusCode::OK) return 0.0f;
  const float dot = r0.ray.direction.dot(r1.ray.direction);
  return static_cast<float>(std::acos(std::clamp(dot, -1.0f, 1.0f)) * 180.0f / detail::kPi);
}

// Length of [a, b] (either order) inside [lo, hi], in degrees.
float clippedSpanDeg(const double a, const double b, const double lo, const double hi)
{
  const double span = std::min(std::max(a, b), hi) - std::max(std::min(a, b), lo);
  return span > 0.0 ? static_cast<float>(span * 180.0 / constants::kPi) : 0.0f;
}

// A lat-long model's field of view is the longitude and latitude range covered by
// the valid pixels of the principal row and column, edge to edge (the angle
// between two edge rays, used for the other models, cannot exceed 180
// degrees). It has no diagonal angle.
FovResult latLongFov(const CameraModel &model, const ImageSize image_size)
{
  const double fx = model.intrinsics.fx;
  const double fy = model.intrinsics.fy;
  const double cx = model.intrinsics.cx;
  const double cy = model.intrinsics.cy;
  const double skew = model.intrinsics.skew;
  const double right = static_cast<double>(image_size.width) - 0.5;
  const double bottom = static_cast<double>(image_size.height) - 0.5;

  FovResult result{};
  // On the principal row (y_d = 0) the longitude is (u - cx) / fx.
  result.horizontal_fov_deg =
    clippedSpanDeg((-0.5 - cx) / fx, (right - cx) / fx, -constants::kPi, constants::kPi);

  // Down the principal column (u = cx) the longitude is -skew * y_d / fx, so
  // with skew the column leaves the valid range once that passes the seam; only
  // the rows where it stays inside [-pi, pi] count.
  double y_lo = std::min((-0.5 - cy) / fy, (bottom - cy) / fy);
  double y_hi = std::max((-0.5 - cy) / fy, (bottom - cy) / fy);
  if (skew != 0.0)
  {
    const double y_seam = constants::kPi * std::abs(fx / skew);
    y_lo = std::max(y_lo, -y_seam);
    y_hi = std::min(y_hi, y_seam);
  }
  if (y_lo >= y_hi)
  {
    result.vertical_fov_deg = 0.0f;
  }
  else if (model.projection.type == ProjectionModelType::EQUIRECTANGULAR)
  {
    result.vertical_fov_deg = clippedSpanDeg(y_lo, y_hi, -constants::kHalfPi, constants::kHalfPi);
  }
  else
  {
    result.vertical_fov_deg =
      clippedSpanDeg(std::atan(y_lo), std::atan(y_hi), -constants::kHalfPi, constants::kHalfPi);
  }
  result.diagonal_fov_deg = 0.0f;
  return result;
}

}  // namespace

const char *toString(const InvalidPixelPolicy pixel_policy)
{
  switch (pixel_policy)
  {
    case InvalidPixelPolicy::WRITE_NEGATIVE_ONE:
      return "write_negative_one";
    case InvalidPixelPolicy::WRITE_RAW_COORDINATE:
      return "write_raw_coordinate";
    default:
      return "unknown";
  }
}

FovResult computeFov(
  const CameraModel &model, const ImageSize image_size, const SolverOptions &solver_options
)
{
  FovResult result{};
  if (!image_size.isValid())
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  const StatusCode valid = validateCameraModel(model);
  if (valid != StatusCode::OK)
  {
    result.status = valid;
    return result;
  }
  if (isLatLongProjection(model.projection.type))
  {
    return latLongFov(model, image_size);
  }

  const float w = static_cast<float>(image_size.width);
  const float h = static_cast<float>(image_size.height);
  const float cx = model.intrinsics.cx;
  const float cy = model.intrinsics.cy;

  result.horizontal_fov_deg = computeFovDeg(model, 0.0f, cy, w - 1.0f, cy, solver_options);
  result.vertical_fov_deg = computeFovDeg(model, cx, 0.0f, cx, h - 1.0f, solver_options);
  result.diagonal_fov_deg = computeFovDeg(model, 0.0f, 0.0f, w - 1.0f, h - 1.0f, solver_options);
  return result;
}

ImageRemapResult buildImageRemapMap(
  const CameraModel &src_model, const ImageSize src_size, const CameraModel &dst_model,
  const ImageSize dst_size, float *map_x, float *map_y, const ImageRemapOptions &options
)
{
  if (!dst_size.isValid() || map_x == nullptr || map_y == nullptr)
  {
    return ImageRemapResult{StatusCode::INVALID_INPUT, 0, 0, 0};
  }
  // The caller asked for source-bounds checking (the default) but passed no
  // usable source size: refuse up front instead of silently skipping the
  // advertised check and reporting every model-valid pixel as in-bounds.
  if (options.require_source_in_bounds && !src_size.isValid())
  {
    return ImageRemapResult{StatusCode::INVALID_INPUT, 0, 0, 0};
  }
  const long long total_ll =
    static_cast<long long>(dst_size.width) * static_cast<long long>(dst_size.height);
  if (total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    return ImageRemapResult{StatusCode::INVALID_INPUT, 0, 0, 0};
  }
  const int total = static_cast<int>(total_ll);

  // Validated once here, so the per-pixel projections skip it (same results).
  const std::optional<ValidatedCameraModel> src_checked = ValidatedCameraModel::tryMake(src_model);
  const std::optional<ValidatedCameraModel> dst_checked = ValidatedCameraModel::tryMake(dst_model);
  if (!src_checked || !dst_checked)
  {
    for (int i = 0; i < total; ++i)
    {
      map_x[i] = -1.0f;
      map_y[i] = -1.0f;
    }
    const StatusCode src_valid = validateCameraModel(src_model);
    return ImageRemapResult{
      src_valid != StatusCode::OK ? src_valid : validateCameraModel(dst_model), 0, 0, total};
  }

  // src_size validity was enforced at entry when the check is requested.
  const bool check_bounds = options.require_source_in_bounds;
  const float src_w = static_cast<float>(src_size.width);
  const float src_h = static_cast<float>(src_size.height);
  const bool is_identity_rotation =
    options.src_from_dst_rotation.isApprox(Eigen::Matrix3f::Identity(), 1e-6f);

  int model_valid = 0;
  int in_bounds = 0;

#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : model_valid, in_bounds) schedule(static)
#endif
  for (int v = 0; v < dst_size.height; ++v)
  {
    const int row_offset = v * dst_size.width;
    for (int u = 0; u < dst_size.width; ++u)
    {
      const int idx = row_offset + u;

      const RayResult ray_result = dst_checked->pixelToRay(
        Pixel2{static_cast<float>(u), static_cast<float>(v)}, options.solver_options
      );

      if (ray_result.status != StatusCode::OK)
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
        continue;
      }

      const Eigen::Vector3f ray_src =
        is_identity_rotation ? ray_result.ray.direction
                             : (options.src_from_dst_rotation * ray_result.ray.direction);

      const PixelResult px = src_checked->rayToPixel(ray_src);
      if (px.status != StatusCode::OK)
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
        continue;
      }

      ++model_valid;

      const bool bounded = !check_bounds || (px.pixel.u >= 0.0f && px.pixel.v >= 0.0f &&
                                             px.pixel.u < src_w && px.pixel.v < src_h);

      if (bounded)
      {
        ++in_bounds;
        map_x[idx] = px.pixel.u;
        map_y[idx] = px.pixel.v;
      }
      else if (options.invalid_pixel_policy == InvalidPixelPolicy::WRITE_RAW_COORDINATE)
      {
        map_x[idx] = px.pixel.u;
        map_y[idx] = px.pixel.v;
      }
      else
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
      }
    }
  }

  return ImageRemapResult{StatusCode::OK, model_valid, in_bounds, total};
}

ImageRemapResult buildUndistortImageRemapMap(
  const CameraModel &src_model, const ImageSize image_size, float *map_x, float *map_y,
  const SolverOptions &solver_options
)
{
  const CameraModel dst_model = makeDistortionFree(src_model);
  ImageRemapOptions options{};
  options.solver_options = solver_options;
  options.require_source_in_bounds = true;
  return buildImageRemapMap(src_model, image_size, dst_model, image_size, map_x, map_y, options);
}

// ===========================================================================
// Rectification
// ===========================================================================

namespace
{

// Shared front half of makeRectifiedOutputModel / buildRectifyRemapMap:
// check the inputs, fit the scale and build the output model.
StatusCode fitRectifiedOutputModel(
  const CameraModel &src_model, const ImageSize src_size,
  const RectifiedOutputModelOptions &options, const ImageSize out_size, CameraModel &output_model
)
{
  if (validateCameraModel(src_model) != StatusCode::OK)
  {
    return StatusCode::INVALID_MODEL;
  }
  const CameraModel &output_template = options.output_template;
  // An image needs two pixel centres along each axis to span the outermost
  // ones that alpha = 0 and 1 refer to.
  const auto spans = [](const ImageSize size) { return size.width >= 2 && size.height >= 2; };
  if (!spans(src_size) || !spans(out_size) || !(options.alpha >= 0.0f && options.alpha <= 1.0f) ||
      output_template.distortion.type != DistortionModelType::NONE ||
      validateCameraModel(output_template) != StatusCode::OK)
  {
    return StatusCode::INVALID_INPUT;
  }

  CameraModel unit_output = output_template;
  unit_output.intrinsics.cx = 0.5f * static_cast<float>(out_size.width - 1);
  unit_output.intrinsics.cy = 0.5f * static_cast<float>(out_size.height - 1);

  // alpha = 0 needs only the inscribed bound and alpha = 1 only the contained
  // one: a source whose far corners the output cannot represent has no
  // contained bound, and that must not block an alpha = 0 rectify.
  const float alpha = options.alpha;
  const bool need_inscribed = alpha < 1.0f;
  const bool need_contained = alpha > 0.0f;
  const detail::RectifyScaleFit fit = detail::fitRectifyScale(
    src_model, src_size, unit_output, out_size, options.src_from_output_rotation,
    options.source_margin_px, options.solver_options, need_inscribed, need_contained
  );
  if ((need_inscribed && !fit.inscribed_valid) || (need_contained && !fit.contained_valid))
  {
    return StatusCode::DOMAIN_ERROR;
  }
  float scale = fit.inscribed_scale;
  if (!need_inscribed)
  {
    scale = fit.contained_scale;
  }
  else if (need_contained)
  {
    scale = (1.0f - alpha) * fit.inscribed_scale + alpha * fit.contained_scale;
  }

  output_model = detail::scaleOutputModel(unit_output, scale);
  if (validateCameraModel(output_model) != StatusCode::OK)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  return StatusCode::OK;
}

}  // namespace

RectifiedOutputModelResult makeRectifiedOutputModel(
  const CameraModel &src_model, const ImageSize src_size, const RectifiedOutputModelOptions &options
)
{
  RectifiedOutputModelResult result{};
  const ImageSize out_size = options.output_size.isValid() ? options.output_size : src_size;
  result.status =
    fitRectifiedOutputModel(src_model, src_size, options, out_size, result.output_model);
  if (result.status != StatusCode::OK)
  {
    result.output_model = CameraModel{};
    return result;
  }

  const FovResult fov = computeFov(result.output_model, out_size, options.solver_options);
  result.horizontal_fov_deg = fov.horizontal_fov_deg;
  result.vertical_fov_deg = fov.vertical_fov_deg;
  result.diagonal_fov_deg = fov.diagonal_fov_deg;

  estimateCoverage(
    src_model, src_size, result.output_model, out_size, options.src_from_output_rotation,
    result.estimated_source_in_bounds_count, result.estimated_total_count
  );
  return result;
}

RectifyRemapResult buildRectifyRemapMap(
  const CameraModel &src_model, const ImageSize src_size,
  const RectifiedOutputModelOptions &options, float *map_x, float *map_y
)
{
  RectifyRemapResult result{};
  if (map_x == nullptr || map_y == nullptr)
  {
    result.remap_result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  const ImageSize out_size = options.output_size.isValid() ? options.output_size : src_size;
  result.output_size = out_size;
  // Same pixel-count guard as buildImageRemapMap, checked before the fit.
  if (static_cast<long long>(out_size.width) * static_cast<long long>(out_size.height) > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    result.remap_result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  const StatusCode fit_status =
    fitRectifiedOutputModel(src_model, src_size, options, out_size, result.output_model);
  if (fit_status != StatusCode::OK)
  {
    result.output_model = CameraModel{};
    result.remap_result.status = fit_status;
    return result;
  }

  const FovResult fov = computeFov(result.output_model, out_size, options.solver_options);
  result.horizontal_fov_deg = fov.horizontal_fov_deg;
  result.vertical_fov_deg = fov.vertical_fov_deg;
  result.diagonal_fov_deg = fov.diagonal_fov_deg;

  ImageRemapOptions remap_options{};
  remap_options.src_from_dst_rotation = options.src_from_output_rotation;
  remap_options.require_source_in_bounds = true;
  remap_options.solver_options = options.solver_options;
  result.remap_result = buildImageRemapMap(
    src_model, src_size, result.output_model, out_size, map_x, map_y, remap_options
  );
  return result;
}

}  // namespace camxiom
