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

CameraModel makePinholeModel(float fx, float fy, float cx, float cy)
{
  CameraModel model{};
  model.projection.type = ProjectionModelType::PINHOLE;
  model.projection.theta_max = 1.5707962f;
  model.intrinsics.fx = fx;
  model.intrinsics.fy = fy;
  model.intrinsics.cx = cx;
  model.intrinsics.cy = cy;
  model.intrinsics.skew = 0.0f;
  model.distortion.type = DistortionModelType::NONE;
  model.distortion.space = DistortionSpace::NONE;
  model.distortion.count = 0U;
  model.distortion.coeffs.fill(0.0f);
  model.distortion.is_rational = false;
  model.distortion.has_thin_prism = false;
  model.distortion.has_tilt = false;
  return model;
}

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

// ---------------------------------------------------------------------------
// Non-pinhole rectification: pixel → unit ray for virtual output projections
// ---------------------------------------------------------------------------

// Cylindrical: u → azimuth, v → perspective vertical
// φ = (u - cx) / f,  y_cyl = (v - cy) / f
// ray = normalize(sin(φ), y_cyl, cos(φ))
Eigen::Vector3f cylindricalPixelToRay(float u, float v, float f, float cx, float cy)
{
  const float phi = (u - cx) / f;
  const float y_cyl = (v - cy) / f;
  Eigen::Vector3f dir(std::sin(phi), y_cyl, std::cos(phi));
  return dir.normalized();
}

// Stereographic: r_img → θ = 2*atan(r_img / (2*f))
// Conformal (angle-preserving), good for wide FOV
Eigen::Vector3f stereographicPixelToRay(float u, float v, float f, float cx, float cy)
{
  const float dx = u - cx;
  const float dy = v - cy;
  const float r_img = std::sqrt(dx * dx + dy * dy);
  const float theta = 2.0f * std::atan2(r_img, 2.0f * f);
  if (r_img < 1e-8f)
  {
    return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  }
  const float sin_theta = std::sin(theta);
  return Eigen::Vector3f(sin_theta * dx / r_img, sin_theta * dy / r_img, std::cos(theta));
}

// Longitude-Latitude (equirectangular):
// longitude = (u - cx) * (2π / width),  latitude = (v - cy) * (π / height)
// ray = (cos(lat)*sin(lon), -sin(lat), cos(lat)*cos(lon))
Eigen::Vector3f lonLatPixelToRay(float u, float v, float cx, float cy, float w, float h)
{
  const float lon = (u - cx) / w * (2.0f * detail::kPi);
  const float lat = (v - cy) / h * detail::kPi;
  const float cos_lat = std::cos(lat);
  return Eigen::Vector3f(cos_lat * std::sin(lon), -std::sin(lat), cos_lat * std::cos(lon));
}

// Boundary check for non-pinhole types (same logic, custom pixel→ray)
bool allBoundaryInBoundsNonPinhole(
  const CameraModel &src_model, const ImageSize &src_size, const ImageSize &output_size,
  RectifiedProjectionType proj_type, float focal, const Eigen::Matrix3f &src_from_output_rotation,
  int boundary_sample_count, float margin
)
{
  const float w = static_cast<float>(output_size.width);
  const float h = static_cast<float>(output_size.height);
  const float cx = (w - 1.0f) * 0.5f;
  const float cy = (h - 1.0f) * 0.5f;
  const float sw = static_cast<float>(src_size.width);
  const float sh = static_cast<float>(src_size.height);

  const int n = std::max(boundary_sample_count, 8);
  const int per_edge = n / 4;

  auto pixToRay = [&](float u, float v) -> Eigen::Vector3f {
    switch (proj_type)
    {
      case RectifiedProjectionType::CYLINDRICAL:
        return cylindricalPixelToRay(u, v, focal, cx, cy);
      case RectifiedProjectionType::STEREOGRAPHIC:
        return stereographicPixelToRay(u, v, focal, cx, cy);
      case RectifiedProjectionType::LONGITUDE_LATITUDE:
        return lonLatPixelToRay(u, v, cx, cy, w, h);
      default:
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }
  };

  auto check = [&](float u, float v) -> bool {
    const Eigen::Vector3f ray = pixToRay(u, v);
    const Eigen::Vector3f rotated = src_from_output_rotation * ray;
    const PixelResult px = rayToPixel(src_model, rotated);
    if (px.status != StatusCode::OK) return false;
    return px.pixel.u >= margin && px.pixel.v >= margin && px.pixel.u <= (sw - 1.0f - margin) &&
           px.pixel.v <= (sh - 1.0f - margin);
  };

  for (int i = 0; i <= per_edge; ++i)
  {
    const float u =
      (per_edge > 0) ? (w - 1.0f) * static_cast<float>(i) / static_cast<float>(per_edge) : 0.0f;
    if (!check(u, 0.0f)) return false;
    if (!check(u, h - 1.0f)) return false;
  }
  for (int i = 1; i < per_edge; ++i)
  {
    const float v = (h - 1.0f) * static_cast<float>(i) / static_cast<float>(per_edge);
    if (!check(0.0f, v)) return false;
    if (!check(w - 1.0f, v)) return false;
  }
  return true;
}

// Build remap map directly for non-pinhole rectification types
ImageRemapResult buildNonPinholeRectifyMap(
  const CameraModel &src_model, const ImageSize src_size, const ImageSize out_size,
  RectifiedProjectionType proj_type, float focal, const Eigen::Matrix3f &src_from_output_rotation,
  float *map_x, float *map_y
)
{
  const int total = out_size.width * out_size.height;
  const float w = static_cast<float>(out_size.width);
  const float h = static_cast<float>(out_size.height);
  const float cx = (w - 1.0f) * 0.5f;
  const float cy = (h - 1.0f) * 0.5f;
  const float src_w = static_cast<float>(src_size.width);
  const float src_h = static_cast<float>(src_size.height);
  const bool is_identity = src_from_output_rotation.isApprox(Eigen::Matrix3f::Identity(), 1e-6f);

  int model_valid = 0;
  int in_bounds = 0;

#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : model_valid, in_bounds) schedule(static)
#endif
  for (int v = 0; v < out_size.height; ++v)
  {
    const int row = v * out_size.width;
    for (int u = 0; u < out_size.width; ++u)
    {
      const int idx = row + u;
      const float uf = static_cast<float>(u);
      const float vf = static_cast<float>(v);

      Eigen::Vector3f ray;
      switch (proj_type)
      {
        case RectifiedProjectionType::CYLINDRICAL:
          ray = cylindricalPixelToRay(uf, vf, focal, cx, cy);
          break;
        case RectifiedProjectionType::STEREOGRAPHIC:
          ray = stereographicPixelToRay(uf, vf, focal, cx, cy);
          break;
        case RectifiedProjectionType::LONGITUDE_LATITUDE:
          ray = lonLatPixelToRay(uf, vf, cx, cy, w, h);
          break;
        default:
          map_x[idx] = -1.0f;
          map_y[idx] = -1.0f;
          continue;
      }

      if (!is_identity) ray = src_from_output_rotation * ray;

      const PixelResult px = rayToPixel(src_model, ray);
      if (px.status != StatusCode::OK)
      {
        map_x[idx] = -1.0f;
        map_y[idx] = -1.0f;
        continue;
      }

      ++model_valid;
      if (px.pixel.u >= 0.0f && px.pixel.v >= 0.0f && px.pixel.u < src_w && px.pixel.v < src_h)
      {
        ++in_bounds;
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

}  // namespace

const char *toString(const RectifiedProjectionType projection_type)
{
  switch (projection_type)
  {
    case RectifiedProjectionType::PINHOLE:
      return "pinhole";
    case RectifiedProjectionType::CYLINDRICAL:
      return "cylindrical";
    case RectifiedProjectionType::STEREOGRAPHIC:
      return "stereographic";
    case RectifiedProjectionType::LONGITUDE_LATITUDE:
      return "longitude_latitude";
    default:
      return "unknown";
  }
}

const char *toString(const RectifyFitPolicy fit_policy)
{
  switch (fit_policy)
  {
    case RectifyFitPolicy::MAX_INSCRIBED_VALID:
      return "max_inscribed_valid";
    case RectifyFitPolicy::ALLOW_INVALID_BORDER:
      return "allow_invalid_border";
    case RectifyFitPolicy::ALL_SOURCE_CONTAINED:
      return "all_source_contained";
    default:
      return "unknown";
  }
}

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

  const StatusCode src_valid = validateCameraModel(src_model);
  const StatusCode dst_valid = validateCameraModel(dst_model);
  if (src_valid != StatusCode::OK || dst_valid != StatusCode::OK)
  {
    for (int i = 0; i < total; ++i)
    {
      map_x[i] = -1.0f;
      map_y[i] = -1.0f;
    }
    return ImageRemapResult{(src_valid != StatusCode::OK) ? src_valid : dst_valid, 0, 0, total};
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

      const RayResult ray_result = pixelToRay(
        dst_model, Pixel2{static_cast<float>(u), static_cast<float>(v)}, options.solver_options
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

      const PixelResult px = rayToPixel(src_model, ray_src);
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

RectifiedOutputModelResult makeRectifiedOutputModel(
  const CameraModel &src_model, const ImageSize src_size, const RectifiedOutputModelOptions &options
)
{
  RectifiedOutputModelResult result{};

  if (options.projection_type != RectifiedProjectionType::PINHOLE)
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  if (validateCameraModel(src_model) != StatusCode::OK)
  {
    result.status = StatusCode::INVALID_MODEL;
    return result;
  }
  if (!src_size.isValid())
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  // 1. Determine output size
  const ImageSize out_size = options.output_size.isValid() ? options.output_size : src_size;

  const float cx = static_cast<float>(out_size.width - 1) * 0.5f;
  const float cy = static_cast<float>(out_size.height - 1) * 0.5f;

  // 2. Compute per-axis fx_base / fy_base via a single source-boundary pass
  //    (D45). The helper returns both inscribed and circumscribed values; we
  //    pick the relevant pair based on fit_policy.
  const detail::RectifyFocalPairs pairs = detail::computeRectifyFocalPairs(
    src_model, src_size, out_size, options.src_from_output_rotation, options.boundary_sample_count,
    options.source_margin_px
  );

  float fx_base = 0.0f;
  float fy_base = 0.0f;

  switch (options.fit_policy)
  {
    case RectifyFitPolicy::MAX_INSCRIBED_VALID:
    case RectifyFitPolicy::ALLOW_INVALID_BORDER:
      if (!pairs.inscribed_valid)
      {
        result.status = StatusCode::DOMAIN_ERROR;
        return result;
      }
      fx_base = pairs.fx_inscribed;
      fy_base = pairs.fy_inscribed;
      break;
    case RectifyFitPolicy::ALL_SOURCE_CONTAINED:
      if (!pairs.circumscribed_valid)
      {
        result.status = StatusCode::DOMAIN_ERROR;
        return result;
      }
      fx_base = pairs.fx_circumscribed;
      fy_base = pairs.fy_circumscribed;
      break;
  }

  // 3. Apply focal_scale (per-axis)
  float fx_final = fx_base;
  float fy_final = fy_base;
  if (options.fit_policy == RectifyFitPolicy::ALLOW_INVALID_BORDER || options.fit_policy == RectifyFitPolicy::ALL_SOURCE_CONTAINED)
  {
    // ALLOW_INVALID_BORDER / ALL_SOURCE_CONTAINED: focal_scale acts as-is.
    fx_final = fx_base * options.focal_scale;
    fy_final = fy_base * options.focal_scale;
  }
  else
  {
    // MAX_INSCRIBED_VALID: focal_scale >= 1.0 zooms in (always valid)
    fx_final = fx_base * std::max(options.focal_scale, 1.0f);
    fy_final = fy_base * std::max(options.focal_scale, 1.0f);
  }

  // 4. Build output model
  result.output_model = makePinholeModel(fx_final, fy_final, cx, cy);

  // 5. Compute FOV info
  const float w_f = static_cast<float>(out_size.width);
  const float h_f = static_cast<float>(out_size.height);
  result.horizontal_fov_deg = computeFovDeg(result.output_model, 0.0f, cy, w_f - 1.0f, cy);
  result.vertical_fov_deg = computeFovDeg(result.output_model, cx, 0.0f, cx, h_f - 1.0f);
  result.diagonal_fov_deg = computeFovDeg(result.output_model, 0.0f, 0.0f, w_f - 1.0f, h_f - 1.0f);

  // 6. Coarse grid coverage estimation
  estimateCoverage(
    src_model, src_size, result.output_model, out_size, options.src_from_output_rotation,
    result.estimated_source_in_bounds_count, result.estimated_total_count
  );

  result.status = StatusCode::OK;
  return result;
}

RectifyRemapResult buildRectifyRemapMap(
  const CameraModel &src_model, const ImageSize src_size,
  const RectifiedOutputModelOptions &options, float *map_x, float *map_y
)
{
  RectifyRemapResult result{};

  if (validateCameraModel(src_model) != StatusCode::OK)
  {
    result.remap_result.status = StatusCode::INVALID_MODEL;
    return result;
  }
  if (!src_size.isValid() || map_x == nullptr || map_y == nullptr)
  {
    result.remap_result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  const ImageSize out_size = options.output_size.isValid() ? options.output_size : src_size;
  result.output_size = out_size;

  // Int-overflow guard for the linear pixel index (same as buildRemapMap /
  // buildImageRemapMap). The pinhole path re-checks inside buildImageRemapMap,
  // but the non-pinhole path indexes out_size directly.
  const long long out_total_ll =
    static_cast<long long>(out_size.width) * static_cast<long long>(out_size.height);
  if (out_total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    result.remap_result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  // ---- PINHOLE path: delegate to makeRectifiedOutputModel + buildImageRemapMap ----
  if (options.projection_type == RectifiedProjectionType::PINHOLE)
  {
    const RectifiedOutputModelResult model_result =
      makeRectifiedOutputModel(src_model, src_size, options);

    if (model_result.status != StatusCode::OK)
    {
      result.remap_result.status = model_result.status;
      return result;
    }

    result.output_model = model_result.output_model;
    result.horizontal_fov_deg = model_result.horizontal_fov_deg;
    result.vertical_fov_deg = model_result.vertical_fov_deg;
    result.diagonal_fov_deg = model_result.diagonal_fov_deg;

    ImageRemapOptions remap_options{};
    remap_options.src_from_dst_rotation = options.src_from_output_rotation;
    remap_options.require_source_in_bounds = true;

    result.remap_result = buildImageRemapMap(
      src_model, src_size, model_result.output_model, out_size, map_x, map_y, remap_options
    );

    return result;
  }

  // ---- Non-pinhole path: CYLINDRICAL / STEREOGRAPHIC / LONGITUDE_LATITUDE ----
  // These projections are not representable as CameraModel, so we build the
  // remap map directly with custom pixel→ray functions.
  // output_model is left default-constructed (UNKNOWN).

  const float max_dim = static_cast<float>(std::max(out_size.width, out_size.height));

  // LONGITUDE_LATITUDE doesn't use focal — it maps the full sphere.
  // Skip binary search; directly build the remap.
  if (options.projection_type == RectifiedProjectionType::LONGITUDE_LATITUDE)
  {
    result.remap_result = buildNonPinholeRectifyMap(
      src_model, src_size, out_size, options.projection_type, 0.0f,
      options.src_from_output_rotation, map_x, map_y
    );
    return result;
  }

  // CYLINDRICAL / STEREOGRAPHIC: binary search for focal length
  float f_lo = max_dim * 0.02f;
  float f_hi = max_dim * 10.0f;

  {
    if (!allBoundaryInBoundsNonPinhole(
          src_model, src_size, out_size, options.projection_type, f_hi,
          options.src_from_output_rotation, options.boundary_sample_count, options.source_margin_px
        ))
    {
      result.remap_result.status = StatusCode::DOMAIN_ERROR;
      return result;
    }
  }

  constexpr int kMaxIter = 64;
  for (int iter = 0; iter < kMaxIter; ++iter)
  {
    if ((f_hi - f_lo) < 0.1f) break;
    const float f_mid = (f_lo + f_hi) * 0.5f;
    if (allBoundaryInBoundsNonPinhole(
          src_model, src_size, out_size, options.projection_type, f_mid,
          options.src_from_output_rotation, options.boundary_sample_count, options.source_margin_px
        ))
    {
      f_hi = f_mid;
    }
    else
    {
      f_lo = f_mid;
    }
  }

  float f_final = f_hi;
  if (options.fit_policy == RectifyFitPolicy::ALLOW_INVALID_BORDER)
  {
    f_final = f_hi * options.focal_scale;
  }
  else
  {
    f_final = f_hi * std::max(options.focal_scale, 1.0f);
  }

  result.remap_result = buildNonPinholeRectifyMap(
    src_model, src_size, out_size, options.projection_type, f_final,
    options.src_from_output_rotation, map_x, map_y
  );

  return result;
}

// ===========================================================================
// Alpha-blended rectification (D36 / MS2-1c)
// ===========================================================================

BuildRectifyMapResult buildRectifyMap(
  const CameraModel &src_model, const ImageSize src_size, const ImageSize dst_size, float alpha,
  float *map_x, float *map_y, const BuildRectifyMapOptions &options
)
{
  BuildRectifyMapResult result{};

  // Clamp alpha to [0, 1].  Echo the clamped value back.
  if (alpha < 0.0f) alpha = 0.0f;
  if (alpha > 1.0f) alpha = 1.0f;
  result.alpha_used = alpha;

  // Validation.
  if (validateCameraModel(src_model) != StatusCode::OK)
  {
    result.remap_result.status = StatusCode::INVALID_MODEL;
    return result;
  }
  if (!src_size.isValid() || !dst_size.isValid() || map_x == nullptr || map_y == nullptr)
  {
    result.remap_result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  // Single source-boundary pass producing both inscribed and circumscribed
  // per-axis focal estimates (D45).
  const detail::RectifyFocalPairs pairs = detail::computeRectifyFocalPairs(
    src_model, src_size, dst_size, options.src_from_output_rotation, options.boundary_sample_count,
    options.source_margin_px
  );

  // Require only the bound the chosen alpha actually uses: inscribed for alpha
  // < 1 (the alpha = 0 preview path), circumscribed for alpha > 0. A folding
  // source legitimately has no circumscribed bound (its far corners are not
  // representable), so an alpha = 0 rectify must not be blocked by it.
  const bool need_inscribed = alpha < 1.0f;
  const bool need_circumscribed = alpha > 0.0f;
  if ((need_inscribed && !pairs.inscribed_valid) || (need_circumscribed && !pairs.circumscribed_valid))
  {
    result.remap_result.status = StatusCode::DOMAIN_ERROR;
    return result;
  }

  // Linear per-axis focal interpolation (D45 generalisation of D36).
  const float fx_final = (1.0f - alpha) * pairs.fx_inscribed + alpha * pairs.fx_circumscribed;
  const float fy_final = (1.0f - alpha) * pairs.fy_inscribed + alpha * pairs.fy_circumscribed;

  const float w_f = static_cast<float>(dst_size.width);
  const float h_f = static_cast<float>(dst_size.height);
  const float cx_f = (w_f - 1.0f) * 0.5f;
  const float cy_f = (h_f - 1.0f) * 0.5f;

  result.output_model = makePinholeModel(fx_final, fy_final, cx_f, cy_f);

  // Recompute FOV with the interpolated focals.
  result.horizontal_fov_deg = computeFovDeg(result.output_model, 0.0f, cy_f, w_f - 1.0f, cy_f);
  result.vertical_fov_deg = computeFovDeg(result.output_model, cx_f, 0.0f, cx_f, h_f - 1.0f);
  result.diagonal_fov_deg = computeFovDeg(result.output_model, 0.0f, 0.0f, w_f - 1.0f, h_f - 1.0f);

  // Build remap map.
  ImageRemapOptions remap_options{};
  remap_options.src_from_dst_rotation = options.src_from_output_rotation;
  remap_options.require_source_in_bounds = true;
  remap_options.solver_options = options.solver_options;

  result.remap_result = buildImageRemapMap(
    src_model, src_size, result.output_model, dst_size, map_x, map_y, remap_options
  );

  return result;
}

}  // namespace camxiom
