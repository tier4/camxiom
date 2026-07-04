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

#include "camxiom/remap_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace camxiom
{
namespace
{

constexpr float kSentinelThreshold = -0.5f;

template <typename PixelT>
inline PixelT sampleNearest(const PixelT *src, int src_w, int /*src_h*/, float u, float v)
{
  const int u_i = static_cast<int>(std::lround(u));
  const int v_i = static_cast<int>(std::lround(v));
  // Caller has already checked bounds.
  return src[v_i * src_w + u_i];
}

// The +1 neighbours are clamped to the image edge so that sample positions
// on the last row/column (e.g. an identity map) stay valid: their fractional
// weights are zero there, so the clamped duplicate never changes the value.
template <typename PixelT>
inline PixelT sampleBilinear(const PixelT *src, int src_w, int src_h, float u, float v);

template <>
inline float sampleBilinear<float>(const float *src, int src_w, int src_h, float u, float v)
{
  const int u0 = static_cast<int>(std::floor(u));
  const int v0 = static_cast<int>(std::floor(v));
  const int u1 = std::min(u0 + 1, src_w - 1);
  const int v1 = std::min(v0 + 1, src_h - 1);
  const float fu = u - static_cast<float>(u0);
  const float fv = v - static_cast<float>(v0);
  const float a = src[v0 * src_w + u0];
  const float b = src[v0 * src_w + u1];
  const float c = src[v1 * src_w + u0];
  const float d = src[v1 * src_w + u1];
  const float ab = a * (1.0f - fu) + b * fu;
  const float cd = c * (1.0f - fu) + d * fu;
  return ab * (1.0f - fv) + cd * fv;
}

template <>
inline std::uint8_t sampleBilinear<std::uint8_t>(
  const std::uint8_t *src, int src_w, int src_h, float u, float v
)
{
  const int u0 = static_cast<int>(std::floor(u));
  const int v0 = static_cast<int>(std::floor(v));
  const int u1 = std::min(u0 + 1, src_w - 1);
  const int v1 = std::min(v0 + 1, src_h - 1);
  const float fu = u - static_cast<float>(u0);
  const float fv = v - static_cast<float>(v0);
  const float a = static_cast<float>(src[v0 * src_w + u0]);
  const float b = static_cast<float>(src[v0 * src_w + u1]);
  const float c = static_cast<float>(src[v1 * src_w + u0]);
  const float d = static_cast<float>(src[v1 * src_w + u1]);
  const float ab = a * (1.0f - fu) + b * fu;
  const float cd = c * (1.0f - fu) + d * fu;
  const float result = ab * (1.0f - fv) + cd * fv;
  const float rounded = std::round(result);
  const float clamped = std::clamp(rounded, 0.0f, 255.0f);
  return static_cast<std::uint8_t>(clamped);
}

}  // anonymous namespace

const char *toString(const InterpolationMode mode)
{
  switch (mode)
  {
    case InterpolationMode::NEAREST:
      return "nearest";
    case InterpolationMode::BILINEAR:
      return "bilinear";
    default:
      return "unknown";
  }
}

template <typename PixelT>
RemapImageResult remapImage(
  const PixelT *src, const ImageSize src_size, const float *map_x, const float *map_y, PixelT *dst,
  const ImageSize dst_size, const InterpolationMode mode, const PixelT fill_value
)
{
  RemapImageResult result{};

  if (src == nullptr || map_x == nullptr || map_y == nullptr || dst == nullptr ||
      !src_size.isValid() || !dst_size.isValid())
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  const int sw = src_size.width;
  const int sh = src_size.height;
  const int dw = dst_size.width;
  const int dh = dst_size.height;

  // Same int-overflow guard as buildRemapMap / buildImageRemapMap: the
  // linear pixel index below must fit in int.
  const long long dst_total_ll = static_cast<long long>(dw) * static_cast<long long>(dh);
  const long long src_total_ll = static_cast<long long>(sw) * static_cast<long long>(sh);
  if (dst_total_ll > static_cast<long long>(std::numeric_limits<int>::max()) || src_total_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  result.total_count = dw * dh;

  // For nearest, after rounding, we need u_i in [0, sw - 1].
  // For bilinear we need 4 neighbours (u0+1, v0+1), so the valid sample range
  // is [0, sw - 1) x [0, sh - 1) (strict less-than upper).
  const float nearest_u_max = static_cast<float>(sw) - 1.0f;
  const float nearest_v_max = static_cast<float>(sh) - 1.0f;
  const float bilin_u_max = static_cast<float>(sw) - 1.0f;
  const float bilin_v_max = static_cast<float>(sh) - 1.0f;

  int valid = 0;
  int border = 0;

  for (int v = 0; v < dh; ++v)
  {
    const int row_offset = v * dw;
    for (int u = 0; u < dw; ++u)
    {
      const int idx = row_offset + u;
      const float fu = map_x[idx];
      const float fv = map_y[idx];

      // Sentinel detection: BOTH map_x and map_y below -0.5f means sentinel.
      if (fu < kSentinelThreshold && fv < kSentinelThreshold)
      {
        dst[idx] = fill_value;
        ++border;
        continue;
      }

      if (mode == InterpolationMode::NEAREST)
      {
        // Need u_i in [0, sw-1] after rounding. The bounds must exclude the
        // exact half-pixel values: std::lround rounds halves away from zero,
        // so lround(-0.5) = -1 and lround(sw-1+0.5) = sw would both index
        // outside the source buffer.
        if (fu <= -0.5f || fv <= -0.5f || fu >= nearest_u_max + 0.5f || fv >= nearest_v_max + 0.5f)
        {
          dst[idx] = fill_value;
          ++border;
          continue;
        }
        dst[idx] = sampleNearest<PixelT>(src, sw, sh, fu, fv);
        ++valid;
      }
      else
      {
        // BILINEAR: the valid sample range is the closed interval
        // [0, sw-1] x [0, sh-1]; the +1 neighbours are edge-clamped inside
        // sampleBilinear, so positions exactly on the last row/column (an
        // identity map's border) resolve to the edge pixel instead of fill.
        if (fu < 0.0f || fv < 0.0f || fu > bilin_u_max || fv > bilin_v_max)
        {
          dst[idx] = fill_value;
          ++border;
          continue;
        }
        dst[idx] = sampleBilinear<PixelT>(src, sw, sh, fu, fv);
        ++valid;
      }
    }
  }

  result.valid_count = valid;
  result.border_count = border;
  result.status = StatusCode::OK;
  return result;
}

// Explicit instantiations.
template RemapImageResult remapImage<std::uint8_t>(
  const std::uint8_t *, ImageSize, const float *, const float *, std::uint8_t *, ImageSize,
  InterpolationMode, std::uint8_t
);

template RemapImageResult remapImage<float>(
  const float *, ImageSize, const float *, const float *, float *, ImageSize, InterpolationMode,
  float
);

}  // namespace camxiom
