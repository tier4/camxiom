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

#ifndef CAMXIOM__REMAP_KERNEL_HPP
#define CAMXIOM__REMAP_KERNEL_HPP

#include "camxiom/remap.hpp"  // ImageSize
#include "camxiom/types.hpp"  // StatusCode

#include <cstdint>

namespace camxiom
{

enum class InterpolationMode : std::uint8_t { NEAREST = 0U, BILINEAR };

struct [[nodiscard]] RemapImageResult
{
  StatusCode status{StatusCode::OK};
  int valid_count{0};   // pixels sampled from a valid source location
  int border_count{0};  // pixels filled with fill_value (sentinel or OOB)
  int total_count{0};   // dst_size.width * dst_size.height

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

/// Apply a precomputed (map_x, map_y) to a source image.
///
/// For each output pixel i = v * dst_w + u:
///   - If map_x[i] < -0.5f and map_y[i] < -0.5f -> sentinel -> fill_value
///   - Otherwise compute (u_src, v_src) = (map_x[i], map_y[i]) and sample.
///   - If sampling out of source bounds -> fill_value.
///
/// NEAREST:  round-to-nearest, cast appropriate to PixelT.
/// BILINEAR: 4-tap interpolation in float, cast to PixelT (uint8_t rounded).
///
/// Explicit instantiations: PixelT = std::uint8_t, PixelT = float.
template <typename PixelT>
[[nodiscard]] RemapImageResult remapImage(
  const PixelT *src, ImageSize src_size, const float *map_x, const float *map_y, PixelT *dst,
  ImageSize dst_size, InterpolationMode mode, PixelT fill_value
);

const char *toString(InterpolationMode mode);

}  // namespace camxiom

#endif  // CAMXIOM__REMAP_KERNEL_HPP
