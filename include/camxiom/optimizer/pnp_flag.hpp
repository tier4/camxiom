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

#ifndef CAMXIOM__OPTIMIZER__PNP_FLAG_HPP
#define CAMXIOM__OPTIMIZER__PNP_FLAG_HPP

#include <cstdint>
#include <type_traits>

namespace camxiom::optimizer
{

// ======================== PnpFlag (all camxiom models) ========================
enum class PnpFlag : std::uint64_t {
  NONE = 0,

  FIX_EXTRINSICS = 1ull << 0,
  FIX_FOCAL_LENGTHS = 1ull << 1,
  FIX_PRINCIPAL_POINTS = 1ull << 2,
  FIX_PROJECTION_PARAMS = 1ull << 3,

  FIX_DIST_0 = 1ull << 8,
  FIX_DIST_1 = 1ull << 9,
  FIX_DIST_2 = 1ull << 10,
  FIX_DIST_3 = 1ull << 11,
  FIX_DIST_4 = 1ull << 12,
  FIX_DIST_5 = 1ull << 13,
  FIX_DIST_6 = 1ull << 14,
  FIX_DIST_7 = 1ull << 15,
  FIX_DIST_8 = 1ull << 16,
  FIX_DIST_9 = 1ull << 17,
  FIX_DIST_10 = 1ull << 18,
  FIX_DIST_11 = 1ull << 19,
  FIX_DIST_12 = 1ull << 20,
  FIX_DIST_13 = 1ull << 21,

  FIX_INTRINSICS = FIX_FOCAL_LENGTHS | FIX_PRINCIPAL_POINTS,
  FIX_DISTORTION = FIX_DIST_0 | FIX_DIST_1 | FIX_DIST_2 | FIX_DIST_3 | FIX_DIST_4 | FIX_DIST_5 |
                   FIX_DIST_6 | FIX_DIST_7 | FIX_DIST_8 | FIX_DIST_9 | FIX_DIST_10 | FIX_DIST_11 |
                   FIX_DIST_12 | FIX_DIST_13
};

using PnpFlagUnderlying = std::underlying_type_t<PnpFlag>;
static_assert(std::is_same_v<PnpFlagUnderlying, std::uint64_t>);
constexpr PnpFlagUnderlying toUnderlying(PnpFlag f) { return static_cast<PnpFlagUnderlying>(f); }

constexpr PnpFlag operator|(PnpFlag a, PnpFlag b)
{
  return static_cast<PnpFlag>(toUnderlying(a) | toUnderlying(b));
}
constexpr PnpFlag operator&(PnpFlag a, PnpFlag b)
{
  return static_cast<PnpFlag>(toUnderlying(a) & toUnderlying(b));
}
constexpr PnpFlag &operator|=(PnpFlag &a, PnpFlag b)
{
  a = a | b;
  return a;
}

[[nodiscard]] constexpr bool hasFlag(PnpFlag f, PnpFlag bit)
{
  return (toUnderlying(f) & toUnderlying(bit)) != 0;
}

[[nodiscard]] constexpr bool hasAnyFlag(PnpFlag f, PnpFlag bits)
{
  return (toUnderlying(f) & toUnderlying(bits)) != 0;
}

[[nodiscard]] constexpr bool hasAllFlags(PnpFlag f, PnpFlag bits)
{
  return (toUnderlying(f) & toUnderlying(bits)) == toUnderlying(bits);
}

constexpr void unsetFlags(PnpFlag &f, PnpFlag bits)
{
  f = static_cast<PnpFlag>(toUnderlying(f) & ~toUnderlying(bits));
}

}  // namespace camxiom::optimizer

#endif  // CAMXIOM__OPTIMIZER__PNP_FLAG_HPP
