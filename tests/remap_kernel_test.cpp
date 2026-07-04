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

// Unit tests for remap_kernel.hpp / remap_kernel.cpp (MS2-2).
//
// Verifies the CPU image-warping kernel that applies a precomputed (map_x,
// map_y) to an input image. Covers identity-map round-trip, bilinear half-
// pixel sampling, sentinel/OOB fill behaviour and invalid-input rejection.

#include "camxiom/remap_kernel.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

using namespace camxiom;

namespace
{

// Build an identity map: map_x[v*w + u] = (float)u, map_y = (float)v.
void buildIdentityMap(int w, int h, std::vector<float> &mx, std::vector<float> &my)
{
  mx.assign(static_cast<std::size_t>(w * h), 0.0f);
  my.assign(static_cast<std::size_t>(w * h), 0.0f);
  for (int v = 0; v < h; ++v)
  {
    for (int u = 0; u < w; ++u)
    {
      const int idx = v * w + u;
      mx[idx] = static_cast<float>(u);
      my[idx] = static_cast<float>(v);
    }
  }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: IdentityNearestU8 — identity map + nearest must be bit-exact.
// ---------------------------------------------------------------------------
TEST(RemapKernel, IdentityNearestU8)
{
  const ImageSize sz{8, 6};
  std::vector<std::uint8_t> src(48);
  for (int i = 0; i < 48; ++i)
  {
    src[i] = static_cast<std::uint8_t>(i);
  }

  std::vector<float> mx;
  std::vector<float> my;
  buildIdentityMap(sz.width, sz.height, mx, my);

  std::vector<std::uint8_t> dst(48, 0u);
  const auto res = remapImage<std::uint8_t>(
    src.data(), sz, mx.data(), my.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 48);
  EXPECT_EQ(res.border_count, 0);
  EXPECT_EQ(res.total_count, 48);
  for (int i = 0; i < 48; ++i)
  {
    EXPECT_EQ(dst[i], src[i]) << "idx=" << i;
  }
}

// ---------------------------------------------------------------------------
// Test 2: IdentityNearestF32 — same with float pixels.
// ---------------------------------------------------------------------------
TEST(RemapKernel, IdentityNearestF32)
{
  const ImageSize sz{8, 6};
  std::vector<float> src(48);
  for (int i = 0; i < 48; ++i)
  {
    src[i] = static_cast<float>(i) * 1.5f - 7.3f;
  }

  std::vector<float> mx;
  std::vector<float> my;
  buildIdentityMap(sz.width, sz.height, mx, my);

  std::vector<float> dst(48, 0.0f);
  const auto res = remapImage<float>(
    src.data(), sz, mx.data(), my.data(), dst.data(), sz, InterpolationMode::NEAREST, 0.0f
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 48);
  EXPECT_EQ(res.border_count, 0);
  for (int i = 0; i < 48; ++i)
  {
    EXPECT_EQ(dst[i], src[i]) << "idx=" << i;
  }
}

// ---------------------------------------------------------------------------
// Test 3: BilinearHalfPixelShift — map shifted by (+0.5, +0.5) sampling the
// interior of a known image; verify against analytic 4-tap mean.
// ---------------------------------------------------------------------------
TEST(RemapKernel, BilinearHalfPixelShift)
{
  // 4x4 image with known content.
  const ImageSize sz{4, 4};
  std::vector<float> src{0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                         8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};

  // Output is 3x3, sampling at (u+0.5, v+0.5) for u, v in [0, 2].
  const ImageSize dst_sz{3, 3};
  std::vector<float> mx{0.5f, 1.5f, 2.5f, 0.5f, 1.5f, 2.5f, 0.5f, 1.5f, 2.5f};
  std::vector<float> my{0.5f, 0.5f, 0.5f, 1.5f, 1.5f, 1.5f, 2.5f, 2.5f, 2.5f};
  std::vector<float> dst(9, -100.0f);

  const auto res = remapImage<float>(
    src.data(), sz, mx.data(), my.data(), dst.data(), dst_sz, InterpolationMode::BILINEAR, -1.0f
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 9);
  EXPECT_EQ(res.border_count, 0);

  // Expected: bilinear at (0.5, 0.5) = (0+1+4+5)/4 = 2.5, etc.
  const std::vector<float> expected{2.5f, 3.5f, 4.5f, 6.5f, 7.5f, 8.5f, 10.5f, 11.5f, 12.5f};
  for (int i = 0; i < 9; ++i)
  {
    EXPECT_NEAR(dst[i], expected[i], 1e-5f) << "idx=" << i;
  }
}

// ---------------------------------------------------------------------------
// Test 4: OutOfBoundsAndSentinelFill — sentinel and OOB write fill_value.
// ---------------------------------------------------------------------------
TEST(RemapKernel, OutOfBoundsAndSentinelFill)
{
  const ImageSize sz{4, 4};
  std::vector<std::uint8_t> src(16, 42u);

  const ImageSize dst_sz{3, 1};
  // map[0] = identity, map[1] = sentinel (-1,-1), map[2] = OOB (100,100)
  std::vector<float> mx{1.0f, -1.0f, 100.0f};
  std::vector<float> my{1.0f, -1.0f, 100.0f};
  std::vector<std::uint8_t> dst(3, 0u);

  const auto res = remapImage<std::uint8_t>(
    src.data(), sz, mx.data(), my.data(), dst.data(), dst_sz, InterpolationMode::BILINEAR, 99u
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 1);
  EXPECT_EQ(res.border_count, 2);
  EXPECT_EQ(res.total_count, 3);
  EXPECT_EQ(dst[0], 42u);  // sampled
  EXPECT_EQ(dst[1], 99u);  // sentinel -> fill
  EXPECT_EQ(dst[2], 99u);  // OOB      -> fill
}

// ---------------------------------------------------------------------------
// Test 3b: IdentityBilinear — an identity map must reproduce the image
// INCLUDING the last row and column. The old strict upper bound (< sw-1)
// treated sample positions exactly on the last row/column as border, so an
// identity bilinear remap filled the right/bottom edges with fill_value.
// ---------------------------------------------------------------------------
TEST(RemapKernel, IdentityBilinear)
{
  const ImageSize sz{5, 4};
  std::vector<std::uint8_t> src(20);
  for (std::size_t i = 0; i < src.size(); ++i)
  {
    src[i] = static_cast<std::uint8_t>(10 + 3 * i);
  }

  std::vector<float> mx(20);
  std::vector<float> my(20);
  for (int v = 0; v < 4; ++v)
  {
    for (int u = 0; u < 5; ++u)
    {
      mx[static_cast<std::size_t>(v * 5 + u)] = static_cast<float>(u);
      my[static_cast<std::size_t>(v * 5 + u)] = static_cast<float>(v);
    }
  }
  std::vector<std::uint8_t> dst(20, 0u);

  const auto res = remapImage<std::uint8_t>(
    src.data(), sz, mx.data(), my.data(), dst.data(), sz, InterpolationMode::BILINEAR, 99u
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 20);
  EXPECT_EQ(res.border_count, 0);
  for (std::size_t i = 0; i < src.size(); ++i)
  {
    EXPECT_EQ(dst[i], src[i]) << "pixel " << i;
  }
}

// ---------------------------------------------------------------------------
// Test 4b: NearestHalfPixelBoundary — exact half-pixel map values must fill,
// not read out of bounds. std::lround rounds halves away from zero, so with
// the old open-interval bounds (-0.5, sw-1+0.5) the values -0.5 and 3.5 on a
// 4x4 source produced indices -1 and 4 (heap out-of-bounds read; caught by
// the ASan CI job with this test).
// ---------------------------------------------------------------------------
TEST(RemapKernel, NearestHalfPixelBoundary)
{
  const ImageSize sz{4, 4};
  std::vector<std::uint8_t> src(16, 7u);

  const ImageSize dst_sz{4, 1};
  // map[0] = top-right half-pixel corner, map[1] = negative half-pixel,
  // map[2] = just inside the upper boundary, map[3] = just inside zero.
  std::vector<float> mx{3.5f, -0.5f, 3.49f, -0.49f};
  std::vector<float> my{3.5f, -0.5f, 3.49f, -0.49f};
  std::vector<std::uint8_t> dst(4, 0u);

  const auto res = remapImage<std::uint8_t>(
    src.data(), sz, mx.data(), my.data(), dst.data(), dst_sz, InterpolationMode::NEAREST, 99u
  );

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.valid_count, 2);
  EXPECT_EQ(res.border_count, 2);
  EXPECT_EQ(dst[0], 99u);  // 3.5  -> would round to 4 -> fill
  EXPECT_EQ(dst[1], 99u);  // -0.5 -> would round to -1 -> fill
  EXPECT_EQ(dst[2], 7u);   // 3.49 -> rounds to 3 -> sampled
  EXPECT_EQ(dst[3], 7u);   // -0.49 -> rounds to 0 -> sampled
}

// ---------------------------------------------------------------------------
// Test 5: InvalidInputRejection — null pointers / invalid sizes.
// ---------------------------------------------------------------------------
TEST(RemapKernel, InvalidInputRejection)
{
  const ImageSize sz{4, 4};
  std::vector<std::uint8_t> src(16, 0u);
  std::vector<float> mx(16, 0.0f);
  std::vector<float> my(16, 0.0f);
  std::vector<std::uint8_t> dst(16, 0u);

  // nullptr src
  {
    const auto r = remapImage<std::uint8_t>(
      nullptr, sz, mx.data(), my.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
    );
    EXPECT_EQ(r.status, StatusCode::INVALID_INPUT);
  }
  // nullptr map_x
  {
    const auto r = remapImage<std::uint8_t>(
      src.data(), sz, nullptr, my.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
    );
    EXPECT_EQ(r.status, StatusCode::INVALID_INPUT);
  }
  // Invalid src size
  {
    const ImageSize bad{-1, 4};
    const auto r = remapImage<std::uint8_t>(
      src.data(), bad, mx.data(), my.data(), dst.data(), sz, InterpolationMode::NEAREST, 0u
    );
    EXPECT_EQ(r.status, StatusCode::INVALID_INPUT);
  }
}
