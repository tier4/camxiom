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

// Unit tests for remap.hpp / remap.cpp (established as safety net for MS2-1b/c).
//
// Strategy: build synthetic camera models with known properties, call the
// buildRectifyRemapMap / makeRectifiedOutputModel APIs, and verify invariants
// (output focal finiteness, boundary fit, monotone focal-scale relationship,
// all map values finite, correct rejection of invalid input).
//
// These are NOT algorithm-correctness tests in the strict sense; they are a
// safety-net that will catch regressions when MS2-1b/c modifies remap.cpp.

#include "camxiom/remap.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace camxiom;

namespace
{

// ---------------------------------------------------------------------------
// Camera model factory helpers (adapted from projection_smoke_test.cpp)
// ---------------------------------------------------------------------------

CameraModel makePinholeNoDistortion()
{
  CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 319.5f;
  m.intrinsics.cy = 239.5f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = 1.5707962f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makePinholeRadTan()
{
  CameraModel m = makePinholeNoDistortion();
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.2f;  // k1 (barrel)
  m.distortion.coeffs[1] = 0.05f;  // k2
  m.distortion.coeffs[2] = 0.0f;   // p1
  m.distortion.coeffs[3] = 0.0f;   // p2
  m.distortion.coeffs[4] = 0.0f;   // k3
  m.distortion.count = 5U;
  return m;
}

// KB4 fisheye with explicit theta_max
CameraModel makeKB4(float theta_max = 1.0f)
{
  CameraModel m;
  m.intrinsics.fx = 300.0f;
  m.intrinsics.fy = 300.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = theta_max;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = 0.01f;   // k1
  m.distortion.coeffs[1] = 0.001f;  // k2
  m.distortion.coeffs[2] = 0.0f;    // k3
  m.distortion.coeffs[3] = 0.0f;    // k4
  m.distortion.count = 4U;
  return m;
}

// MEI omnidirectional, xi=1.0
CameraModel makeMEIOmni()
{
  CameraModel m;
  m.intrinsics.fx = 200.0f;
  m.intrinsics.fy = 200.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 1.0f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

// Double-sphere, xi=-0.2, alpha=0.5
CameraModel makeDoubleSphere()
{
  CameraModel m;
  m.intrinsics.fx = 200.0f;
  m.intrinsics.fy = 200.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = -0.2f;
  m.projection.alpha = 0.5f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

// EUCM, alpha=0.5, beta=1.0
CameraModel makeEUCM()
{
  CameraModel m;
  m.intrinsics.fx = 200.0f;
  m.intrinsics.fy = 200.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.alpha = 0.5f;
  m.projection.beta = 1.0f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

// Allocate and zero-fill map buffers for a given image size.
// Returns vector storage; caller keeps them alive.
void allocMaps(int w, int h, std::vector<float> &mx, std::vector<float> &my)
{
  mx.assign(static_cast<std::size_t>(w * h), 0.0f);
  my.assign(static_cast<std::size_t>(w * h), 0.0f);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: PinholeBasicNoDistortion
//
// Pinhole with no distortion, output_size = src_size, focal_scale = 1.0.
// K_new.fx should be close to source K.fx within 1 % — small drift is
// acceptable because the binary search finds the maximum inscribed focal,
// which for a distortion-free pinhole source converges very close to the
// source focal.
// ---------------------------------------------------------------------------
TEST(Rectify, PinholeBasicNoDistortion)
{
  const CameraModel src = makePinholeNoDistortion();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);

  const float fx_new = res.output_model.intrinsics.fx;
  const float fy_new = res.output_model.intrinsics.fy;
  ASSERT_TRUE(std::isfinite(fx_new));
  ASSERT_TRUE(std::isfinite(fy_new));

  // D45 semantics: fx and fy are independent. For a centered, square-pixel
  // source they are similar but the source_margin_px=1 affects horizontal
  // vs vertical proportionally differently. Verify each axis is within
  // 1% of the source focal individually.
  const float fx_rel_err = std::abs(fx_new - src.intrinsics.fx) / src.intrinsics.fx;
  const float fy_rel_err = std::abs(fy_new - src.intrinsics.fy) / src.intrinsics.fy;
  EXPECT_LT(fx_rel_err, 0.01f) << "fx_new=" << fx_new << " src.fx=" << src.intrinsics.fx;
  EXPECT_LT(fy_rel_err, 0.01f) << "fy_new=" << fy_new << " src.fy=" << src.intrinsics.fy;
}

// ---------------------------------------------------------------------------
// Test 2: PinholeWithRadTan
//
// Pinhole with barrel distortion (k1=-0.2). Verifies that buildRectifyRemapMap
// succeeds and returns a finite, plausible K_new. The direction of the focal
// shift relative to the source focal depends on the distortion magnitude and
// the output image geometry, so we only check finiteness and range here.
// The precise value is verified indirectly by the BoundaryFitInvariant test.
// ---------------------------------------------------------------------------
TEST(Rectify, PinholeWithRadTan)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new)) << "K_new.fx not finite";

  // Plausible range: 50 to 5000 px.
  // Tolerance note: the binary-search MIN focal is model-dependent; for
  // a 640×480 source the result landed at ~478 px in practice.
  EXPECT_GT(fx_new, 50.0f) << "K_new.fx=" << fx_new << " suspiciously small";
  EXPECT_LT(fx_new, 5000.0f) << "K_new.fx=" << fx_new << " suspiciously large";
}

// ---------------------------------------------------------------------------
// Test 3: FisheyeKB4
//
// KB4 fisheye, theta_max=1.0 rad (≈57.3°). Verifies basic correctness:
// OK status, finite focal in plausible range, and boundary fit invariant.
// ---------------------------------------------------------------------------
TEST(Rectify, FisheyeKB4)
{
  const CameraModel src = makeKB4(1.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyRemapMap failed: " << toString(res.remap_result.status);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new)) << "K_new.fx is not finite";

  // Plausible range: 50 to 5000 px (generous, model-independent sanity).
  EXPECT_GT(fx_new, 50.0f) << "K_new.fx=" << fx_new << " suspiciously small";
  EXPECT_LT(fx_new, 5000.0f) << "K_new.fx=" << fx_new << " suspiciously large";

  // Boundary fit: every output corner must map to a valid source pixel.
  // We verify the 4 corners manually via the output model.
  const CameraModel &out_m = res.output_model;
  const float w = static_cast<float>(sz.width);
  const float h = static_cast<float>(sz.height);
  const float margin = 1.0f;  // matches default source_margin_px

  auto checkCorner = [&](float u, float v, const char *label) {
    const RayResult ray = pixelToRay(out_m, Pixel2{u, v});
    ASSERT_EQ(ray.status, StatusCode::OK) << label << " pixelToRay failed";
    const PixelResult px = rayToPixel(src, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK) << label << " rayToPixel failed";
    EXPECT_GE(px.pixel.u, margin) << label << " u below margin";
    EXPECT_GE(px.pixel.v, margin) << label << " v below margin";
    EXPECT_LE(px.pixel.u, w - 1.0f - margin) << label << " u above margin";
    EXPECT_LE(px.pixel.v, h - 1.0f - margin) << label << " v above margin";
  };

  checkCorner(0.0f, 0.0f, "top-left");
  checkCorner(w - 1.0f, 0.0f, "top-right");
  checkCorner(0.0f, h - 1.0f, "bottom-left");
  checkCorner(w - 1.0f, h - 1.0f, "bottom-right");
}

// ---------------------------------------------------------------------------
// Test 3b: Non-pinhole rectified projections.
//
// CYLINDRICAL / STEREOGRAPHIC / LONGITUDE_LATITUDE are fully implemented in
// buildRectifyRemapMap (including the focal bisection) but had zero test
// coverage. Structural checks per type: the map builds OK, the output model
// is documented to be UNKNOWN (these projections are not representable as a
// CameraModel), a solid majority of output pixels is valid, every valid map
// entry lands inside the source image, and the output centre — the optical
// axis in all three projections — maps to the source principal point.
// ---------------------------------------------------------------------------
TEST(Rectify, NonPinholeProjections)
{
  const CameraModel src = makeKB4(1.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  struct TypeCase
  {
    RectifiedProjectionType type;
    // Minimum fraction (as numerator of /8) of output pixels that must map
    // into the source. CYLINDRICAL / STEREOGRAPHIC use the inscribed focal
    // fit, so most of the output is valid; LONGITUDE_LATITUDE spans the full
    // sphere by construction, so only the source-FOV band is valid (~19%
    // for this KB4 source).
    int min_valid_eighths;
  };
  const TypeCase cases[] = {
    {RectifiedProjectionType::CYLINDRICAL, 6},
    {RectifiedProjectionType::STEREOGRAPHIC, 6},
    {RectifiedProjectionType::LONGITUDE_LATITUDE, 1},
  };

  for (const TypeCase &tc : cases)
  {
    const RectifiedProjectionType type = tc.type;
    std::vector<float> mx;
    std::vector<float> my;
    allocMaps(sz.width, sz.height, mx, my);

    RectifiedOutputModelOptions opts;
    opts.projection_type = type;
    opts.output_size = sz;
    opts.focal_scale = 1.0f;

    const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

    ASSERT_EQ(res.remap_result.status, StatusCode::OK)
      << toString(type) << ": buildRectifyRemapMap failed";
    EXPECT_EQ(res.output_model.projection.type, ProjectionModelType::UNKNOWN)
      << toString(type) << ": non-pinhole outputs are not representable as a CameraModel";

    const int total = sz.width * sz.height;
    EXPECT_EQ(res.remap_result.total_count, total) << toString(type);
    EXPECT_GT(res.remap_result.source_in_bounds_count, (tc.min_valid_eighths * total) / 8)
      << toString(type) << ": too few output pixels map into the source";

    // Every non-sentinel map entry must land inside the source image (the
    // in-bounds contract is the half-open pixel grid [0, w) x [0, h)).
    const float w = static_cast<float>(sz.width);
    const float h = static_cast<float>(sz.height);
    int checked = 0;
    for (int i = 0; i < total; ++i)
    {
      if (mx[static_cast<std::size_t>(i)] < -0.5f && my[static_cast<std::size_t>(i)] < -0.5f)
      {
        continue;  // sentinel (invalid output pixel)
      }
      ASSERT_GE(mx[static_cast<std::size_t>(i)], 0.0f) << toString(type) << " map entry " << i;
      ASSERT_GE(my[static_cast<std::size_t>(i)], 0.0f) << toString(type) << " map entry " << i;
      ASSERT_LT(mx[static_cast<std::size_t>(i)], w) << toString(type) << " map entry " << i;
      ASSERT_LT(my[static_cast<std::size_t>(i)], h) << toString(type) << " map entry " << i;
      ++checked;
    }
    ASSERT_GT(checked, 0) << toString(type);

    // The output-image centre is the optical axis in all three projections,
    // so it must map (nearly) onto the source principal point.
    const int cu = sz.width / 2;
    const int cv = sz.height / 2;
    const std::size_t centre = static_cast<std::size_t>(cv * sz.width + cu);
    EXPECT_NEAR(mx[centre], src.intrinsics.cx, 2.0f)
      << toString(type) << ": centre must map to the principal point";
    EXPECT_NEAR(my[centre], src.intrinsics.cy, 2.0f)
      << toString(type) << ": centre must map to the principal point";
  }
}

// ---------------------------------------------------------------------------
// Test 4: OmnidirectionalMEI
//
// MEI omni with xi=1.0, K=diag(200,200). Same structural checks as Test 3.
// ---------------------------------------------------------------------------
TEST(Rectify, OmnidirectionalMEI)
{
  const CameraModel src = makeMEIOmni();
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyRemapMap failed: " << toString(res.remap_result.status);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new)) << "K_new.fx is not finite";

  // Tolerance note: MEI omni with xi=1.0 is an extremely wide-angle camera
  // (full hemisphere). The MAX_INSCRIBED focal for such a source is very small
  // (empirically ~32 px for a 640×480 output). Use a lower bound of 5.0 px.
  EXPECT_GT(fx_new, 5.0f) << "K_new.fx=" << fx_new << " suspiciously small (expected > 5 px)";
  EXPECT_LT(fx_new, 5000.0f) << "K_new.fx=" << fx_new << " suspiciously large";

  // 4 corners boundary-fit check
  const CameraModel &out_m = res.output_model;
  const float w = static_cast<float>(sz.width);
  const float h = static_cast<float>(sz.height);
  const float margin = 1.0f;

  auto checkCorner = [&](float u, float v, const char *label) {
    const RayResult ray = pixelToRay(out_m, Pixel2{u, v});
    ASSERT_EQ(ray.status, StatusCode::OK) << label << " pixelToRay failed";
    const PixelResult px = rayToPixel(src, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK) << label << " rayToPixel failed";
    EXPECT_GE(px.pixel.u, margin) << label;
    EXPECT_GE(px.pixel.v, margin) << label;
    EXPECT_LE(px.pixel.u, w - 1.0f - margin) << label;
    EXPECT_LE(px.pixel.v, h - 1.0f - margin) << label;
  };

  checkCorner(0.0f, 0.0f, "top-left");
  checkCorner(w - 1.0f, 0.0f, "top-right");
  checkCorner(0.0f, h - 1.0f, "bottom-left");
  checkCorner(w - 1.0f, h - 1.0f, "bottom-right");
}

// ---------------------------------------------------------------------------
// Test 5: DoubleSphere
//
// DS with xi=-0.2, alpha=0.5. Same structural checks as Tests 3-4.
// ---------------------------------------------------------------------------
TEST(Rectify, DoubleSphere)
{
  const CameraModel src = makeDoubleSphere();
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyRemapMap failed: " << toString(res.remap_result.status);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new)) << "K_new.fx is not finite";
  EXPECT_GT(fx_new, 50.0f);
  EXPECT_LT(fx_new, 5000.0f);

  const CameraModel &out_m = res.output_model;
  const float w = static_cast<float>(sz.width);
  const float h = static_cast<float>(sz.height);
  const float margin = 1.0f;

  auto checkCorner = [&](float u, float v, const char *label) {
    const RayResult ray = pixelToRay(out_m, Pixel2{u, v});
    ASSERT_EQ(ray.status, StatusCode::OK) << label << " pixelToRay failed";
    const PixelResult px = rayToPixel(src, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK) << label << " rayToPixel failed";
    EXPECT_GE(px.pixel.u, margin) << label;
    EXPECT_GE(px.pixel.v, margin) << label;
    EXPECT_LE(px.pixel.u, w - 1.0f - margin) << label;
    EXPECT_LE(px.pixel.v, h - 1.0f - margin) << label;
  };

  checkCorner(0.0f, 0.0f, "top-left");
  checkCorner(w - 1.0f, 0.0f, "top-right");
  checkCorner(0.0f, h - 1.0f, "bottom-left");
  checkCorner(w - 1.0f, h - 1.0f, "bottom-right");
}

// ---------------------------------------------------------------------------
// Test 6: EUCM
//
// EUCM with alpha=0.5, beta=1.0. Same structural checks as Tests 3-5.
// ---------------------------------------------------------------------------
TEST(Rectify, EUCM)
{
  const CameraModel src = makeEUCM();
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyRemapMap failed: " << toString(res.remap_result.status);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new)) << "K_new.fx is not finite";
  EXPECT_GT(fx_new, 50.0f);
  EXPECT_LT(fx_new, 5000.0f);

  const CameraModel &out_m = res.output_model;
  const float w = static_cast<float>(sz.width);
  const float h = static_cast<float>(sz.height);
  const float margin = 1.0f;

  auto checkCorner = [&](float u, float v, const char *label) {
    const RayResult ray = pixelToRay(out_m, Pixel2{u, v});
    ASSERT_EQ(ray.status, StatusCode::OK) << label << " pixelToRay failed";
    const PixelResult px = rayToPixel(src, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK) << label << " rayToPixel failed";
    EXPECT_GE(px.pixel.u, margin) << label;
    EXPECT_GE(px.pixel.v, margin) << label;
    EXPECT_LE(px.pixel.u, w - 1.0f - margin) << label;
    EXPECT_LE(px.pixel.v, h - 1.0f - margin) << label;
  };

  checkCorner(0.0f, 0.0f, "top-left");
  checkCorner(w - 1.0f, 0.0f, "top-right");
  checkCorner(0.0f, h - 1.0f, "bottom-left");
  checkCorner(w - 1.0f, h - 1.0f, "bottom-right");
}

// ---------------------------------------------------------------------------
// Test 7: BoundaryFitInvariant
//
// For the pinhole-with-RadTan source, sample 8 boundary points of the output
// image (4 corners + 4 edge midpoints), project each through the output model
// and back to the source, verify all land within [margin, src_dim-1-margin].
//
// This is the core invariant of MAX_INSCRIBED_VALID policy: every boundary
// pixel of the output image is guaranteed to have a valid source pixel.
// ---------------------------------------------------------------------------
TEST(Rectify, BoundaryFitInvariant)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  // Build the output model using makeRectifiedOutputModel directly.
  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;
  opts.source_margin_px = 1.0f;  // default margin

  const RectifiedOutputModelResult model_res = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(model_res.status, StatusCode::OK)
    << "makeRectifiedOutputModel failed: " << toString(model_res.status);

  const CameraModel &out_m = model_res.output_model;
  const float w = static_cast<float>(sz.width);
  const float h = static_cast<float>(sz.height);
  const float sw = static_cast<float>(sz.width);
  const float sh = static_cast<float>(sz.height);
  const float margin = opts.source_margin_px;

  // 8 boundary points: 4 corners + 4 edge midpoints.
  // Tolerance note: margin=1.0 px is the default; all 8 points must satisfy
  // this because the binary-search focal was chosen to guarantee exactly that.
  const std::array<std::pair<float, float>, 8> pts = {{
    {0.0f, 0.0f},          // top-left corner
    {w - 1.0f, 0.0f},      // top-right corner
    {0.0f, h - 1.0f},      // bottom-left corner
    {w - 1.0f, h - 1.0f},  // bottom-right corner
    {w * 0.5f, 0.0f},      // top-edge midpoint
    {w * 0.5f, h - 1.0f},  // bottom-edge midpoint
    {0.0f, h * 0.5f},      // left-edge midpoint
    {w - 1.0f, h * 0.5f},  // right-edge midpoint
  }};

  for (const auto &[u, v] : pts)
  {
    const RayResult ray = pixelToRay(out_m, Pixel2{u, v});
    ASSERT_EQ(ray.status, StatusCode::OK)
      << "pixelToRay failed at output pixel (" << u << "," << v << ")";

    const PixelResult px = rayToPixel(src, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK)
      << "rayToPixel failed at output pixel (" << u << "," << v << ")";

    EXPECT_GE(px.pixel.u, margin) << "src u=" << px.pixel.u << " below margin for output (" << u
                                  << "," << v << ")";
    EXPECT_GE(px.pixel.v, margin) << "src v=" << px.pixel.v << " below margin for output (" << u
                                  << "," << v << ")";
    EXPECT_LE(px.pixel.u, sw - 1.0f - margin)
      << "src u=" << px.pixel.u << " above upper margin for output (" << u << "," << v << ")";
    EXPECT_LE(px.pixel.v, sh - 1.0f - margin)
      << "src v=" << px.pixel.v << " above upper margin for output (" << u << "," << v << ")";
  }
}

// ---------------------------------------------------------------------------
// Test 7b: CircumscribedSmoke (MS2-1b)
//
// Compare ALL_SOURCE_CONTAINED (alpha=1) vs MAX_INSCRIBED_VALID (alpha=0)
// for a barrel-distorted pinhole. The circumscribed focal must be strictly
// smaller (wider FOV) than the inscribed focal so that every source pixel
// fits inside the rectified frame.
// ---------------------------------------------------------------------------
TEST(Rectify, CircumscribedSmoke)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  // alpha=0 baseline (existing MAX_INSCRIBED_VALID)
  RectifiedOutputModelOptions opts_inscribed;
  opts_inscribed.output_size = sz;
  opts_inscribed.fit_policy = RectifyFitPolicy::MAX_INSCRIBED_VALID;
  opts_inscribed.focal_scale = 1.0f;
  const auto res_inscribed = makeRectifiedOutputModel(src, sz, opts_inscribed);
  ASSERT_EQ(res_inscribed.status, StatusCode::OK);

  // alpha=1 (new ALL_SOURCE_CONTAINED)
  RectifiedOutputModelOptions opts_circ;
  opts_circ.output_size = sz;
  opts_circ.fit_policy = RectifyFitPolicy::ALL_SOURCE_CONTAINED;
  opts_circ.focal_scale = 1.0f;
  const auto res_circ = makeRectifiedOutputModel(src, sz, opts_circ);
  ASSERT_EQ(res_circ.status, StatusCode::OK);

  ASSERT_TRUE(std::isfinite(res_circ.output_model.intrinsics.fx));
  ASSERT_TRUE(std::isfinite(res_circ.output_model.intrinsics.fy));
  EXPECT_GT(res_circ.output_model.intrinsics.fx, 0.0f);
  EXPECT_GT(res_circ.output_model.intrinsics.fy, 0.0f);

  // ALL_SOURCE_CONTAINED gives a smaller focal (wider FOV) than MAX_INSCRIBED_VALID
  // on both axes (D45).
  // Tolerance note: for a barrel-distorted pinhole, alpha=1 must zoom out beyond
  // the alpha=0 focal per axis, so f_circ < f_inscribed strictly.
  EXPECT_LT(res_circ.output_model.intrinsics.fx, res_inscribed.output_model.intrinsics.fx)
    << "fx_circumscribed=" << res_circ.output_model.intrinsics.fx
    << " not strictly less than fx_inscribed=" << res_inscribed.output_model.intrinsics.fx;
  EXPECT_LT(res_circ.output_model.intrinsics.fy, res_inscribed.output_model.intrinsics.fy)
    << "fy_circumscribed=" << res_circ.output_model.intrinsics.fy
    << " not strictly less than fy_inscribed=" << res_inscribed.output_model.intrinsics.fy;
}

// ---------------------------------------------------------------------------
// Test 7c-1: AlphaZeroMatchesInscribed (MS2-1c)
//
// alpha=0 must produce the same output focal as the existing
// MAX_INSCRIBED_VALID policy.
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaZeroMatchesInscribed)
{
  // alpha=0 must produce the same output focal as the existing
  // MAX_INSCRIBED_VALID policy.
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  // Inscribed via existing low-level API.
  RectifiedOutputModelOptions opts_in;
  opts_in.output_size = sz;
  opts_in.fit_policy = RectifyFitPolicy::MAX_INSCRIBED_VALID;
  opts_in.focal_scale = 1.0f;
  const auto res_in = makeRectifiedOutputModel(src, sz, opts_in);
  ASSERT_EQ(res_in.status, StatusCode::OK);
  const float f_inscribed = res_in.output_model.intrinsics.fx;

  // alpha=0 via new high-level API.
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const auto res = buildRectifyMap(src, sz, sz, 0.0f, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);
  EXPECT_EQ(res.alpha_used, 0.0f);

  // Tolerance note: alpha=0 path inside buildRectifyMap calls
  // MAX_INSCRIBED_VALID identically.  Difference should be bit-exact.
  EXPECT_NEAR(res.output_model.intrinsics.fx, res_in.output_model.intrinsics.fx, 1e-4f)
    << "alpha=0 fx " << res.output_model.intrinsics.fx << " differs from inscribed fx "
    << res_in.output_model.intrinsics.fx;
  EXPECT_NEAR(res.output_model.intrinsics.fy, res_in.output_model.intrinsics.fy, 1e-4f)
    << "alpha=0 fy " << res.output_model.intrinsics.fy << " differs from inscribed fy "
    << res_in.output_model.intrinsics.fy;
}

// ---------------------------------------------------------------------------
// Test 7c-2: AlphaOneMatchesCircumscribed (MS2-1c, updated for D45)
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaOneMatchesCircumscribed)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  RectifiedOutputModelOptions opts_circ;
  opts_circ.output_size = sz;
  opts_circ.fit_policy = RectifyFitPolicy::ALL_SOURCE_CONTAINED;
  opts_circ.focal_scale = 1.0f;
  const auto res_circ = makeRectifiedOutputModel(src, sz, opts_circ);
  ASSERT_EQ(res_circ.status, StatusCode::OK);

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const auto res = buildRectifyMap(src, sz, sz, 1.0f, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);
  EXPECT_EQ(res.alpha_used, 1.0f);

  EXPECT_NEAR(res.output_model.intrinsics.fx, res_circ.output_model.intrinsics.fx, 1e-4f)
    << "alpha=1 fx " << res.output_model.intrinsics.fx << " differs from circumscribed fx "
    << res_circ.output_model.intrinsics.fx;
  EXPECT_NEAR(res.output_model.intrinsics.fy, res_circ.output_model.intrinsics.fy, 1e-4f)
    << "alpha=1 fy " << res.output_model.intrinsics.fy << " differs from circumscribed fy "
    << res_circ.output_model.intrinsics.fy;
}

// ---------------------------------------------------------------------------
// Test 7c-3: AlphaMonotone (MS2-1c)
//
// alpha=0, 0.5, 1 must give strictly decreasing focals; the 0.5 point must
// sit at the linear interpolation midpoint.
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaMonotone)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const auto r0 = buildRectifyMap(src, sz, sz, 0.0f, mx.data(), my.data());
  ASSERT_EQ(r0.remap_result.status, StatusCode::OK);

  const auto r1 = buildRectifyMap(src, sz, sz, 0.5f, mx.data(), my.data());
  ASSERT_EQ(r1.remap_result.status, StatusCode::OK);

  const auto r2 = buildRectifyMap(src, sz, sz, 1.0f, mx.data(), my.data());
  ASSERT_EQ(r2.remap_result.status, StatusCode::OK);

  // Per-axis monotonicity (D45): both fx and fy must strictly decrease.
  EXPECT_GT(r0.output_model.intrinsics.fx, r1.output_model.intrinsics.fx);
  EXPECT_GT(r1.output_model.intrinsics.fx, r2.output_model.intrinsics.fx);
  EXPECT_GT(r0.output_model.intrinsics.fy, r1.output_model.intrinsics.fy);
  EXPECT_GT(r1.output_model.intrinsics.fy, r2.output_model.intrinsics.fy);

  // Linear interpolation midpoint check per axis.
  // Tolerance note: <= 1% accounts for binary-search quantisation in any
  // upstream estimate; the D45 helper is analytic so error should be near
  // machine epsilon.
  const float fx_mid_expected =
    0.5f * (r0.output_model.intrinsics.fx + r2.output_model.intrinsics.fx);
  const float fy_mid_expected =
    0.5f * (r0.output_model.intrinsics.fy + r2.output_model.intrinsics.fy);
  EXPECT_NEAR(r1.output_model.intrinsics.fx, fx_mid_expected, 0.01f * fx_mid_expected)
    << "f(0.5).fx=" << r1.output_model.intrinsics.fx << " expected " << fx_mid_expected
    << " (linear midpoint of " << r0.output_model.intrinsics.fx << ", "
    << r2.output_model.intrinsics.fx << ")";
  EXPECT_NEAR(r1.output_model.intrinsics.fy, fy_mid_expected, 0.01f * fy_mid_expected)
    << "f(0.5).fy=" << r1.output_model.intrinsics.fy << " expected " << fy_mid_expected
    << " (linear midpoint of " << r0.output_model.intrinsics.fy << ", "
    << r2.output_model.intrinsics.fy << ")";
}

// ---------------------------------------------------------------------------
// Test 7c-4: AlphaClampedOutsideRange (MS2-1c)
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaClampedOutsideRange)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const auto r_neg = buildRectifyMap(src, sz, sz, -0.5f, mx.data(), my.data());
  ASSERT_EQ(r_neg.remap_result.status, StatusCode::OK);
  EXPECT_EQ(r_neg.alpha_used, 0.0f) << "Negative alpha should clamp to 0";

  const auto r_big = buildRectifyMap(src, sz, sz, 1.5f, mx.data(), my.data());
  ASSERT_EQ(r_big.remap_result.status, StatusCode::OK);
  EXPECT_EQ(r_big.alpha_used, 1.0f) << "Alpha > 1 should clamp to 1";
}

// ---------------------------------------------------------------------------
// Test 7c-5: AlphaFisheyeKB4 (MS2-1c)
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaFisheyeKB4)
{
  const CameraModel src = makeKB4(1.0f);
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const auto res = buildRectifyMap(src, sz, sz, 0.5f, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);

  const float fx_new = res.output_model.intrinsics.fx;
  ASSERT_TRUE(std::isfinite(fx_new));
  EXPECT_GT(fx_new, 5.0f);
  EXPECT_LT(fx_new, 5000.0f);

  const int total = res.remap_result.total_count;
  const int model_valid = res.remap_result.model_valid_count;
  ASSERT_GT(total, 0);
  EXPECT_GT(model_valid, 0) << "Fisheye alpha=0.5 produced zero valid pixels";
}

// ---------------------------------------------------------------------------
// Test 7c-6: AlphaInvalidInput (MS2-1c)
// ---------------------------------------------------------------------------
TEST(Rectify, AlphaInvalidInput)
{
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  // (a) nullptr map_x.
  {
    const CameraModel src = makePinholeNoDistortion();
    const auto res = buildRectifyMap(src, sz, sz, 0.5f, nullptr, my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_INPUT);
  }
  // (b) invalid src_size.
  {
    const CameraModel src = makePinholeNoDistortion();
    const ImageSize bad{-1, 480};
    const auto res = buildRectifyMap(src, bad, sz, 0.5f, mx.data(), my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_INPUT);
  }
  // (c) NaN focal in source.
  {
    CameraModel src = makePinholeNoDistortion();
    src.intrinsics.fx = std::numeric_limits<float>::quiet_NaN();
    const auto res = buildRectifyMap(src, sz, sz, 0.5f, mx.data(), my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_MODEL);
  }
}

// ---------------------------------------------------------------------------
// Test 8: FocalScaleEffectMaxInscribed
//
// Call makeRectifiedOutputModel twice: once with focal_scale=1.0 and once with
// focal_scale=1.5. Under MAX_INSCRIBED_VALID, focal_scale >= 1.0 zooms in, so
// K_new(1.5).fx ≈ 1.5 * K_new(1.0).fx within 1%.
// Tolerance note: 1% because binary-search stops when |f_hi - f_lo| < 0.1 px;
// the relative error from discrete search is << 1% for any focal > 50 px.
// ---------------------------------------------------------------------------
TEST(Rectify, FocalScaleEffectMaxInscribed)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.fit_policy = RectifyFitPolicy::MAX_INSCRIBED_VALID;

  opts.focal_scale = 1.0f;
  const RectifiedOutputModelResult res1 = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(res1.status, StatusCode::OK);
  const float fx1 = res1.output_model.intrinsics.fx;

  opts.focal_scale = 1.5f;
  const RectifiedOutputModelResult res2 = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(res2.status, StatusCode::OK);
  const float fx2 = res2.output_model.intrinsics.fx;

  ASSERT_GT(fx1, 0.0f) << "focal_scale=1.0 produced non-positive focal";
  ASSERT_GT(fx2, 0.0f) << "focal_scale=1.5 produced non-positive focal";

  // fx2 should be exactly 1.5 * fx1 (both derived from same f_base).
  // Tolerance note: 1% for binary-search quantisation + float rounding.
  const float rel_err = std::abs(fx2 - 1.5f * fx1) / (1.5f * fx1);
  EXPECT_LT(rel_err, 0.01f) << "focal_scale=1.5 result fx=" << fx2 << " expected ~" << (1.5f * fx1)
                            << " (1.5 * " << fx1 << ")"
                            << " relative error=" << rel_err;
}

// ---------------------------------------------------------------------------
// Test 9: RotationY5Deg
//
// KB4 fisheye source, src_from_output_rotation = 5° around Y axis.
// Verifies buildRectifyRemapMap returns OK and that model_valid_count > 95%
// of total pixels.  This is the stereo-rectify regression baseline — we
// only assert the map builds successfully and is mostly populated.
// ---------------------------------------------------------------------------
TEST(Rectify, RotationY5Deg)
{
  const CameraModel src = makeKB4(1.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const float angle_rad = 5.0f * static_cast<float>(camxiom::constants::kPi) / 180.0f;
  const Eigen::Matrix3f rot =
    Eigen::AngleAxisf(angle_rad, Eigen::Vector3f::UnitY()).toRotationMatrix();

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;
  opts.src_from_output_rotation = rot;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());

  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyRemapMap with Y-rotation failed: " << toString(res.remap_result.status);

  const int total = res.remap_result.total_count;
  const int model_valid = res.remap_result.model_valid_count;

  ASSERT_GT(total, 0) << "total pixel count is zero";

  // Tolerance note: 5° rotation is small; the intersection of the rotated FOV
  // with the source FOV should still cover > 95% of the output pixels.
  EXPECT_GT(model_valid, static_cast<int>(0.95 * total))
    << "model_valid_count=" << model_valid << " total=" << total
    << " ratio=" << (static_cast<float>(model_valid) / static_cast<float>(total));
}

// ---------------------------------------------------------------------------
// Test 10: InvalidInputRejection
//
// Verify that each invalid input case returns the correct StatusCode.
// Subcase (a): map_x == nullptr → INVALID_INPUT
// Subcase (b): dst_size = {-1, 480} → INVALID_INPUT (via buildRectifyRemapMap
//              which checks src_size, not dst_size; use makeRectifiedOutputModel
//              with invalid output_size instead)
// Subcase (c): src CameraModel with fx=NaN → INVALID_MODEL
// Subcase (d): src CameraModel with projection.type=UNKNOWN → INVALID_MODEL
// ---------------------------------------------------------------------------
TEST(Rectify, InvalidInputRejection)
{
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;

  // (a) map_x == nullptr
  {
    const CameraModel src = makePinholeNoDistortion();
    const RectifyRemapResult res =
      buildRectifyRemapMap(src, sz, opts, /*map_x=*/nullptr, my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_INPUT)
      << "nullptr map_x should return INVALID_INPUT, got " << toString(res.remap_result.status);
  }

  // (b) invalid src_size (negative width) passed to buildRectifyRemapMap
  {
    const CameraModel src = makePinholeNoDistortion();
    const ImageSize bad_sz{-1, 480};
    const RectifyRemapResult res = buildRectifyRemapMap(src, bad_sz, opts, mx.data(), my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_INPUT)
      << "Negative src width should return INVALID_INPUT, got "
      << toString(res.remap_result.status);
  }

  // (c) fx = NaN → INVALID_MODEL
  {
    CameraModel src = makePinholeNoDistortion();
    src.intrinsics.fx = std::numeric_limits<float>::quiet_NaN();
    const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_MODEL)
      << "NaN focal should return INVALID_MODEL, got " << toString(res.remap_result.status);
  }

  // (d) projection.type = UNKNOWN → INVALID_MODEL
  {
    CameraModel src = makePinholeNoDistortion();
    src.projection.type = ProjectionModelType::UNKNOWN;
    const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
    EXPECT_EQ(res.remap_result.status, StatusCode::INVALID_MODEL)
      << "UNKNOWN projection type should return INVALID_MODEL, got "
      << toString(res.remap_result.status);
  }
}

// ---------------------------------------------------------------------------
// Test 11b: PinholeRadTanFoldOverInscribed
//
// Regression guard for the forward-only inscribed focal computation introduced
// to handle pinhole + RADTAN5 sources whose distortion polynomial folds over
// near the image corners.
//
// Background: the real-world camera parameters below (960×540, fx≈497,
// k1=-0.347) produce a source model whose raster corners lie past the
// distortion's fold-over radius. The old inverse-based focal search called
// pixelToRay on source boundary pixels; near the fold-over radius the Newton
// solver converges to the WRONG branch (e.g. a left-edge pixel maps to a
// positive normalised-x), which corrupted the inscribed rectangle so that
// buildRectifyMap returned DOMAIN_ERROR. The fixed implementation drives the
// inscribed bound FORWARD only (rayToPixel), never inverting the source, so a
// folding source cannot corrupt it.
//
// Precondition (fold-over demonstration): pixelToRay on a corner pixel must
// return non-OK, confirming that the inverse solver fails there. This is the
// root cause of the original bug.
//
// Regression guard: buildRectifyMap with alpha=0 must return StatusCode::OK.
// Before the fix this returned DOMAIN_ERROR.
//
// Tolerance notes:
//   - focal range 50..2000 is generous; empirically ~380-450 px for this model.
//   - max identity deviation > 10 px is a conservative lower bound; empirically
//     ~109 px (the image needs significant warping to undo barrel distortion).
//   - valid pixel fraction > 90%: the corner-fit guarantees ~100% for alpha=0.
// ---------------------------------------------------------------------------
TEST(Rectify, PinholeRadTanFoldOverInscribed)
{
  // Real 4K plumb_bob parameters scaled by 0.25 -> 960x540.
  // These coefficients are verified to produce fold-over near the corners.
  CameraModel src{};
  src.projection.type = ProjectionModelType::PINHOLE;
  src.projection.theta_max = 1.5707962f;
  src.intrinsics.fx = 496.94f;
  src.intrinsics.fy = 495.44f;
  src.intrinsics.cx = 477.44f;
  src.intrinsics.cy = 249.79f;
  src.intrinsics.skew = 0.0f;
  src.distortion.type = DistortionModelType::RADTAN5;
  src.distortion.space = DistortionSpace::PLANE;
  src.distortion.count = 5U;
  src.distortion.coeffs[0] = -0.347168833f;      // k1 (strong barrel)
  src.distortion.coeffs[1] = 0.1494961828f;      // k2
  src.distortion.coeffs[2] = 0.003684729338f;    // p1
  src.distortion.coeffs[3] = -0.0002720679913f;  // p2
  src.distortion.coeffs[4] = -0.034232568f;      // k3
  // coeffs[5..13] remain zero (default-constructed)
  src.distortion.is_rational = false;
  src.distortion.has_thin_prism = false;
  src.distortion.has_tilt = false;

  ASSERT_EQ(validateCameraModel(src), StatusCode::OK)
    << "Source model must be valid before the test exercises the bug path";

  // Precondition: the bottom-right corner lies past the fold-over radius.
  // pixelToRay at that corner must return a non-OK status (NON_CONVERGED or
  // similar), demonstrating the failure mode that corrupted the old inverse-
  // based inscribed focal search.
  const RayResult corner_ray = pixelToRay(src, Pixel2{959.f, 539.f});
  EXPECT_NE(corner_ray.status, StatusCode::OK)
    << "Expected pixelToRay to fail at the bottom-right corner (fold-over "
    << "radius), but got OK. The model may no longer exhibit the fold-over; "
    << "if the camera parameters changed, re-verify the precondition.";

  // Build the rectify map (alpha=0 = fully inscribed).
  const ImageSize sz{960, 540};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const BuildRectifyMapResult res =
    buildRectifyMap(src, sz, sz, /*alpha=*/0.0f, mx.data(), my.data());

  // Regression guard: this returned DOMAIN_ERROR before the fix.
  // Tolerance note: DOMAIN_ERROR was triggered because the old inverse path
  // produced a bogus inscribed rectangle with negative or zero width/height.
  ASSERT_EQ(res.remap_result.status, StatusCode::OK)
    << "buildRectifyMap returned " << toString(res.remap_result.status)
    << " — DOMAIN_ERROR here is the regression being guarded";

  // Output model must be a sane distortion-free PINHOLE.
  EXPECT_EQ(res.output_model.projection.type, ProjectionModelType::PINHOLE)
    << "Rectified output model should be PINHOLE";
  const float fx_new = res.output_model.intrinsics.fx;
  const float fy_new = res.output_model.intrinsics.fy;
  ASSERT_TRUE(std::isfinite(fx_new)) << "output fx is not finite";
  ASSERT_TRUE(std::isfinite(fy_new)) << "output fy is not finite";
  // Tolerance note: inscribed focal for a ~60-deg half-FOV pinhole at 960x540
  // is empirically 380-450 px; allow a generous [50, 2000] to be model-robust.
  EXPECT_GT(fx_new, 50.0f) << "output fx=" << fx_new << " suspiciously small";
  EXPECT_LT(fx_new, 2000.0f) << "output fx=" << fx_new << " suspiciously large";
  EXPECT_GT(fy_new, 50.0f) << "output fy=" << fy_new << " suspiciously small";
  EXPECT_LT(fy_new, 2000.0f) << "output fy=" << fy_new << " suspiciously large";

  // The rectification map must warp the image (not be identity).
  // We scan all valid entries (map value >= -0.5) and find the maximum
  // deviation from the identity map (map_x[v*w+u] == u, map_y[v*w+u] == v).
  // Tolerance note: for a barrel-distorted source the remap shifts pixels
  // outward towards the centre; empirically the max shift is ~109 px for this
  // model. A lower bound of 10 px is conservative and model-robust.
  const int w = sz.width;
  const int h = sz.height;
  float max_deviation = 0.0f;
  int valid_count = 0;
  for (int row = 0; row < h; ++row)
  {
    for (int col = 0; col < w; ++col)
    {
      const int idx = row * w + col;
      const float vx = mx[idx];
      const float vy = my[idx];
      if (vx < -0.5f || vy < -0.5f) continue;  // sentinel: skip invalid
      ++valid_count;
      const float du = vx - static_cast<float>(col);
      const float dv = vy - static_cast<float>(row);
      const float dev = std::sqrt(du * du + dv * dv);
      if (dev > max_deviation) max_deviation = dev;
    }
  }

  EXPECT_GT(
    max_deviation, 10.0f
  ) << "Max identity deviation="
    << max_deviation << " px is suspiciously small; the map may be an identity (no warp applied)";

  // Tolerance note: alpha=0 (fully inscribed) guarantees that all output pixels
  // have valid source coordinates, so the valid fraction should be ~100%.
  // Assert > 90% as a robust lower bound.
  const int total = w * h;
  ASSERT_GT(total, 0);
  const float valid_fraction = static_cast<float>(valid_count) / static_cast<float>(total);
  EXPECT_GT(valid_fraction, 0.90f)
    << "valid_count=" << valid_count << " / total=" << total << " (" << (100.0f * valid_fraction)
    << "%) — expected > 90% for alpha=0";
}

// ---------------------------------------------------------------------------
// Test 11: MapValuesAreFinite
//
// For the pinhole-with-RadTan source, build the rectify remap map and verify:
//   - Every non-(-1) entry in map_x and map_y is finite (no NaN / Inf).
//   - Every such entry lies in [-1.0, src_dim] (conservative range sanity).
//
// The -1 sentinel value is produced for out-of-FOV destination pixels and
// source-out-of-bounds pixels; those are skipped.
// ---------------------------------------------------------------------------
TEST(Rectify, MapValuesAreFinite)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};

  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.focal_scale = 1.0f;

  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);

  const float src_w = static_cast<float>(sz.width);
  const float src_h = static_cast<float>(sz.height);
  int invalid_sentinels = 0;
  int valid_checked = 0;

  for (int i = 0; i < sz.width * sz.height; ++i)
  {
    const float vx = mx[i];
    const float vy = my[i];

    if (vx < -0.5f && vy < -0.5f)
    {
      // Sentinel pixel: should be exactly -1.0f (or very close).
      EXPECT_NEAR(vx, -1.0f, 1e-6f) << "map_x sentinel at idx=" << i;
      EXPECT_NEAR(vy, -1.0f, 1e-6f) << "map_y sentinel at idx=" << i;
      ++invalid_sentinels;
      continue;
    }

    // Non-sentinel: must be finite and in plausible range.
    // Tolerance note: [-1.0, src_dim] is a generous range; bilinear
    // interpolation in a remap consumer would clamp or skip out-of-range.
    EXPECT_TRUE(std::isfinite(vx)) << "map_x[" << i << "]=" << vx << " is not finite";
    EXPECT_TRUE(std::isfinite(vy)) << "map_y[" << i << "]=" << vy << " is not finite";
    EXPECT_GE(vx, -1.0f) << "map_x[" << i << "]=" << vx << " below -1";
    EXPECT_LE(vx, src_w) << "map_x[" << i << "]=" << vx << " above src width";
    EXPECT_GE(vy, -1.0f) << "map_y[" << i << "]=" << vy << " below -1";
    EXPECT_LE(vy, src_h) << "map_y[" << i << "]=" << vy << " above src height";
    ++valid_checked;
  }

  // At least some valid pixels must exist.
  EXPECT_GT(valid_checked, 0) << "No valid (non-sentinel) pixels in remap map";

  // For a MAX_INSCRIBED_VALID rectification of a barrel-distorted source,
  // virtually all output pixels should have valid source coordinates.
  // Tolerance note: expect > 90% valid (generous; in practice ~100%).
  const int total = sz.width * sz.height;
  EXPECT_GT(valid_checked, static_cast<int>(0.9 * total))
    << "valid_checked=" << valid_checked << " total=" << total
    << " — fewer than 90% of pixels have valid source coordinates";
}

// ---------------------------------------------------------------------------
// computeFov (public single-model FOV API)
// ---------------------------------------------------------------------------

TEST(ComputeFov, PinholeMatchesAnalyticFov)
{
  const CameraModel m = makePinholeNoDistortion();  // fx=fy=500, cx=319.5, cy=239.5
  const camxiom::FovResult fov = camxiom::computeFov(m, camxiom::ImageSize{640, 480});
  ASSERT_TRUE(fov.ok());

  // Distortion-free pinhole: symmetric around the principal point, so
  // h = 2*atan(cx/fx) etc. (u in [0, w-1], principal point at the center).
  const float h_expected =
    2.0f * std::atan(319.5f / 500.0f) * 180.0f / static_cast<float>(camxiom::constants::kPi);
  const float v_expected =
    2.0f * std::atan(239.5f / 500.0f) * 180.0f / static_cast<float>(camxiom::constants::kPi);
  EXPECT_NEAR(fov.horizontal_fov_deg, h_expected, 0.05f);
  EXPECT_NEAR(fov.vertical_fov_deg, v_expected, 0.05f);
  EXPECT_GT(fov.diagonal_fov_deg, fov.horizontal_fov_deg);
  EXPECT_GT(fov.horizontal_fov_deg, fov.vertical_fov_deg);
}

TEST(ComputeFov, MatchesRectifyReportedFov)
{
  // The rectify APIs fill their *_fov_deg fields with the same measurement;
  // computeFov on the generated output model must agree.
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  RectifiedOutputModelOptions opts{};
  opts.output_size = sz;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);

  const camxiom::FovResult fov = camxiom::computeFov(r.output_model, sz);
  ASSERT_TRUE(fov.ok());
  EXPECT_NEAR(fov.horizontal_fov_deg, r.horizontal_fov_deg, 1e-3f);
  EXPECT_NEAR(fov.vertical_fov_deg, r.vertical_fov_deg, 1e-3f);
  EXPECT_NEAR(fov.diagonal_fov_deg, r.diagonal_fov_deg, 1e-3f);
}

TEST(ComputeFov, WideAngleModelsProduceWideFov)
{
  const camxiom::FovResult kb4 = camxiom::computeFov(makeKB4(1.5f), camxiom::ImageSize{640, 480});
  ASSERT_TRUE(kb4.ok());
  EXPECT_GT(kb4.horizontal_fov_deg, 90.0f);
  EXPECT_TRUE(std::isfinite(kb4.diagonal_fov_deg));

  const camxiom::FovResult omni = camxiom::computeFov(makeMEIOmni(), camxiom::ImageSize{640, 480});
  ASSERT_TRUE(omni.ok());
  EXPECT_GT(omni.horizontal_fov_deg, 90.0f);
}

TEST(ComputeFov, RejectsInvalidInputs)
{
  const CameraModel m = makePinholeNoDistortion();

  const camxiom::FovResult bad_size = camxiom::computeFov(m, camxiom::ImageSize{0, 480});
  EXPECT_EQ(bad_size.status, StatusCode::INVALID_INPUT);
  EXPECT_FALSE(bad_size.ok());

  CameraModel invalid = m;
  invalid.intrinsics.fx = 0.0f;
  const camxiom::FovResult bad_model = camxiom::computeFov(invalid, camxiom::ImageSize{640, 480});
  EXPECT_NE(bad_model.status, StatusCode::OK);
}

TEST(ImageRemap, RejectsInvalidSourceSizeWhenBoundsCheckRequested)
{
  // require_source_in_bounds defaults to true; a default-constructed (0x0)
  // src_size used to silently DISABLE the advertised bounds check and
  // report OK with every model-valid pixel counted as in-bounds. It must be
  // an INVALID_INPUT rejection instead.
  const camxiom::CameraModel src = makePinholeNoDistortion();
  const camxiom::CameraModel dst = makePinholeNoDistortion();
  std::vector<float> map_x(64 * 48);
  std::vector<float> map_y(64 * 48);

  const camxiom::ImageRemapResult rejected = camxiom::buildImageRemapMap(
    src, camxiom::ImageSize{}, dst, camxiom::ImageSize{64, 48}, map_x.data(), map_y.data()
  );
  EXPECT_EQ(rejected.status, StatusCode::INVALID_INPUT);

  // Explicitly opting OUT of the bounds check keeps the permissive path.
  camxiom::ImageRemapOptions no_check;
  no_check.require_source_in_bounds = false;
  const camxiom::ImageRemapResult ok = camxiom::buildImageRemapMap(
    src, camxiom::ImageSize{}, dst, camxiom::ImageSize{64, 48}, map_x.data(), map_y.data(), no_check
  );
  EXPECT_EQ(ok.status, StatusCode::OK);
  EXPECT_GT(ok.model_valid_count, 0);
}
