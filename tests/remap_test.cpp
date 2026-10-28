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
// (alpha = 0 leaves no invalid pixel, alpha = 1 contains the source, the
// template's projection and pixel shape survive, the returned model
// reproduces the map, correct rejection of invalid input), and that the fit
// is tight against references computed here from dense samples.

#include "camxiom/remap.hpp"

#include "camxiom/compat.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/types.hpp"
#include "remap_rectify.hpp"  // the fit's internal patch budget

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
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
  m.distortion.count = 0U;
  return m;
}

CameraModel makePinholeRadTan()
{
  CameraModel m = makePinholeNoDistortion();
  m.distortion.type = DistortionModelType::RADTAN5;
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
  m.projection.type = ProjectionModelType::EQUIDISTANT;
  m.projection.theta_max = theta_max;
  m.distortion.type = DistortionModelType::KB4;
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
// Rectification: output template + alpha
// ---------------------------------------------------------------------------

namespace
{

// An output template: the projection with unit focal lengths. theta_max is
// set wide enough for the law to show a whole hemisphere where it can.
CameraModel makeTemplate(const ProjectionModelType type)
{
  CameraModel m;
  m.projection.type = type;
  m.intrinsics.fx = 1.0f;
  m.intrinsics.fy = 1.0f;
  switch (type)
  {
    case ProjectionModelType::STEREOGRAPHIC:
      m.projection.theta_max = 3.0f;
      break;
    case ProjectionModelType::ORTHOGRAPHIC:
      m.projection.theta_max = static_cast<float>(constants::kHalfPi) - 1e-3f;
      break;
    case ProjectionModelType::PINHOLE:
      break;
    case ProjectionModelType::OMNIDIRECTIONAL:
      m.projection.xi = 0.8f;
      m.projection.theta_max = static_cast<float>(constants::kPi);
      break;
    case ProjectionModelType::DOUBLE_SPHERE:
      m.projection.xi = -0.2f;
      m.projection.alpha = 0.5f;
      m.projection.theta_max = static_cast<float>(constants::kPi);
      break;
    case ProjectionModelType::EUCM:
      m.projection.alpha = 0.5f;
      m.projection.beta = 1.0f;
      m.projection.theta_max = static_cast<float>(constants::kPi);
      break;
    default:
      m.projection.theta_max = static_cast<float>(constants::kPi);
      break;
  }
  return m;
}

const ProjectionModelType kTemplateTypes[] = {
  ProjectionModelType::PINHOLE,         ProjectionModelType::EQUIDISTANT,
  ProjectionModelType::EQUISOLID,       ProjectionModelType::STEREOGRAPHIC,
  ProjectionModelType::ORTHOGRAPHIC,    ProjectionModelType::OMNIDIRECTIONAL,
  ProjectionModelType::DOUBLE_SPHERE,   ProjectionModelType::EUCM,
  ProjectionModelType::EQUIRECTANGULAR, ProjectionModelType::CYLINDRICAL};

struct NamedSource
{
  const char *name;
  CameraModel model;
};

std::vector<NamedSource> rectifySources()
{
  return {
    {"pinhole_radtan5", makePinholeRadTan()},
    {"kb4", makeKB4(1.0f)},
    {"double_sphere", makeDoubleSphere()},
    {"eucm", makeEUCM()}};
}

RectifiedOutputModelOptions rectifyOptions(const ProjectionModelType type, const float alpha)
{
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(type);
  opts.output_size = ImageSize{320, 240};
  opts.alpha = alpha;
  return opts;
}

}  // namespace

TEST(Rectify, DefaultTemplateIsASquarePixelPinhole)
{
  const CameraModel src = makePinholeNoDistortion();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);

  const RectifyRemapResult res =
    buildRectifyRemapMap(src, sz, RectifiedOutputModelOptions{}, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);
  const CameraModel &out = res.output_model;
  EXPECT_EQ(out.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(out.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(out.intrinsics.fx, out.intrinsics.fy);
  EXPECT_EQ(out.intrinsics.skew, 0.0f);
  EXPECT_FLOAT_EQ(out.intrinsics.cx, 319.5f);
  EXPECT_FLOAT_EQ(out.intrinsics.cy, 239.5f);
  EXPECT_EQ(res.output_size.width, 640);
  EXPECT_EQ(res.output_size.height, 480);
  // A distortion-free source of the same size: the output is the source
  // narrowed only by the one-pixel source margin.
  EXPECT_NEAR(out.intrinsics.fx, src.intrinsics.fx, 0.01f * src.intrinsics.fx);
  EXPECT_GE(out.intrinsics.fx, src.intrinsics.fx);
  EXPECT_EQ(res.remap_result.source_in_bounds_count, res.remap_result.total_count);
}

// alpha = 0: every output pixel shows source content, for every projection
// the output can take.
TEST(Rectify, AlphaZeroLeavesNoInvalidPixels)
{
  for (const NamedSource &src : rectifySources())
  {
    const ImageSize src_size{640, 480};
    for (const ProjectionModelType type : kTemplateTypes)
    {
      const RectifiedOutputModelOptions opts = rectifyOptions(type, 0.0f);
      std::vector<float> mx, my;
      allocMaps(opts.output_size.width, opts.output_size.height, mx, my);
      const RectifyRemapResult res =
        buildRectifyRemapMap(src.model, src_size, opts, mx.data(), my.data());
      ASSERT_EQ(res.remap_result.status, StatusCode::OK) << src.name << " -> " << toString(type);
      EXPECT_EQ(res.output_model.projection.type, type);
      EXPECT_EQ(res.output_model.intrinsics.fx, res.output_model.intrinsics.fy);
      EXPECT_EQ(res.remap_result.model_valid_count, res.remap_result.total_count)
        << src.name << " -> " << toString(type);
      EXPECT_EQ(res.remap_result.source_in_bounds_count, res.remap_result.total_count)
        << src.name << " -> " << toString(type);
    }
  }
}

// alpha = 1: every source boundary pixel (the outermost pixel centres)
// lands inside the output, and the fit is tight (some boundary pixel reaches
// the output edge).
TEST(Rectify, AlphaOneContainsTheSource)
{
  for (const NamedSource &src : rectifySources())
  {
    const ImageSize src_size{640, 480};
    for (const ProjectionModelType type : kTemplateTypes)
    {
      const RectifiedOutputModelOptions opts = rectifyOptions(type, 1.0f);
      const RectifiedOutputModelResult res = makeRectifiedOutputModel(src.model, src_size, opts);
      ASSERT_EQ(res.status, StatusCode::OK) << src.name << " -> " << toString(type);
      const CameraModel &out = res.output_model;
      const float half_w = 0.5f * static_cast<float>(opts.output_size.width - 1);
      const float half_h = 0.5f * static_cast<float>(opts.output_size.height - 1);

      float reach = 0.0f;
      int skipped = 0;
      for (int i = 0; i <= 64; ++i)
      {
        const float t = static_cast<float>(i) / 64.0f;
        // The outermost pixel centres: alpha = 1 keeps them all.
        const Pixel2 edges[] = {
          {639.0f * t, 0.0f}, {639.0f * t, 479.0f}, {0.0f, 479.0f * t}, {639.0f, 479.0f * t}};
        for (const Pixel2 &p : edges)
        {
          // Skip pixels outside the source's own domain, rays the output
          // cannot represent, and rays within the pinhole horizon margin.
          const RayResult ray = pixelToRay(src.model, p);
          const PixelResult px = ray.status == StatusCode::OK
                                   ? rayToPixel(out, ray.ray.direction)
                                   : PixelResult{StatusCode::OUT_OF_FOV, {}};
          if (px.status != StatusCode::OK ||
              (type == ProjectionModelType::PINHOLE && ray.ray.direction.z() < std::sin(0.05f)))
          {
            ++skipped;
            continue;
          }
          const float du = std::abs(px.pixel.u - out.intrinsics.cx) / half_w;
          const float dv = std::abs(px.pixel.v - out.intrinsics.cy) / half_h;
          EXPECT_LE(du, 1.0f + 1e-4f)
            << src.name << " -> " << toString(type) << " at (" << p.u << ", " << p.v << ")";
          EXPECT_LE(dv, 1.0f + 1e-4f)
            << src.name << " -> " << toString(type) << " at (" << p.u << ", " << p.v << ")";
          reach = std::max(reach, std::max(du, dv));
        }
      }
      // Tight: when the whole source boundary was usable, some of it reaches
      // the output edge. (Otherwise the fit is bound by points pulled inward
      // from the skipped ones, which this scan does not see.)
      if (skipped == 0)
      {
        EXPECT_GT(reach, 0.99f) << src.name << " -> " << toString(type);
      }
    }
  }
}

TEST(Rectify, AlphaInterpolatesOneScale)
{
  const CameraModel src = makeKB4(1.0f);
  const ImageSize sz{640, 480};
  for (const ProjectionModelType type : kTemplateTypes)
  {
    float fx[3] = {};
    const float alphas[3] = {0.0f, 0.5f, 1.0f};
    for (int k = 0; k < 3; ++k)
    {
      const RectifiedOutputModelResult r =
        makeRectifiedOutputModel(src, sz, rectifyOptions(type, alphas[k]));
      ASSERT_EQ(r.status, StatusCode::OK) << toString(type) << " alpha=" << alphas[k];
      fx[k] = r.output_model.intrinsics.fx;
      EXPECT_EQ(r.output_model.intrinsics.fy, fx[k]);
    }
    // The contained view is wider (smaller focal) than the inscribed one,
    // and alpha interpolates the focal linearly.
    EXPECT_LT(fx[2], fx[0]) << toString(type);
    EXPECT_NEAR(fx[1], 0.5f * (fx[0] + fx[2]), 1e-4f * fx[0]) << toString(type);
  }
}

TEST(Rectify, KeepsTheTemplatePixelShape)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  RectifiedOutputModelOptions opts;
  opts.output_template.intrinsics = IntrinsicsModel{2.0f, 2.5f, 123.0f, -7.0f, 0.2f};
  opts.output_size = ImageSize{400, 300};
  for (const float alpha : {0.0f, 1.0f})
  {
    opts.alpha = alpha;
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
    ASSERT_EQ(r.status, StatusCode::OK);
    const IntrinsicsModel &k = r.output_model.intrinsics;
    EXPECT_NEAR(k.fy / k.fx, 1.25f, 1e-6f);
    EXPECT_NEAR(k.skew / k.fx, 0.1f, 1e-6f);
    // The template's principal point is not used: the output is centred.
    EXPECT_FLOAT_EQ(k.cx, 199.5f);
    EXPECT_FLOAT_EQ(k.cy, 149.5f);
  }
}

// The returned model is the whole description of the output: rebuilding the
// map from it reproduces the rectify map exactly.
TEST(Rectify, ReturnedModelReproducesTheMap)
{
  const CameraModel src = makeKB4(1.0f);
  const ImageSize sz{640, 480};
  for (const ProjectionModelType type :
       {ProjectionModelType::PINHOLE, ProjectionModelType::STEREOGRAPHIC,
        ProjectionModelType::CYLINDRICAL, ProjectionModelType::EQUIRECTANGULAR})
  {
    RectifiedOutputModelOptions opts = rectifyOptions(type, 0.5f);
    opts.src_from_output_rotation =
      Eigen::AngleAxisf(0.1f, Eigen::Vector3f(0.3f, 1.0f, 0.2f).normalized()).toRotationMatrix();
    const int n = opts.output_size.width * opts.output_size.height;
    std::vector<float> mx(n), my(n), mx2(n), my2(n);
    const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
    ASSERT_EQ(res.remap_result.status, StatusCode::OK) << toString(type);

    const RectifiedOutputModelResult model_only = makeRectifiedOutputModel(src, sz, opts);
    ASSERT_EQ(model_only.status, StatusCode::OK);
    EXPECT_EQ(model_only.output_model.intrinsics.fx, res.output_model.intrinsics.fx);
    EXPECT_EQ(model_only.horizontal_fov_deg, res.horizontal_fov_deg);

    ImageRemapOptions remap_options;
    remap_options.src_from_dst_rotation = opts.src_from_output_rotation;
    const ImageRemapResult again = buildImageRemapMap(
      src, sz, res.output_model, opts.output_size, mx2.data(), my2.data(), remap_options
    );
    ASSERT_EQ(again.status, StatusCode::OK);
    EXPECT_EQ(again.source_in_bounds_count, res.remap_result.source_in_bounds_count);
    EXPECT_EQ(mx, mx2) << toString(type);
    EXPECT_EQ(my, my2) << toString(type);

    const FovResult fov = computeFov(res.output_model, opts.output_size);
    EXPECT_EQ(fov.horizontal_fov_deg, res.horizontal_fov_deg);
    EXPECT_EQ(fov.vertical_fov_deg, res.vertical_fov_deg);
  }
}

// An equirectangular output is upright: rows above the centre look up and
// sample the upper half of a forward-looking source.
TEST(Rectify, LatLongOutputIsUpright)
{
  const CameraModel src = makeKB4(1.0f);
  const ImageSize sz{640, 480};
  for (const ProjectionModelType type :
       {ProjectionModelType::EQUIRECTANGULAR, ProjectionModelType::CYLINDRICAL})
  {
    const RectifiedOutputModelOptions opts = rectifyOptions(type, 1.0f);
    const int n = opts.output_size.width * opts.output_size.height;
    std::vector<float> mx(n), my(n);
    const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
    ASSERT_EQ(res.remap_result.status, StatusCode::OK);
    const int w = opts.output_size.width;
    const int centre_col = w / 2;
    const float upper = my[static_cast<std::size_t>(60 * w + centre_col)];
    const float lower = my[static_cast<std::size_t>(180 * w + centre_col)];
    ASSERT_GE(upper, 0.0f);
    ASSERT_GE(lower, 0.0f);
    EXPECT_LT(upper, src.intrinsics.cy);
    EXPECT_GT(lower, src.intrinsics.cy);
    // Vertically the source image ends 45.6 degrees above and below the axis
    // (radius 240 px); horizontally its theta_max of 1 rad (57.3 degrees)
    // ends it first. The output holds both: the vertical span binds.
    EXPECT_NEAR(res.vertical_fov_deg, 91.1f, 0.5f) << toString(type);
    EXPECT_GT(res.horizontal_fov_deg, 114.6f) << toString(type);
  }
}

TEST(Rectify, RotatedOutput)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  RectifiedOutputModelOptions opts;
  opts.output_size = sz;
  opts.src_from_output_rotation =
    Eigen::AngleAxisf(5.0f * static_cast<float>(constants::kPi) / 180.0f, Eigen::Vector3f::UnitY())
      .toRotationMatrix();
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);
  EXPECT_EQ(res.remap_result.source_in_bounds_count, res.remap_result.total_count);
}

TEST(Rectify, NoSolutionWhenTheOutputLooksAway)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  for (const float alpha : {0.0f, 1.0f})
  {
    RectifiedOutputModelOptions opts;
    opts.alpha = alpha;
    opts.src_from_output_rotation =
      Eigen::AngleAxisf(static_cast<float>(constants::kPi), Eigen::Vector3f::UnitY())
        .toRotationMatrix();
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
    EXPECT_EQ(r.status, StatusCode::DOMAIN_ERROR) << "alpha=" << alpha;
    EXPECT_EQ(r.output_model.projection.type, ProjectionModelType::UNKNOWN);
  }
}

namespace
{

// Source pixel centres on a `step` lattice (outermost rows and columns
// included) that the output can show at least `margin` from where its
// coordinates run off (pinhole horizon, cylinder poles, stereographic
// antipode) but that land outside the output image: what alpha = 1 promises
// never happens.
int countUncontained(
  const CameraModel &src, const ImageSize &src_size, const CameraModel &out,
  const ImageSize &out_size, const Eigen::Matrix3f &src_from_output, const int step,
  const float margin = 0.05f
)
{
  int outside = 0;
  const auto coords = [step](const int n) {
    std::vector<float> c;
    for (int i = 0; i < n - 1; i += step)
    {
      c.push_back(static_cast<float>(i));
    }
    c.push_back(static_cast<float>(n - 1));
    return c;
  };
  for (const float v : coords(src_size.height))
  {
    for (const float u : coords(src_size.width))
    {
      const RayResult ray = pixelToRay(src, Pixel2{u, v});
      if (ray.status != StatusCode::OK)
      {
        continue;
      }
      const Eigen::Vector3f d = (src_from_output.transpose() * ray.ray.direction).normalized();
      // Rays within `margin` of where the output's coordinates run off or
      // fold together.
      const bool far_side_law = out.projection.type == ProjectionModelType::STEREOGRAPHIC ||
                                out.projection.type == ProjectionModelType::EQUIDISTANT ||
                                out.projection.type == ProjectionModelType::EQUISOLID;
      if ((out.projection.type == ProjectionModelType::PINHOLE && d.z() < std::sin(margin)) ||
          (out.projection.type == ProjectionModelType::CYLINDRICAL &&
           std::abs(d.y()) > std::cos(margin)) ||
          (far_side_law && d.z() < -std::cos(margin)))
      {
        continue;
      }
      const PixelResult px = rayToPixel(out, d);
      if (px.status != StatusCode::OK)
      {
        continue;
      }
      const float tol = 1e-3f;
      if (px.pixel.u < -tol || px.pixel.v < -tol ||
          px.pixel.u > static_cast<float>(out_size.width - 1) + tol ||
          px.pixel.v > static_cast<float>(out_size.height - 1) + tol)
      {
        ++outside;
      }
    }
  }
  return outside;
}

}  // namespace

// A lat-long model as the source, pointed several ways: alpha = 1 keeps every
// source pixel the output can show, including those whose limit lies inside
// the source image rather than on its border, and alpha = 0 leaves no invalid
// pixel. An output centred on the seam (the second rotation looks straight
// back) has no alpha = 0 view: the seam is a cut through the source, and
// every view around it reaches across.
TEST(Rectify, LatLongSourceInAnyDirection)
{
  CameraModel equirect;
  ASSERT_EQ(makeEquirectangularModel(ImageSize{2048, 1024}, equirect), StatusCode::OK);
  const ImageSize equirect_size{2048, 1024};
  const Eigen::Matrix3f rotations[] = {
    Eigen::Matrix3f::Identity(), Eigen::Vector3f(-1.0f, 1.0f, -1.0f).asDiagonal(),
    Eigen::AngleAxisf(0.7f, Eigen::Vector3f(0.3f, 1.0f, -0.2f).normalized()).toRotationMatrix(),
    Eigen::AngleAxisf(1.4f, Eigen::Vector3f::UnitX()).toRotationMatrix()};
  for (const ProjectionModelType type :
       {ProjectionModelType::PINHOLE, ProjectionModelType::EQUIDISTANT,
        ProjectionModelType::CYLINDRICAL})
  {
    for (int k = 0; k < 4; ++k)
    {
      const Eigen::Matrix3f &R = rotations[k];
      RectifiedOutputModelOptions opts;
      opts.output_template = makeTemplate(type);
      opts.output_size = ImageSize{320, 640};
      opts.src_from_output_rotation = R;
      opts.alpha = 1.0f;
      const RectifiedOutputModelResult contained =
        makeRectifiedOutputModel(equirect, equirect_size, opts);
      ASSERT_EQ(contained.status, StatusCode::OK) << toString(type) << " rotation " << k;
      EXPECT_EQ(
        countUncontained(equirect, equirect_size, contained.output_model, opts.output_size, R, 4), 0
      ) << toString(type)
        << " rotation " << k;

      opts.alpha = 0.0f;
      std::vector<float> mx, my;
      allocMaps(opts.output_size.width, opts.output_size.height, mx, my);
      const RectifyRemapResult inscribed =
        buildRectifyRemapMap(equirect, equirect_size, opts, mx.data(), my.data());
      if (k == 1)
      {
        EXPECT_EQ(inscribed.remap_result.status, StatusCode::DOMAIN_ERROR) << toString(type);
        continue;
      }
      ASSERT_EQ(inscribed.remap_result.status, StatusCode::OK)
        << toString(type) << " rotation " << k;
      EXPECT_EQ(inscribed.remap_result.source_in_bounds_count, inscribed.remap_result.total_count)
        << toString(type) << " rotation " << k;
    }
  }
}

// The case where the pinhole output's limit (87 degrees off its axis) runs
// through the inside of the source image: the border alone does not bound the
// scale, and pixels near that limit must still fit.
TEST(Rectify, AlphaOneSeesLimitsInsideTheSource)
{
  CameraModel equirect;
  ASSERT_EQ(makeEquirectangularModel(ImageSize{2048, 1024}, equirect), StatusCode::OK);
  RectifiedOutputModelOptions opts;
  opts.output_size = ImageSize{320, 640};
  opts.alpha = 1.0f;
  opts.src_from_output_rotation = Eigen::Vector3f(-1.0f, 1.0f, -1.0f).asDiagonal();
  const RectifiedOutputModelResult r =
    makeRectifiedOutputModel(equirect, ImageSize{2048, 1024}, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(
    countUncontained(
      equirect, ImageSize{2048, 1024}, r.output_model, opts.output_size,
      opts.src_from_output_rotation, 1
    ),
    0
  );
  // A source pixel about 87 degrees off the output axis, where the old
  // border-only fit let it fall 165 px past the right edge.
  const RayResult ray = pixelToRay(equirect, Pixel2{494.36f, 511.5f});
  ASSERT_EQ(ray.status, StatusCode::OK);
  const PixelResult px =
    rayToPixel(r.output_model, opts.src_from_output_rotation.transpose() * ray.ray.direction);
  ASSERT_EQ(px.status, StatusCode::OK);
  EXPECT_LE(px.pixel.u, 319.0f);
}

// alpha = 1 keeps the outermost source pixel centres whatever the source
// margin (that margin only keeps the alpha = 0 view off the source edge).
TEST(Rectify, AlphaOneKeepsTheOutermostPixels)
{
  const CameraModel src = makePinholeNoDistortion();
  const ImageSize sz{640, 480};
  for (const float margin : {0.0f, 1.0f, 5.0f})
  {
    RectifiedOutputModelOptions opts;
    opts.alpha = 1.0f;
    opts.source_margin_px = margin;
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
    ASSERT_EQ(r.status, StatusCode::OK);
    // (The fit keeps 3e-4 of tolerance and rounding headroom.)
    EXPECT_LE(r.output_model.intrinsics.fx, 500.0f) << margin;
    EXPECT_GT(r.output_model.intrinsics.fx, 499.8f) << margin;
    EXPECT_EQ(countUncontained(src, sz, r.output_model, sz, Eigen::Matrix3f::Identity(), 1), 0)
      << margin;
  }
}

// A radtan source whose polynomial folds over inside the frame (k1 = -0.3:
// the distorted radius peaks at r = 1/sqrt(0.9) = 1.054, 422 px from the
// centre, while the corners lie 734 px out). Pixels past the fold unproject
// onto the folded branch (e.g. the left-edge pixel (4, 354) returns a ray
// 2.2 to the RIGHT of the axis); they are not valid source pixels, and
// alpha = 1 must neither chase them nor drop the valid ones.
TEST(Rectify, AlphaOneStopsAtTheDistortionFold)
{
  CameraModel src = makePinholeNoDistortion();
  src.intrinsics.fx = 600.0f;
  src.intrinsics.fy = 600.0f;
  src.intrinsics.cx = 639.5f;
  src.intrinsics.cy = 359.5f;
  src.distortion.type = DistortionModelType::RADTAN5;
  src.distortion.count = 5U;
  src.distortion.coeffs[0] = -0.3f;
  const ImageSize sz{1280, 720};
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const float r_fold = 1.0f / std::sqrt(0.9f);

  const RayResult folded = pixelToRay(src, Pixel2{4.0f, 354.0f});
  ASSERT_EQ(folded.status, StatusCode::OK);
  EXPECT_GT(folded.ray.direction.x() / folded.ray.direction.z(), r_fold);

  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  const CameraModel &out = r.output_model;

  // Every valid pixel (its ray before the fold) lands inside, and one of
  // them reaches the edge.
  float reach = 0.0f;
  int valid = 0;
  for (int v = 0; v < 720; v += 3)
  {
    for (int u = 0; u < 1280; u += 3)
    {
      const float uu = static_cast<float>(std::min(u, 1279));
      const float vv = static_cast<float>(std::min(v, 719));
      const RayResult ray = pixelToRay(src, Pixel2{uu, vv});
      if (ray.status != StatusCode::OK)
      {
        continue;
      }
      const Eigen::Vector3f d = ray.ray.direction / ray.ray.direction.z();
      if (ray.ray.direction.z() <= 0.0f || std::hypot(d.x(), d.y()) >= 0.999f * r_fold)
      {
        continue;
      }
      ++valid;
      const PixelResult px = rayToPixel(out, ray.ray.direction);
      ASSERT_EQ(px.status, StatusCode::OK);
      const float du = std::abs(px.pixel.u - out.intrinsics.cx) / out.intrinsics.cx;
      const float dv = std::abs(px.pixel.v - out.intrinsics.cy) / out.intrinsics.cy;
      EXPECT_LE(du, 1.0f + 1e-4f) << "(" << uu << ", " << vv << ")";
      EXPECT_LE(dv, 1.0f + 1e-4f) << "(" << uu << ", " << vv << ")";
      reach = std::max(reach, std::max(du, dv));
    }
  }
  EXPECT_GT(valid, 1000);
  // The fit is bound by the valid pixel centres next to the fold; the
  // distorted radius is flat there, so the pixels of this coarser lattice
  // sit a few percent inside them.
  EXPECT_GT(reach, 0.95f);
  // The folded pixels do not count: the view stays far narrower than one
  // stretched to hold the folded ray of (4, 354).
  const PixelResult folded_px = rayToPixel(out, folded.ray.direction);
  ASSERT_EQ(folded_px.status, StatusCode::OK);
  EXPECT_GT(folded_px.pixel.u, 1279.0f);
}

namespace
{

/// The output scale (template fx = fy = 1, principal point at the image
/// centre) that just contains the given output-frame rays.
float containingScale(const std::vector<Eigen::Vector3f> &rays, const ImageSize &out_size)
{
  CameraModel unit = makeTemplate(ProjectionModelType::PINHOLE);
  unit.intrinsics.cx = 0.5f * static_cast<float>(out_size.width - 1);
  unit.intrinsics.cy = 0.5f * static_cast<float>(out_size.height - 1);
  float du = 0.0f;
  float dv = 0.0f;
  for (const Eigen::Vector3f &ray : rays)
  {
    const PixelResult px = rayToPixel(unit, ray);
    if (px.ok())
    {
      du = std::max(du, std::abs(px.pixel.u - unit.intrinsics.cx));
      dv = std::max(dv, std::abs(px.pixel.v - unit.intrinsics.cy));
    }
  }
  return std::min(unit.intrinsics.cx / du, unit.intrinsics.cy / dv);
}

}  // namespace

// Two folds close together: with g(r) = r (1 + k1 r^2 + k2 r^4) and
// g'(r) = (1 - r^2 / a^2)(1 - r^2 / b^2), a = 0.5 and b = 0.505, the image
// turns back at r = 0.5 and forward again at 0.505, a sliver of a degree.
// Rays past the first fold are not valid, however short the sliver: the
// pixel (1000, 0) unprojects to x = 0.93 and must not count. Every ray
// before r = 0.5 that lands in the image is kept: they reach x = 0.5 on the
// image's top row, so the view is the one at fx = 500 / 0.5 = 1000.
TEST(Rectify, AlphaOneStopsAtTheFirstOfTwoCloseFolds)
{
  const double a = 0.5;
  const double b = 0.505;
  CameraModel src = makePinholeNoDistortion();
  src.intrinsics.fx = 1000.0f;
  src.intrinsics.fy = 1000.0f;
  src.intrinsics.cx = 0.0f;
  src.intrinsics.cy = 0.0f;
  src.distortion.type = DistortionModelType::RADTAN5;
  src.distortion.count = 5U;
  src.distortion.coeffs[0] = static_cast<float>(-(1.0 / (a * a) + 1.0 / (b * b)) / 3.0);
  src.distortion.coeffs[1] = static_cast<float>(1.0 / (5.0 * a * a * b * b));
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{1001, 1001};

  const RayResult past = pixelToRay(src, Pixel2{1000.0f, 0.0f});
  ASSERT_EQ(past.status, StatusCode::OK);
  EXPECT_GT(past.ray.direction.x() / past.ray.direction.z(), 0.9f);

  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);

  std::vector<Eigen::Vector3f> valid;
  for (int v = 0; v < sz.height; ++v)
  {
    for (int u = 0; u < sz.width; ++u)
    {
      const RayResult ray = pixelToRay(src, Pixel2{float(u), float(v)});
      if (ray.ok() && ray.ray.direction.z() > 0.0f &&
          ray.ray.direction.head<2>().norm() / ray.ray.direction.z() < 0.5f)
      {
        valid.push_back(ray.ray.direction);
      }
    }
  }
  EXPECT_LE(r.output_model.intrinsics.fx, containingScale(valid, sz));
  EXPECT_LE(r.output_model.intrinsics.fx, 1000.0f);
  EXPECT_GT(r.output_model.intrinsics.fx, 999.5f);
}

namespace
{

/// Rays (x, y, 1) of a pinhole-law source on an n x n lattice over
/// [-extent, extent]^2 that `valid` accepts and whose source pixel lies
/// between the outermost pixel centres of the image: rays alpha = 1 keeps.
std::vector<Eigen::Vector3f> raysInImage(
  const CameraModel &src, const ImageSize &size, const float extent, const int n,
  const std::function<bool(float, float)> &valid
)
{
  std::vector<Eigen::Vector3f> rays;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      const float x =
        extent * (2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n) - 1.0f);
      const float y =
        extent * (2.0f * (static_cast<float>(j) + 0.5f) / static_cast<float>(n) - 1.0f);
      if (!valid(x, y))
      {
        continue;
      }
      const Eigen::Vector3f ray(x, y, 1.0f);
      const PixelResult px = rayToPixel(src, ray);
      if (px.ok() && px.pixel.u >= 0.0f && px.pixel.v >= 0.0f &&
          px.pixel.u <= static_cast<float>(size.width - 1) &&
          px.pixel.v <= static_cast<float>(size.height - 1))
      {
        rays.push_back(ray);
      }
    }
  }
  return rays;
}

/// How many of `rays` (output frame) land outside the output image.
int countOutside(
  const std::vector<Eigen::Vector3f> &rays, const CameraModel &out, const ImageSize &out_size
)
{
  int outside = 0;
  for (const Eigen::Vector3f &ray : rays)
  {
    const PixelResult px = rayToPixel(out, ray);
    if (!px.ok() || px.pixel.u < -1e-3f || px.pixel.v < -1e-3f ||
        px.pixel.u > static_cast<float>(out_size.width - 1) + 1e-3f ||
        px.pixel.v > static_cast<float>(out_size.height - 1) + 1e-3f)
    {
      ++outside;
    }
  }
  return outside;
}

/// The scale of a pinhole output (unit template, centred) that just holds
/// the rays (x, y, 1) that `keep` accepts, estimated from inside: from a
/// lattice over [-extent, extent]^2, refined around its widest rays. The
/// alpha = 1 scale must not exceed it, and a tight fit comes close to it.
float referenceScale(
  const ImageSize &out_size, const float extent, const std::function<bool(float, float)> &keep
)
{
  const float half_w = 0.5f * static_cast<float>(out_size.width - 1);
  const float half_h = 0.5f * static_cast<float>(out_size.height - 1);
  const auto reach = [&](const float x, const float y) {
    return std::max(std::abs(x) / half_w, std::abs(y) / half_h);
  };
  constexpr int kCoarse = 600;
  const float step = 2.0f * extent / kCoarse;
  std::vector<std::pair<float, Eigen::Vector2f>> widest;
  for (int i = 0; i <= kCoarse; ++i)
  {
    for (int j = 0; j <= kCoarse; ++j)
    {
      const Eigen::Vector2f p(-extent + step * float(i), -extent + step * float(j));
      if (keep(p.x(), p.y()))
      {
        widest.emplace_back(reach(p.x(), p.y()), p);
      }
    }
  }
  const std::size_t top = std::min<std::size_t>(widest.size(), 32);
  std::partial_sort(
    widest.begin(), widest.begin() + top, widest.end(),
    [](const auto &a, const auto &b) { return a.first > b.first; }
  );
  float best = top > 0 ? widest[0].first : 0.0f;
  for (std::size_t k = 0; k < top; ++k)
  {
    for (int i = -100; i <= 100; ++i)
    {
      for (int j = -100; j <= 100; ++j)
      {
        const Eigen::Vector2f p =
          widest[k].second + 0.015f * step * Eigen::Vector2f(float(i), float(j));
        if (keep(p.x(), p.y()))
        {
          best = std::max(best, reach(p.x(), p.y()));
        }
      }
    }
  }
  return 1.0f / best;
}

/// Whether ray (x, y, 1) lands between the outermost pixel centres of the
/// source image.
bool landsInside(const CameraModel &src, const ImageSize &size, const float x, const float y)
{
  const PixelResult px = rayToPixel(src, Eigen::Vector3f(x, y, 1.0f));
  return px.ok() && px.pixel.u >= 0.0f && px.pixel.v >= 0.0f &&
         px.pixel.u <= static_cast<float>(size.width - 1) &&
         px.pixel.v <= static_cast<float>(size.height - 1);
}

/// A pinhole source of the given plane distortion (OpenCV coefficient
/// order), through the CameraInfo path so that the derived fields are set.
CameraModel planeSource(
  const char *distortion_model, const std::vector<double> &d, const double f, const ImageSize &size
)
{
  CameraInfo info;
  info.width = size.width;
  info.height = size.height;
  info.k = {f, 0.0, 0.5 * (size.width - 1), 0.0, f, 0.5 * (size.height - 1), 0.0, 0.0, 1.0};
  info.distortion_model = distortion_model;
  info.d = d;
  return makeCameraModel(info);
}

}  // namespace

// A fold whose distance from the axis depends on the direction (strong
// tangential, then thin-prism terms). Along each of 720 directions the fold
// is found by a scan of the image's orientation; every ray clearly before it
// that lands in the image must land inside the output, and the view must be
// as narrow as those rays allow.
TEST(Rectify, AlphaOneFollowsADirectionDependentFold)
{
  for (const DistortionModelType type :
       {DistortionModelType::RADTAN5, DistortionModelType::THIN_PRISM12})
  {
    CameraModel src = makePinholeNoDistortion();
    src.intrinsics.fx = 200.0f;
    src.intrinsics.fy = 200.0f;
    src.intrinsics.cx = 199.5f;
    src.intrinsics.cy = 149.5f;
    src.distortion.type = type;
    src.distortion.coeffs[0] = -0.3f;
    if (type == DistortionModelType::RADTAN5)
    {
      src.distortion.count = 5U;
      src.distortion.coeffs[2] = 0.02f;
      src.distortion.coeffs[3] = -0.015f;
    }
    else
    {
      src.distortion.count = 12U;
      src.distortion.is_rational = true;
      src.distortion.has_thin_prism = true;
      src.distortion.coeffs[8] = 0.03f;
      src.distortion.coeffs[10] = -0.02f;
    }
    ASSERT_EQ(validateCameraModel(src), StatusCode::OK) << toString(type);
    const ImageSize sz{400, 300};

    // Whether the image of a small radial and tangential step at (t, phi)
    // keeps its orientation.
    const auto keeps = [&](const float t, const float phi) {
      const auto pixel = [&](const float r, const float angle) {
        return rayToPixel(src, Eigen::Vector3f(r * std::cos(angle), r * std::sin(angle), 1.0f));
      };
      const PixelResult p = pixel(t, phi);
      const PixelResult q = pixel(t + 1e-4f, phi);
      const PixelResult s = pixel(t, phi + 1e-4f);
      const float area = (q.pixel.u - p.pixel.u) * (s.pixel.v - p.pixel.v) -
                         (q.pixel.v - p.pixel.v) * (s.pixel.u - p.pixel.u);
      return p.ok() && q.ok() && s.ok() && area > 0.0f;
    };
    std::vector<Eigen::Vector3f> rays;
    constexpr int kDirections = 720;
    for (int k = 0; k < kDirections; ++k)
    {
      const float phi = 2.0f * constants::kPiF * static_cast<float>(k) / kDirections;
      float fold = 3.0f;
      for (float t = 1e-3f; t < 3.0f; t += 5e-4f)
      {
        if (!keeps(t, phi))
        {
          fold = t - 5e-4f;
          break;
        }
      }
      for (int j = 1; j <= 2000; ++j)
      {
        const float t = 0.999f * fold * static_cast<float>(j) / 2000.0f;
        const Eigen::Vector3f ray(t * std::cos(phi), t * std::sin(phi), 1.0f);
        const PixelResult px = rayToPixel(src, ray);
        if (px.ok() && px.pixel.u >= 0.0f && px.pixel.v >= 0.0f && px.pixel.u <= 399.0f && px.pixel.v <= 299.0f)
        {
          rays.push_back(ray);
        }
      }
    }

    RectifiedOutputModelOptions opts;
    opts.alpha = 1.0f;
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
    ASSERT_EQ(r.status, StatusCode::OK) << toString(type);
    EXPECT_GT(rays.size(), 100000U) << toString(type);
    EXPECT_EQ(countOutside(rays, r.output_model, sz), 0) << toString(type);
    // Tight: the sampled rays reach within 0.3% of the view's edge (they stop
    // 0.1% short of the fold).
    EXPECT_GT(r.output_model.intrinsics.fx, 0.997f * containingScale(rays, sz)) << toString(type);
  }
}

// A thin prism (s3 = 1 alone: (x, y) -> (x, y + x^2 + y^2), Jacobian
// determinant 1 + 2 y) folds along the line y = -1/2, past which rays are not
// valid. Every valid ray that lands in the image is kept, and the view is
// set by the fold itself (y -> -1/2, inside the image): fx = 359.5 / 0.5.
TEST(Rectify, AlphaOneKeepsTheValidSideOfAPrismFold)
{
  const ImageSize sz{1280, 720};
  std::vector<double> d(12, 0.0);
  d[10] = 1.0;
  const CameraModel src = planeSource("rational_polynomial", d, 700.0, sz);
  ASSERT_EQ(src.distortion.type, DistortionModelType::THIN_PRISM12);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  const auto valid = [](const float, const float y) { return y > -0.5f; };
  const std::vector<Eigen::Vector3f> rays = raysInImage(src, sz, 1.0f, 1200, valid);
  EXPECT_GT(rays.size(), 100000U);
  EXPECT_EQ(countOutside(rays, r.output_model, sz), 0);
  const float reference = referenceScale(sz, 1.0f, [&](const float x, const float y) {
    return valid(x, y) && landsInside(src, sz, x, y);
  });
  EXPECT_NEAR(reference, 719.0f, 0.1f);
  EXPECT_LE(r.output_model.intrinsics.fx, reference);
  EXPECT_GT(r.output_model.intrinsics.fx, 0.999f * reference);
}

// A thin prism whose distorted radius is not monotone though its Jacobian
// determinant stays positive near the x axis (s3 = 10, s4 = -10). Every
// valid ray that lands in the image is kept, tightly.
TEST(Rectify, AlphaOneKeepsAPrismThatBendsBack)
{
  const ImageSize sz{1280, 720};
  std::vector<double> d(12, 0.0);
  d[10] = 10.0;
  d[11] = -10.0;
  const CameraModel src = planeSource("rational_polynomial", d, 700.0, sz);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  // det J = 1 + y (2 s3 + 4 s4 r^2); along t (x, y) it is
  // 1 + t y (20 - 40 t^2 r^2), lowest at t = 1 or where t^2 r^2 = 1 / 6.
  const auto valid = [](const float x, const float y) {
    const float r2 = x * x + y * y;
    const auto det = [&](const float t) { return 1.0f + t * y * (20.0f - 40.0f * t * t * r2); };
    const float t_low = std::sqrt(1.0f / (6.0f * r2));
    return det(1.0f) > 0.0f && (t_low >= 1.0f || det(t_low) > 0.0f);
  };
  const std::vector<Eigen::Vector3f> rays = raysInImage(src, sz, 1.5f, 1500, valid);
  EXPECT_GT(rays.size(), 10000U);
  EXPECT_EQ(countOutside(rays, r.output_model, sz), 0);
  const float reference = referenceScale(sz, 1.5f, [&](const float x, const float y) {
    return valid(x, y) && landsInside(src, sz, x, y);
  });
  EXPECT_LE(r.output_model.intrinsics.fx, reference);
  EXPECT_GT(r.output_model.intrinsics.fx, 0.999f * reference);
}

// A rational denominator that reaches zero: Q = 1 - r^2 (k4 = -1) sends the
// image out to infinity at r = 1, and rays past it land back inside the
// image, on the opposite side. Those are not valid: alpha = 1 keeps only
// the rays before r = 1, tightly, and alpha = 0 shows only them (an output
// looking at one past it has no view).
TEST(Rectify, StopsWhereARationalDenominatorVanishes)
{
  const ImageSize sz{640, 480};
  const CameraModel src =
    planeSource("rational_polynomial", {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0}, 300.0, sz);
  ASSERT_EQ(src.distortion.type, DistortionModelType::RATIONAL8);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const PixelResult past = rayToPixel(src, Eigen::Vector3f(3.0f, 0.0f, 1.0f));
  ASSERT_TRUE(past.ok());
  ASSERT_LT(past.pixel.u, 319.5f);  // r = 3 lands on the left half

  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  const auto valid = [](const float x, const float y) { return x * x + y * y < 1.0f; };
  const std::vector<Eigen::Vector3f> rays = raysInImage(src, sz, 1.0f, 1000, valid);
  EXPECT_GT(rays.size(), 100000U);
  EXPECT_EQ(countOutside(rays, r.output_model, sz), 0);
  const float reference = referenceScale(sz, 1.0f, [&](const float x, const float y) {
    return valid(x, y) && landsInside(src, sz, x, y);
  });
  EXPECT_LE(r.output_model.intrinsics.fx, reference);
  EXPECT_GT(r.output_model.intrinsics.fx, 0.999f * reference);

  opts.alpha = 0.0f;
  const RectifiedOutputModelResult inscribed = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(inscribed.status, StatusCode::OK);
  for (const Pixel2 corner : {Pixel2{0.0f, 0.0f}, Pixel2{639.0f, 479.0f}})
  {
    const RayResult ray = pixelToRay(inscribed.output_model, corner);
    ASSERT_TRUE(ray.ok());
    EXPECT_LT(ray.ray.direction.head<2>().norm() / ray.ray.direction.z(), 1.0f);
  }
  opts.src_from_output_rotation =
    Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), Eigen::Vector3f(3.0f, 0.0f, 1.0f))
      .toRotationMatrix();
  EXPECT_EQ(makeRectifiedOutputModel(src, sz, opts).status, StatusCode::DOMAIN_ERROR);
}

// A tilt whose plane turns edge-on in the field (tau_x = 0.5): the tilted
// point's third component, cos(0.5) - sin(0.5) y, reaches zero at
// y = 1.83, and rays past it are not valid, though those far past it land
// in the image (above the tilted plane's vanishing line, v = 30.9). An
// output looking at one of them has no alpha = 0 view. Facing forward, the
// valid rays run up to that vanishing line, out to the source's horizon,
// so alpha = 1 keeps them up to the output's horizon margin.
TEST(Rectify, StopsWhereATiltTurnsEdgeOn)
{
  const ImageSize sz{640, 480};
  std::vector<double> d(14, 0.0);
  d[12] = 0.5;
  const CameraModel src = planeSource("rational_polynomial", d, 100.0, sz);
  ASSERT_EQ(src.distortion.type, DistortionModelType::TILTED14);
  ASSERT_TRUE(src.distortion.has_tilt);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const auto &t = src.distortion.tilt_matrix;
  const auto valid = [&](const float x, const float y) {
    return t[6] * x + t[7] * y + t[8] > 0.0f;
  };
  ASSERT_FALSE(valid(0.0f, 17.0f));
  ASSERT_TRUE(landsInside(src, sz, 0.0f, 17.0f));

  RectifiedOutputModelOptions opts;
  opts.alpha = 0.0f;
  opts.src_from_output_rotation =
    Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), Eigen::Vector3f(0.0f, 17.0f, 1.0f))
      .toRotationMatrix();
  EXPECT_EQ(makeRectifiedOutputModel(src, sz, opts).status, StatusCode::DOMAIN_ERROR);

  opts.alpha = 1.0f;
  opts.src_from_output_rotation = Eigen::Matrix3f::Identity();
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  const float horizon = 1.0f / std::tan(0.05f);
  const std::vector<Eigen::Vector3f> rays =
    raysInImage(src, sz, horizon, 1600, [&](const float x, const float y) {
      return valid(x, y) && std::hypot(x, y) < horizon;
    });
  EXPECT_GT(rays.size(), 100000U);
  EXPECT_EQ(countOutside(rays, r.output_model, sz), 0);
  EXPECT_LE(r.output_model.intrinsics.fx, 239.5f / horizon);
  EXPECT_GT(r.output_model.intrinsics.fx, 0.999f * 239.5f / horizon);
}

// A double root: k1 = -10, k2 = 45 give g'(r) = (1 - 15 r^2)^2, which touches
// zero at r = 1 / sqrt(15) without changing sign. The rays past it are valid,
// but no certified bound can tell a touching root from a crossing one, so
// the fit treats them as possibly invalid, on the safe side for each alpha:
// alpha = 1 keeps them (all the image) and alpha = 0 stops at the root (its
// corners reach it).
TEST(Rectify, TreatsADoubleRootOnTheSafeSide)
{
  const ImageSize sz{1001, 1001};
  const CameraModel src = planeSource("plumb_bob", {-10.0, 45.0, 0.0, 0.0, 0.0}, 1000.0, sz);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const float root = 1.0f / std::sqrt(15.0f);

  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  const std::vector<Eigen::Vector3f> rays =
    raysInImage(src, sz, 1.0f, 1000, [](const float, const float) { return true; });
  EXPECT_GT(rays.size(), 100000U);
  EXPECT_EQ(countOutside(rays, r.output_model, sz), 0);
  const float reference = referenceScale(sz, 1.0f, [&](const float x, const float y) {
    return landsInside(src, sz, x, y);
  });
  EXPECT_LE(r.output_model.intrinsics.fx, reference);
  EXPECT_GT(r.output_model.intrinsics.fx, 0.999f * reference);

  opts.alpha = 0.0f;
  const RectifiedOutputModelResult inscribed = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(inscribed.status, StatusCode::OK);
  const RayResult corner = pixelToRay(inscribed.output_model, Pixel2{1000.0f, 1000.0f});
  ASSERT_TRUE(corner.ok());
  const float reach = corner.ray.direction.head<2>().norm() / corner.ray.direction.z();
  EXPECT_LE(reach, root);
  EXPECT_GT(reach, 0.999f * root);
}

// alpha = 1 into turned fisheye-law and lat-long outputs: the source border,
// every quarter pixel, lands inside the output and reaches its edge. The
// fit bounds these outputs to second order over each kind of source patch
// (the undistorted plane of a pinhole source, the angles of a fisheye, the
// longitudes and latitudes of an equirectangular one); this checks those
// bounds hold and are tight.
TEST(Rectify, AlphaOneHoldsTheBorderInTurnedOutputs)
{
  CameraModel equirect;
  ASSERT_EQ(
    makeEquirectangularModel(ImageSize{800, 400}, -1.0, 1.0, -0.5, 0.5, equirect), StatusCode::OK
  );
  const std::vector<NamedSource> sources = {
    {"pinhole_radtan5", makePinholeRadTan()}, {"kb4", makeKB4(1.0f)}, {"equirect", equirect}};
  const ImageSize sizes[] = {{640, 480}, {640, 480}, {800, 400}};
  const Eigen::Matrix3f rotation =
    Eigen::AngleAxisf(0.2f, Eigen::Vector3f(0.3f, 1.0f, 0.2f).normalized()).toRotationMatrix();
  for (std::size_t k = 0; k < sources.size(); ++k)
  {
    const CameraModel &src = sources[k].model;
    const ImageSize sz = sizes[k];
    for (const ProjectionModelType type :
         {ProjectionModelType::EQUIDISTANT, ProjectionModelType::STEREOGRAPHIC,
          ProjectionModelType::OMNIDIRECTIONAL, ProjectionModelType::EQUIRECTANGULAR,
          ProjectionModelType::CYLINDRICAL})
    {
      RectifiedOutputModelOptions opts = rectifyOptions(type, 1.0f);
      opts.src_from_output_rotation = rotation;
      const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
      ASSERT_EQ(r.status, StatusCode::OK) << sources[k].name << " -> " << toString(type);
      const CameraModel &out = r.output_model;
      const float half_w = 0.5f * static_cast<float>(opts.output_size.width - 1);
      const float half_h = 0.5f * static_cast<float>(opts.output_size.height - 1);
      float reach = 0.0f;
      int skipped = 0;
      const float w = static_cast<float>(sz.width - 1);
      const float h = static_cast<float>(sz.height - 1);
      const int steps = 4 * (sz.width + sz.height);
      for (int i = 0; i <= steps; ++i)
      {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        for (const Pixel2 &p :
             {Pixel2{w * t, 0.0f}, Pixel2{w * t, h}, Pixel2{0.0f, h * t}, Pixel2{w, h * t}})
        {
          const RayResult ray = pixelToRay(src, p);
          const PixelResult px = ray.ok()
                                   ? rayToPixel(out, rotation.transpose() * ray.ray.direction)
                                   : PixelResult{StatusCode::OUT_OF_FOV, {}};
          if (!px.ok())
          {
            ++skipped;
            continue;
          }
          const float du = std::abs(px.pixel.u - out.intrinsics.cx) / half_w;
          const float dv = std::abs(px.pixel.v - out.intrinsics.cy) / half_h;
          EXPECT_LE(std::max(du, dv), 1.0f + 1e-5f)
            << sources[k].name << " -> " << toString(type) << " at (" << p.u << ", " << p.v << ")";
          reach = std::max(reach, std::max(du, dv));
        }
      }
      if (skipped == 0)
      {
        EXPECT_GT(reach, 0.999f) << sources[k].name << " -> " << toString(type);
      }
    }
  }
}

// The seam of a lat-long output with skew: its two sides land on the same
// source pixels but at opposite output edges, and with skew the two edges
// are not mirror images. Every pixel of a cropped equirectangular source
// (all longitudes, a band of latitudes) lands inside.
TEST(Rectify, AlphaOneHoldsBothSidesOfASkewedLatLongSeam)
{
  CameraModel src;
  ASSERT_EQ(
    makeEquirectangularModel(
      ImageSize{4096, 200}, -constants::kPi, constants::kPi, -0.5, -0.2, src
    ),
    StatusCode::OK
  );
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(ProjectionModelType::EQUIRECTANGULAR);
  opts.output_template.intrinsics.skew = 5.0f;
  opts.output_size = ImageSize{640, 480};
  opts.src_from_output_rotation =
    Eigen::AngleAxisf(0.1f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  opts.alpha = 1.0f;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, ImageSize{4096, 200}, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_EQ(
    countUncontained(
      src, ImageSize{4096, 200}, r.output_model, opts.output_size, opts.src_from_output_rotation, 1
    ),
    0
  );
}

// A finite widest view close to the horizon is still finite: a 1001 x 1001
// pinhole source reaching 1e-3 rad from its horizon (fx = 0.5), on either
// side of that, and 2e-7 rad from it (fx = 1e-4), gives a 101 x 101 output
// its own reach (fx_out = fx_src / 10), not the margin stop meant for a
// source that covers the horizon itself.
TEST(Rectify, AlphaZeroFindsAFiniteWidestViewNearTheHorizon)
{
  for (const float fx : {1.0f, 0.5f, 0.25f, 1e-4f})
  {
    CameraModel src = makePinholeNoDistortion();
    src.intrinsics.fx = fx;
    src.intrinsics.fy = fx;
    src.intrinsics.cx = 500.0f;
    src.intrinsics.cy = 500.0f;
    RectifiedOutputModelOptions opts;
    opts.alpha = 0.0f;
    opts.output_size = ImageSize{101, 101};
    opts.source_margin_px = 0.0f;
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, ImageSize{1001, 1001}, opts);
    ASSERT_EQ(r.status, StatusCode::OK) << fx;
    EXPECT_GE(r.output_model.intrinsics.fx, 0.1f * fx) << fx;
    EXPECT_LE(r.output_model.intrinsics.fx, 0.1f * fx * 1.001f) << fx;
  }
}

// A plane distortion with every coefficient zero is the same lens as none:
// the fit does not cut it off at some search radius. A pinhole source
// reaching 2e-5 rad from its horizon gives the same widest view either way.
TEST(Rectify, AZeroPlaneDistortionIsNoDistortion)
{
  const ImageSize sz{1001, 1001};
  RectifiedOutputModelOptions opts;
  opts.alpha = 0.0f;
  opts.output_size = ImageSize{101, 101};
  opts.source_margin_px = 0.0f;
  const CameraModel zeros = planeSource("plumb_bob", {0.0, 0.0, 0.0, 0.0, 0.0}, 0.01, sz);
  ASSERT_EQ(zeros.distortion.type, DistortionModelType::RADTAN5);
  CameraModel plain = zeros;
  plain.distortion = DistortionModel{};
  const RectifiedOutputModelResult none = makeRectifiedOutputModel(plain, sz, opts);
  const RectifiedOutputModelResult radtan = makeRectifiedOutputModel(zeros, sz, opts);
  ASSERT_EQ(none.status, StatusCode::OK);
  ASSERT_EQ(radtan.status, StatusCode::OK);
  EXPECT_GE(none.output_model.intrinsics.fx, 0.001f);
  EXPECT_LE(none.output_model.intrinsics.fx, 0.001001f);
  EXPECT_NEAR(radtan.output_model.intrinsics.fx, none.output_model.intrinsics.fx, 1e-6f);
}

// An output that reaches round behind the source: a 175-degree equidistant
// source seen from an equidistant output turned 70 degrees. The source's
// back, past 175 degrees, can lie inside the output while the output's
// whole border is valid (at fx = 225.3 the pixel (932, 500) looks 179.9
// degrees off the source axis), and widening the view does not grow the
// invalid part monotonically along the border. alpha = 0 shows no invalid
// pixel, and is as wide as a dense scan of the output's rays allows.
TEST(Rectify, AlphaZeroKeepsTheSourceBackOutOfTheView)
{
  CameraModel src;
  src.projection.type = ProjectionModelType::EQUIDISTANT;
  src.projection.theta_max = 175.0f * constants::kPiF / 180.0f;
  src.intrinsics.fx = 100.0f;
  src.intrinsics.fy = 100.0f;
  src.intrinsics.cx = 500.0f;
  src.intrinsics.cy = 500.0f;
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{1001, 1001};
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(ProjectionModelType::EQUIDISTANT);
  opts.output_template.projection.theta_max = constants::kPiF - 1e-4f;
  opts.output_size = sz;
  opts.alpha = 0.0f;
  opts.source_margin_px = 0.0f;
  opts.src_from_output_rotation =
    Eigen::AngleAxisf(70.0f * constants::kPiF / 180.0f, Eigen::Vector3f::UnitY())
      .toRotationMatrix();
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const RectifyRemapResult r = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(r.remap_result.status, StatusCode::OK);
  EXPECT_EQ(r.remap_result.model_valid_count, r.remap_result.total_count);
  EXPECT_EQ(r.remap_result.source_in_bounds_count, r.remap_result.total_count);

  // The widest view whose rays on a 201 x 201 lattice are all valid and land
  // in the source (the set of rays grows with the view, so bisect).
  const auto
    good =
      [&](const float scale) {
        CameraModel out = opts.output_template;
        out.intrinsics.fx = scale;
        out.intrinsics.fy = scale;
        out.intrinsics.cx = 500.0f;
        out.intrinsics.cy = 500.0f;
        for (int i = 0; i <= 200; ++i)
        {
          for (int j = 0; j <= 200; ++j)
          {
            const RayResult ray = pixelToRay(out, Pixel2{5.0f * float(i), 5.0f * float(j)});
            const PixelResult px =
              ray.ok() ? rayToPixel(src, opts.src_from_output_rotation * ray.ray.direction)
                       : PixelResult{StatusCode::OUT_OF_FOV, {}};
            if (!px.ok() || px.pixel.u < 0.0f || px.pixel.v < 0.0f || px.pixel.u > 1000.0f ||
            px.pixel.v > 1000.0f)
            {
              return false;
            }
          }
        }
        return true;
      };
  float narrow = 1000.0f;
  float wide = 100.0f;
  for (int step = 0; step < 30; ++step)
  {
    const float mid = std::sqrt(narrow * wide);
    (good(mid) ? narrow : wide) = mid;
  }
  EXPECT_GE(r.output_model.intrinsics.fx, wide);
  EXPECT_LE(r.output_model.intrinsics.fx, 1.001f * narrow);
}

// alpha = 1 where the output's own limit crosses the middle of the source
// image: a narrow equidistant source (fx = 1e6) sitting on the 1 rad circle
// that an equidistant output with theta_max = 1 ends at, turned a little
// about the axis. The source rays inside that circle, a band through the
// image's centre, all land inside the output.
TEST(Rectify, AlphaOneKeepsRaysWhereTheOutputEndsInsideTheSource)
{
  CameraModel src;
  src.projection.type = ProjectionModelType::EQUIDISTANT;
  src.projection.theta_max = 1.1f;
  src.intrinsics.fx = 1e6f;
  src.intrinsics.fy = 1e6f;
  src.intrinsics.cx = -999950.0f;
  src.intrinsics.cy = 50.0f;
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const ImageSize sz{101, 101};
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(ProjectionModelType::EQUIDISTANT);
  opts.output_template.projection.theta_max = 1.0f;
  opts.output_size = sz;
  opts.alpha = 1.0f;
  opts.src_from_output_rotation =
    Eigen::AngleAxisf(-constants::kPiF / 256.0f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  // Sub-pixel source lattice: the kept band is a few pixels wide.
  int kept = 0;
  for (int i = 0; i <= 400; ++i)
  {
    for (int j = 0; j <= 400; ++j)
    {
      const RayResult ray = pixelToRay(src, Pixel2{0.25f * float(i), 0.25f * float(j)});
      ASSERT_TRUE(ray.ok());
      const PixelResult px =
        rayToPixel(r.output_model, opts.src_from_output_rotation.transpose() * ray.ray.direction);
      if (!px.ok())
      {
        continue;  // past the output's theta_max
      }
      ++kept;
      EXPECT_GE(px.pixel.u, -1e-3f) << i << ", " << j;
      EXPECT_LE(px.pixel.u, 100.001f) << i << ", " << j;
      EXPECT_GE(px.pixel.v, -1e-3f) << i << ", " << j;
      EXPECT_LE(px.pixel.v, 100.001f) << i << ", " << j;
    }
  }
  EXPECT_GT(kept, 1000);
}

// The fit's searches stop after a budget of patches; a search cut short
// returns the bound it holds, which is less tight but still on the safe
// side: alpha = 0 no wider and alpha = 1 no narrower than the full search.
// (Cut shorter still, the alpha = 0 search may hold no bound at all, and the
// fit then reports none.)
TEST(Rectify, ASearchCutShortStaysOnTheSafeSide)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  CameraModel unit = makeTemplate(ProjectionModelType::PINHOLE);
  unit.intrinsics.cx = 319.5f;
  unit.intrinsics.cy = 239.5f;
  const Eigen::Matrix3f rotation =
    Eigen::AngleAxisf(0.2f, Eigen::Vector3f(0.3f, 1.0f, 0.1f).normalized()).toRotationMatrix();
  const detail::RectifyScaleFit full =
    detail::fitRectifyScale(src, sz, unit, sz, rotation, 1.0f, SolverOptions{}, true, true);
  ASSERT_TRUE(full.inscribed_valid && full.contained_valid);
  for (const int budget : {4, 16, 64})
  {
    const detail::RectifyScaleFit cut = detail::fitRectifyScale(
      src, sz, unit, sz, rotation, 1.0f, SolverOptions{}, false, true, budget
    );
    ASSERT_TRUE(cut.contained_valid) << budget;
    // (The full search itself sits within 3e-4 of the exact scale.)
    EXPECT_LE(cut.contained_scale, full.contained_scale * 1.0003f) << budget;
    if (budget == 4)
    {
      EXPECT_LT(cut.contained_scale, 0.99f * full.contained_scale);  // visibly looser
    }
  }
  for (const int budget : {256, 1024})
  {
    const detail::RectifyScaleFit cut = detail::fitRectifyScale(
      src, sz, unit, sz, rotation, 1.0f, SolverOptions{}, true, false, budget
    );
    ASSERT_TRUE(cut.inscribed_valid) << budget;
    EXPECT_GE(cut.inscribed_scale, full.inscribed_scale / 1.0003f) << budget;
    if (budget == 256)
    {
      EXPECT_GT(cut.inscribed_scale, 1.01f * full.inscribed_scale);  // visibly looser
    }
  }
}

// The output's own domain bounds the alpha = 0 view whatever its size: an
// equidistant template with theta_max = 0.001 needs its corners, sqrt(2) 50
// pixels out, within 0.001 rad (fx >= 70710.7), far narrower than any
// fixed starting range of the search; and a pinhole source reaching
// 1.4e12 normalized (fx = 1.1e-8) rectifies to itself.
TEST(Rectify, AlphaZeroFindsTheOutputDomainAtAnyScale)
{
  CameraModel src = makePinholeNoDistortion();
  src.intrinsics.cx = 500.0f;
  src.intrinsics.cy = 500.0f;
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(ProjectionModelType::EQUIDISTANT);
  opts.output_template.projection.theta_max = 0.001f;
  opts.output_size = ImageSize{101, 101};
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, ImageSize{1001, 1001}, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_GE(r.output_model.intrinsics.fx, 70710.7f);
  EXPECT_LE(r.output_model.intrinsics.fx, 70710.7f * 1.001f);
  for (const Pixel2 corner : {Pixel2{0.0f, 0.0f}, Pixel2{100.0f, 100.0f}})
  {
    EXPECT_TRUE(pixelToRay(r.output_model, corner).ok());
  }

  CameraModel wide = makePinholeNoDistortion();
  wide.intrinsics.fx = 1.1e-8f;
  wide.intrinsics.fy = 1.1e-8f;
  wide.intrinsics.cx = 15000.0f;
  wide.intrinsics.cy = 15000.0f;
  ASSERT_EQ(validateCameraModel(wide), StatusCode::OK);
  RectifiedOutputModelOptions same;
  same.source_margin_px = 0.0f;
  const RectifiedOutputModelResult itself =
    makeRectifiedOutputModel(wide, ImageSize{30001, 30001}, same);
  ASSERT_EQ(itself.status, StatusCode::OK);
  EXPECT_GE(itself.output_model.intrinsics.fx, 1.1e-8f);
  EXPECT_LE(itself.output_model.intrinsics.fx, 1.1e-8f * 1.001f);
}

// Rays near an equirectangular output's pole are shown (the projection only
// fails within 1e-8 of it): a narrow source whose rays lie 4.5e-6 to 5.5e-6
// rad from the output's pole is held, at about fx = 50 / (pi / 2).
TEST(Rectify, AlphaOneKeepsRaysNearAnEquirectangularPole)
{
  CameraModel src = makePinholeNoDistortion();
  src.intrinsics.fx = 1e8f;
  src.intrinsics.fy = 1e8f;
  src.intrinsics.cx = 50.0f;
  src.intrinsics.cy = 50.0f;
  const ImageSize sz{101, 101};
  RectifiedOutputModelOptions opts;
  opts.output_template = makeTemplate(ProjectionModelType::EQUIRECTANGULAR);
  opts.output_size = sz;
  opts.alpha = 1.0f;
  opts.src_from_output_rotation =
    Eigen::AngleAxisf(constants::kHalfPiF - 5e-6f, Eigen::Vector3f::UnitX()).toRotationMatrix();
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  EXPECT_NEAR(r.output_model.intrinsics.fx, 50.0f / constants::kHalfPiF, 0.5f);
  EXPECT_EQ(countUncontained(src, sz, r.output_model, sz, opts.src_from_output_rotation, 1), 0);
}

// An image with a single column or row has no span between outermost pixel
// centres for alpha to refer to: refused, as an output and as a source.
TEST(Rectify, RefusesImagesWithoutTwoPixelCentres)
{
  const CameraModel src = makePinholeNoDistortion();
  std::vector<float> mx(101 * 101), my(101 * 101);
  for (const ImageSize thin : {ImageSize{1, 101}, ImageSize{101, 1}})
  {
    for (const float alpha : {0.0f, 1.0f})
    {
      RectifiedOutputModelOptions opts;
      opts.alpha = alpha;
      opts.output_size = thin;
      EXPECT_EQ(
        makeRectifiedOutputModel(src, ImageSize{640, 480}, opts).status, StatusCode::INVALID_INPUT
      );
      EXPECT_EQ(
        buildRectifyRemapMap(src, ImageSize{640, 480}, opts, mx.data(), my.data())
          .remap_result.status,
        StatusCode::INVALID_INPUT
      );
      opts.output_size = ImageSize{101, 101};
      EXPECT_EQ(makeRectifiedOutputModel(src, thin, opts).status, StatusCode::INVALID_INPUT);
      EXPECT_EQ(
        buildRectifyRemapMap(src, thin, opts, mx.data(), my.data()).remap_result.status,
        StatusCode::INVALID_INPUT
      );
    }
  }
}

// alpha = 0 takes the widest view with no invalid pixel even when it reaches
// close to the pinhole horizon (a 178-degree pinhole source: the source
// itself is the answer).
TEST(Rectify, AlphaZeroReachesPastTheHorizonMargin)
{
  CameraModel src = makePinholeNoDistortion();
  src.intrinsics.fx = 10.0f;
  src.intrinsics.fy = 10.0f;
  src.intrinsics.cx = 500.0f;
  src.intrinsics.cy = 500.0f;
  const ImageSize sz{1001, 1001};
  RectifiedOutputModelOptions opts;
  opts.source_margin_px = 0.0f;
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const RectifyRemapResult r = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(r.remap_result.status, StatusCode::OK);
  EXPECT_NEAR(r.output_model.intrinsics.fx, 10.0f, 0.05f);
  EXPECT_EQ(r.remap_result.source_in_bounds_count, r.remap_result.total_count);
}

// A source covering the whole half-space in front (a fisheye circle inside
// the frame) has no widest pinhole view: alpha = 0 stops 0.05 rad short of
// the pinhole horizon instead of failing.
TEST(Rectify, AlphaZeroStopsShortOfAnUnboundedHorizon)
{
  CameraModel src = makeKB4(constants::kPiF * 0.55f);
  src.intrinsics.fx = 150.0f;
  src.intrinsics.fy = 150.0f;
  const ImageSize sz{640, 480};  // radius 150 * 1.73 ~ 259 px > 240: the circle is cut
  CameraModel circle = src;
  circle.intrinsics.fx = 120.0f;  // radius ~ 207 px: the whole circle is inside
  circle.intrinsics.fy = 120.0f;
  updateThetaMax(circle);
  RectifiedOutputModelOptions opts;
  const RectifiedOutputModelResult r = makeRectifiedOutputModel(circle, sz, opts);
  ASSERT_EQ(r.status, StatusCode::OK);
  // The corner ray of the output sits at the 0.05 rad margin.
  const RayResult corner = pixelToRay(r.output_model, Pixel2{0.0f, 0.0f});
  ASSERT_EQ(corner.status, StatusCode::OK);
  const float theta = std::acos(corner.ray.direction.normalized().z());
  EXPECT_NEAR(theta, constants::kHalfPiF - 0.05f, 2e-3f);
}

// A long lens: the narrow starting view of the alpha = 0 search is still
// wider than the source, so the search narrows first.
TEST(Rectify, AlphaZeroFitsALongLens)
{
  for (const float f : {5e3f, 1e5f, 1e7f})
  {
    CameraModel src = makePinholeNoDistortion();
    src.intrinsics.fx = f;
    src.intrinsics.fy = f;
    RectifiedOutputModelOptions opts;
    opts.source_margin_px = 0.0f;
    const RectifiedOutputModelResult r = makeRectifiedOutputModel(src, ImageSize{640, 480}, opts);
    ASSERT_EQ(r.status, StatusCode::OK) << f;
    EXPECT_NEAR(r.output_model.intrinsics.fx, f, 0.005f * f) << f;
  }
}

// No source pixel within the output's reach: alpha = 1 has no scale.
TEST(Rectify, AlphaOneWithNothingToShow)
{
  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  opts.src_from_output_rotation = Eigen::Vector3f(-1.0f, 1.0f, -1.0f).asDiagonal();
  const RectifiedOutputModelResult r =
    makeRectifiedOutputModel(makePinholeNoDistortion(), ImageSize{640, 480}, opts);
  EXPECT_EQ(r.status, StatusCode::DOMAIN_ERROR);
}

TEST(Rectify, RejectsBadInputs)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  const RectifiedOutputModelOptions good;

  EXPECT_EQ(
    buildRectifyRemapMap(src, sz, good, nullptr, my.data()).remap_result.status,
    StatusCode::INVALID_INPUT
  );
  EXPECT_EQ(
    buildRectifyRemapMap(src, ImageSize{-1, 480}, good, mx.data(), my.data()).remap_result.status,
    StatusCode::INVALID_INPUT
  );

  CameraModel bad_src = src;
  bad_src.intrinsics.fx = 0.0f;
  EXPECT_EQ(makeRectifiedOutputModel(bad_src, sz, good).status, StatusCode::INVALID_MODEL);

  for (const float alpha : {-0.1f, 1.1f, std::nanf("")})
  {
    RectifiedOutputModelOptions opts;
    opts.alpha = alpha;
    EXPECT_EQ(makeRectifiedOutputModel(src, sz, opts).status, StatusCode::INVALID_INPUT) << alpha;
  }

  RectifiedOutputModelOptions distorted;
  distorted.output_template = src;  // carries RADTAN5
  EXPECT_EQ(makeRectifiedOutputModel(src, sz, distorted).status, StatusCode::INVALID_INPUT);

  RectifiedOutputModelOptions degenerate;
  degenerate.output_template.intrinsics.fx = 0.0f;
  const RectifyRemapResult r = buildRectifyRemapMap(src, sz, degenerate, mx.data(), my.data());
  EXPECT_EQ(r.remap_result.status, StatusCode::INVALID_INPUT);
  EXPECT_EQ(r.output_model.projection.type, ProjectionModelType::UNKNOWN);
}

// Regression guard for a pinhole + RADTAN5 source whose distortion polynomial
// folds over before the image corners (real 4K plumb_bob parameters scaled to
// 960x540). Past the fold, forward projection maps rays back INSIDE the
// image, so "lands inside the source" alone would let the alpha = 0 view grow
// across the fold. The fit must stop before it, and must not depend on
// inverting the source there (pixelToRay fails at the corner).
TEST(Rectify, PinholeRadTanFoldOverInscribed)
{
  CameraModel src{};
  src.projection.type = ProjectionModelType::PINHOLE;
  src.projection.theta_max = 1.5707962f;
  src.intrinsics.fx = 496.94f;
  src.intrinsics.fy = 495.44f;
  src.intrinsics.cx = 477.44f;
  src.intrinsics.cy = 249.79f;
  src.distortion.type = DistortionModelType::RADTAN5;
  src.distortion.count = 5U;
  const float k1 = -0.347168833f;
  const float k2 = 0.1494961828f;
  const float k3 = -0.034232568f;
  src.distortion.coeffs[0] = k1;
  src.distortion.coeffs[1] = k2;
  src.distortion.coeffs[2] = 0.003684729338f;
  src.distortion.coeffs[3] = -0.0002720679913f;
  src.distortion.coeffs[4] = k3;
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  ASSERT_NE(pixelToRay(src, Pixel2{959.f, 539.f}).status, StatusCode::OK)
    << "the corner is expected to lie past the fold";

  // First radius where the radial map r(1 + k1 r^2 + k2 r^4 + k3 r^6) stops
  // growing.
  double fold = 0.0;
  for (double r = 0.0; r < 3.0; r += 1e-4)
  {
    if (1.0 + 3.0 * k1 * r * r + 5.0 * k2 * std::pow(r, 4) + 7.0 * k3 * std::pow(r, 6) <= 0.0)
    {
      fold = r;
      break;
    }
  }
  ASSERT_GT(fold, 0.0);

  const ImageSize sz{960, 540};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  RectifiedOutputModelOptions opts;
  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);
  EXPECT_EQ(res.remap_result.source_in_bounds_count, res.remap_result.total_count);

  // Every output pixel's ray stays inside the fold radius.
  const CameraModel &out = res.output_model;
  for (const Pixel2 corner :
       {Pixel2{0.f, 0.f}, Pixel2{959.f, 0.f}, Pixel2{0.f, 539.f}, Pixel2{959.f, 539.f}})
  {
    const RayResult ray = pixelToRay(out, corner);
    ASSERT_EQ(ray.status, StatusCode::OK);
    const Eigen::Vector3f d = ray.ray.direction;
    EXPECT_LT(std::hypot(d.x() / d.z(), d.y() / d.z()), fold);
  }

  // The map warps the image.
  float max_deviation = 0.0f;
  for (int row = 0; row < sz.height; ++row)
  {
    for (int col = 0; col < sz.width; ++col)
    {
      const int idx = row * sz.width + col;
      max_deviation = std::max(max_deviation, std::hypot(mx[idx] - col, my[idx] - row));
    }
  }
  EXPECT_GT(max_deviation, 10.0f);
}

TEST(Rectify, MapValuesAreFinite)
{
  const CameraModel src = makePinholeRadTan();
  const ImageSize sz{640, 480};
  std::vector<float> mx, my;
  allocMaps(sz.width, sz.height, mx, my);
  RectifiedOutputModelOptions opts;
  opts.alpha = 1.0f;
  const RectifyRemapResult res = buildRectifyRemapMap(src, sz, opts, mx.data(), my.data());
  ASSERT_EQ(res.remap_result.status, StatusCode::OK);

  int valid = 0;
  for (int i = 0; i < sz.width * sz.height; ++i)
  {
    if (mx[i] == -1.0f && my[i] == -1.0f) continue;
    ASSERT_TRUE(std::isfinite(mx[i]) && std::isfinite(my[i])) << i;
    EXPECT_GE(mx[i], 0.0f);
    EXPECT_LT(mx[i], 640.0f);
    EXPECT_GE(my[i], 0.0f);
    EXPECT_LT(my[i], 480.0f);
    ++valid;
  }
  EXPECT_EQ(valid, res.remap_result.source_in_bounds_count);
  // alpha = 1 keeps a border where the barrel-shaped source does not reach.
  EXPECT_LT(valid, sz.width * sz.height);
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
