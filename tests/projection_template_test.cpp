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

#include "camxiom/projection_template.hpp"

#include "camxiom/types64.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace camxiom;
namespace tpl = camxiom::projection_template;

std::vector<CameraModel64> models()
{
  std::vector<CameraModel64> out;
  for (auto projection :
       {ProjectionModelType::PINHOLE, ProjectionModelType::EQUIDISTANT,
        ProjectionModelType::EQUISOLID, ProjectionModelType::STEREOGRAPHIC,
        ProjectionModelType::ORTHOGRAPHIC, ProjectionModelType::OMNIDIRECTIONAL,
        ProjectionModelType::DOUBLE_SPHERE, ProjectionModelType::EUCM})
  {
    using Layouts = std::vector<std::pair<DistortionModelType, int>>;
    const Layouts kb_layouts = {
      {DistortionModelType::NONE, 0}, {DistortionModelType::KB4, 4}, {DistortionModelType::KB8, 8}};
    const std::vector<std::pair<DistortionModelType, int>> layouts =
      projection == ProjectionModelType::EQUIDISTANT ? kb_layouts
      : isFisheyeProjection(projection)
        ? Layouts{{DistortionModelType::NONE, 0}}
        : std::vector<std::pair<DistortionModelType, int>>{
            {DistortionModelType::NONE, 0},      {DistortionModelType::RADTAN4, 4},
            {DistortionModelType::RADTAN4, 5},   {DistortionModelType::RADTAN5, 5},
            {DistortionModelType::RATIONAL8, 8}, {DistortionModelType::THIN_PRISM12, 12},
            {DistortionModelType::TILTED14, 14}};
    for (const auto &[distortion, count] : layouts)
    {
      CameraModel64 m;
      m.intrinsics.fx = 500;
      m.intrinsics.fy = 490;
      m.intrinsics.cx = 320;
      m.intrinsics.cy = 240;
      m.projection.type = projection;
      m.projection.xi = 0.2;
      m.projection.alpha = 0.6;
      m.projection.beta = 0.9;
      m.projection.theta_max = 1.4;
      m.distortion.type = distortion;
      m.distortion.count = count;
      out.push_back(m);
    }
  }
  return out;
}

template <typename T>
void expectProjection(const CameraModel64 &m, const std::array<T, 3> &ray, bool expected)
{
  T u = T(17), v = T(19);
  EXPECT_EQ(tpl::projectGeneric(m, ray[0], ray[1], ray[2], u, v), expected);
  if (!expected)
  {
    EXPECT_EQ(u, T(0));
    EXPECT_EQ(v, T(0));
  }
  const T intr[] = {T(m.intrinsics.fx), T(m.intrinsics.fy), T(m.intrinsics.cx), T(m.intrinsics.cy)};
  std::vector<T> dist(m.distortion.count);
  for (std::size_t i = 0; i < dist.size(); ++i) dist[i] = T(m.distortion.coeffs[i]);
  u = T(17);
  v = T(19);
  EXPECT_EQ(
    tpl::projectGenericParametric(
      m.projection.type, m.distortion.type, intr, dist.data(), int(dist.size()), T(m.projection.xi),
      T(m.projection.alpha), T(m.projection.beta), ray[0], ray[1], ray[2], u, v,
      m.projection.theta_max
    ),
    expected
  );
  if (!expected)
  {
    EXPECT_EQ(u, T(0));
    EXPECT_EQ(v, T(0));
  }
}

template <typename T>
class ProjectionTemplate : public ::testing::Test
{
};
using Scalars = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ProjectionTemplate, Scalars);

TYPED_TEST(ProjectionTemplate, RejectsEveryNonFiniteInputAndCoefficient)
{
  using T = TypeParam;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const auto &model : models())
  {
    SCOPED_TRACE(int(model.projection.type));
    SCOPED_TRACE(int(model.distortion.type));
    SCOPED_TRACE(model.distortion.count);
    for (auto ray : {std::array<T, 3>{T(.2), T(.1), T(1)}, std::array<T, 3>{T(0), T(0), T(1)}})
    {
      expectProjection(model, ray, true);
      for (double invalid : {nan, inf, -inf})
      {
        for (int axis = 0; axis < 3; ++axis)
        {
          auto bad_ray = ray;
          bad_ray[axis] = T(invalid);
          expectProjection(model, bad_ray, false);
        }
        for (int i = 0; i < 4 + model.distortion.count; ++i)
        {
          auto m = model;
          const std::array<double *, 4> intr = {
            &m.intrinsics.fx, &m.intrinsics.fy, &m.intrinsics.cx, &m.intrinsics.cy};
          if (i < 4)
            *intr[i] = invalid;
          else
            m.distortion.coeffs[i - 4] = invalid;
          SCOPED_TRACE(i);
          expectProjection(m, ray, false);
        }
      }
    }
  }
}

TYPED_TEST(ProjectionTemplate, RetainsUnsmoothedFisheyeCapAtTheAxisAndBoundary)
{
  using T = TypeParam;
  for (auto m : models())
  {
    if (!isFisheyeProjection(m.projection.type)) continue;
    for (T x : {T(0), T(1e-16), T(1e-12), T(.25), T(1)})
      for (T z : {T(1), T(0), T(-1)})
      {
        const std::array<T, 3> ray{x, T(0), z};
        const double angle = std::atan2(std::sqrt(x * x), z);
        const bool valid_ray =
          x * x + z * z > tpl::detail::epsilon<T>() && !(x <= tpl::detail::epsilon<T>() && z < 0);
        const bool valid_mapping = m.projection.type != ProjectionModelType::ORTHOGRAPHIC || z >= 0;
        for (double cap :
             {std::nextafter(angle, 0.0), angle,
              std::nextafter(angle, std::numeric_limits<double>::infinity()), 1e-20, 0.5,
              constants::kPi})
        {
          m.projection.theta_max = cap;
          expectProjection(m, ray, valid_ray && valid_mapping && cap > 0 && angle <= cap);
        }
      }
  }
}

TYPED_TEST(ProjectionTemplate, DoesNotCacheMutableModelParameters)
{
  using T = TypeParam;
  for (auto m : models())
  {
    const std::array<T, 3> ray{T(.25), T(0), T(1)};
    expectProjection(m, ray, true);
    m.intrinsics.fx = 0;
    expectProjection(m, ray, false);
    m.intrinsics.fx = 500;
    expectProjection(m, ray, true);
    if (m.projection.type != ProjectionModelType::PINHOLE)
    {
      m.projection.theta_max = .1;
      expectProjection(m, ray, false);
      m.projection.theta_max = 1.4;
      expectProjection(m, ray, true);
    }
    if (m.distortion.count)
    {
      m.distortion.coeffs[0] = std::numeric_limits<double>::quiet_NaN();
      expectProjection(m, ray, false);
      m.distortion.coeffs[0] = 0;
      expectProjection(m, ray, true);
    }
  }
}

TYPED_TEST(ProjectionTemplate, PreservesValidSubnormalPinholeDepth)
{
  using T = TypeParam;
  for (const auto &m : models())
    if (m.projection.type == ProjectionModelType::PINHOLE)
      expectProjection(m, std::array<T, 3>{T(0), T(0), std::numeric_limits<T>::denorm_min()}, true);
}
// Compare the optimized range checks to ordinary floating-point predicates,
// including arbitrary IEEE encodings, signed zero, subnormals and NaN payloads.
TYPED_TEST(ProjectionTemplate, ClassifiesDepthAndNormLikeStandardFloatingPointChecks)
{
  using T = TypeParam;
  using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
  auto pinhole = models()[3];  // RADTAN5 consumes the checked positive depth.
  ASSERT_EQ(pinhole.distortion.type, DistortionModelType::RADTAN5);
  auto fisheye = pinhole;
  fisheye.projection.type = ProjectionModelType::EQUIDISTANT;
  fisheye.distortion.type = DistortionModelType::KB4;
  fisheye.distortion.count = 4;
  std::mt19937_64 random(123456);
  for (int i = 0; i < 8192; ++i)
  {
    const Bits bits = Bits(random());
    T z;
    std::memcpy(&z, &bits, sizeof(z));
    const std::array<T, 3> ray{T(0), T(0), z};
    expectProjection(pinhole, ray, std::isfinite(z) && z > 0);
    const T norm_sq = z * z;
    expectProjection(
      fisheye, ray, z > 0 && std::isfinite(norm_sq) && norm_sq > tpl::detail::epsilon<T>()
    );
  }
}

TEST(ProjectionTemplateFallback, PreservesTheLongDoubleHalfPiBoundary)
{
  auto model = models().front();
  model.projection.type = ProjectionModelType::EQUIDISTANT;
  model.distortion.type = DistortionModelType::NONE;
  model.projection.theta_max = constants::kHalfPi;
  const std::array<long double, 3> ray{1, 0, 0};
  expectProjection(model, ray, std::atan2(1.0L, 0.0L) <= model.projection.theta_max);
}

}  // namespace
