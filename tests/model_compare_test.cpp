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

// Tests for camxiom/model_compare.hpp: exact operator== / operator!= and
// tolerance-based isApprox on the camera model structs (float and double
// aliases).

#include "camxiom/model_compare.hpp"

#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace
{

using camxiom::CameraModel;
using camxiom::CameraModel64;
using camxiom::DistortionModelType;
using camxiom::DistortionSpace;
using camxiom::ProjectionModelType;

CameraModel makePinholeRadtan()
{
  CameraModel m{};
  m.intrinsics.fx = 800.0f;
  m.intrinsics.fy = 810.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = 1.5f;
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5;
  m.distortion.coeffs = {-0.28f, 0.07f, 1e-4f, -2e-4f, 0.0f};
  return m;
}

TEST(ModelCompareExact, IdenticalModelsAreEqual)
{
  const CameraModel a = makePinholeRadtan();
  const CameraModel b = a;
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
}

TEST(ModelCompareExact, AnyFieldChangeBreaksEquality)
{
  const CameraModel a = makePinholeRadtan();

  CameraModel b = a;
  // Must exceed the float ULP at 800 (~6e-5) to actually change the value.
  b.intrinsics.fx += 1e-3f;
  EXPECT_TRUE(a != b);

  b = a;
  b.projection.xi = 0.5f;
  EXPECT_TRUE(a != b);

  b = a;
  b.distortion.coeffs[4] = 1e-7f;
  EXPECT_TRUE(a != b);

  b = a;
  b.distortion.count = 4;
  EXPECT_TRUE(a != b);

  // Derived aux state participates in exact equality (cache-invalidation
  // semantics).
  b = a;
  b.distortion.has_thin_prism = true;
  EXPECT_TRUE(a != b);

  b = a;
  b.distortion.tilt_matrix[0] = 2.0f;
  EXPECT_TRUE(a != b);
}

TEST(ModelCompareExact, NanNeverCompareEqual)
{
  CameraModel a = makePinholeRadtan();
  a.intrinsics.fx = std::numeric_limits<float>::quiet_NaN();
  const CameraModel b = a;
  EXPECT_FALSE(a == b);
}

TEST(ModelCompareApprox, WithinToleranceIsApprox)
{
  const CameraModel a = makePinholeRadtan();
  CameraModel b = a;
  b.intrinsics.fx += 800.0f * 1e-6f;  // well inside rel_tol = 1e-5
  b.distortion.coeffs[0] += 1e-7f;    // inside abs_tol = 1e-6
  EXPECT_TRUE(camxiom::isApprox(a, b));
  // Exact equality must still fail.
  EXPECT_TRUE(a != b);
}

TEST(ModelCompareApprox, BeyondToleranceIsNotApprox)
{
  const CameraModel a = makePinholeRadtan();
  CameraModel b = a;
  b.intrinsics.fx += 1.0f;  // 800 -> 801: rel error ~1.2e-3 >> 1e-5
  EXPECT_FALSE(camxiom::isApprox(a, b));
  // ... but a caller-widened tolerance accepts it.
  EXPECT_TRUE(camxiom::isApprox(a, b, 1e-2f, 1e-6f));
}

TEST(ModelCompareApprox, TypeAndCountMustMatchExactly)
{
  const CameraModel a = makePinholeRadtan();

  CameraModel b = a;
  b.distortion.type = DistortionModelType::RADTAN4;
  EXPECT_FALSE(camxiom::isApprox(a, b));

  b = a;
  b.distortion.count = 4;
  EXPECT_FALSE(camxiom::isApprox(a, b));

  b = a;
  b.projection.type = ProjectionModelType::FISHEYE_THETA;
  EXPECT_FALSE(camxiom::isApprox(a, b));
}

TEST(ModelCompareApprox, DerivedStateIsIgnored)
{
  const CameraModel a = makePinholeRadtan();
  CameraModel b = a;
  // Aux flags / tilt cache / space are derived from type+coeffs; isApprox
  // must not look at them.
  b.distortion.has_thin_prism = !a.distortion.has_thin_prism;
  b.distortion.tilt_matrix[0] = 3.0f;
  b.distortion.space = DistortionSpace::NONE;
  EXPECT_TRUE(camxiom::isApprox(a, b));
}

TEST(ModelCompareApprox, CoefficientsBeyondCountAreIgnored)
{
  const CameraModel a = makePinholeRadtan();
  CameraModel b = a;
  b.distortion.coeffs[10] = 42.0f;  // beyond count = 5
  EXPECT_TRUE(camxiom::isApprox(a, b));
  EXPECT_TRUE(a != b);  // exact equality still sees it
}

TEST(ModelCompareApprox, NonFiniteScalarsAreNotApproximatelyEqualUnlessIdentical)
{
  CameraModel a = makePinholeRadtan();
  CameraModel b = a;

  a.intrinsics.fx = std::numeric_limits<float>::infinity();
  b.intrinsics.fx = 800.0f;
  EXPECT_FALSE(camxiom::isApprox(a, b));

  b.intrinsics.fx = -std::numeric_limits<float>::infinity();
  EXPECT_FALSE(camxiom::isApprox(a, b));

  b.intrinsics.fx = std::numeric_limits<float>::infinity();
  EXPECT_TRUE(camxiom::isApprox(a, b));

  b.intrinsics.fx = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(camxiom::isApprox(a, b));
}

TEST(ModelCompareDouble, WorksForDoubleAliases)
{
  CameraModel64 a{};
  a.intrinsics.fx = 800.0;
  a.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  a.projection.xi = 0.2;
  a.projection.alpha = 0.6;

  CameraModel64 b = a;
  EXPECT_TRUE(a == b);
  b.projection.alpha += 1e-9;
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(camxiom::isApprox(a, b));
  b.projection.alpha = 0.7;
  EXPECT_FALSE(camxiom::isApprox(a, b));
}

}  // namespace
