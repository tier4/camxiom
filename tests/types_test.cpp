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

#include "camxiom/types.hpp"

#include "camxiom/model.hpp"  // parseDistortionModelType (round-trip test)
#include "camxiom/types64.hpp"

#include <gtest/gtest.h>

using namespace camxiom;

// ---------------------------------------------------------------------------
// toString sanity
// ---------------------------------------------------------------------------

TEST(Types, StatusCodeToString)
{
  EXPECT_STREQ(toString(StatusCode::OK), "ok");
  EXPECT_STREQ(toString(StatusCode::INVALID_INPUT), "invalid_input");
  EXPECT_STREQ(toString(StatusCode::INVALID_MODEL), "invalid_model");
  EXPECT_STREQ(toString(StatusCode::BEHIND_CAMERA), "behind_camera");
  EXPECT_STREQ(toString(StatusCode::OUT_OF_FOV), "out_of_fov");
  EXPECT_STREQ(toString(StatusCode::DOMAIN_ERROR), "domain_error");
  EXPECT_STREQ(toString(StatusCode::NON_CONVERGED), "non_converged");
  EXPECT_STREQ(toString(StatusCode::NUMERIC_ERROR), "numeric_error");
  EXPECT_STREQ(toString(StatusCode::DEGENERATE_CONFIG), "degenerate_config");
}

TEST(Types, DistortionModelTypeToString)
{
  EXPECT_STREQ(toString(DistortionModelType::NONE), "none");
  EXPECT_STREQ(toString(DistortionModelType::RADTAN4), "radtan4");
  EXPECT_STREQ(toString(DistortionModelType::RADTAN5), "radtan5");
  EXPECT_STREQ(toString(DistortionModelType::RATIONAL8), "rational8");
  EXPECT_STREQ(toString(DistortionModelType::THIN_PRISM12), "thin_prism12");
  EXPECT_STREQ(toString(DistortionModelType::TILTED14), "tilted14");
  EXPECT_STREQ(toString(DistortionModelType::OPENCV_FISHEYE4), "opencv_fisheye4");
  EXPECT_STREQ(toString(DistortionModelType::KB4), "kb4");
  EXPECT_STREQ(toString(DistortionModelType::KB8), "kb8");
  EXPECT_STREQ(toString(DistortionModelType::EQUIDISTANT), "ideal_equidistant");
  EXPECT_STREQ(toString(DistortionModelType::OMNIDIRECTIONAL), "omnidirectional");
  EXPECT_STREQ(toString(DistortionModelType::UNKNOWN), "unknown");
}

TEST(Types, DistortionModelToStringRoundTripsThroughParse)
{
  // Every canonical toString output must parse back to the same enum so
  // persisted model strings survive a round trip. UNKNOWN is excluded: its
  // string is the parser's failure value by definition. Historically
  // "opencv_fisheye4"/"rational8"/... parsed to UNKNOWN and "equidistant"
  // parsed to a *different* type, silently corrupting persisted models.
  const DistortionModelType all[] = {
    DistortionModelType::NONE,
    DistortionModelType::RADTAN4,
    DistortionModelType::RADTAN5,
    DistortionModelType::RATIONAL8,
    DistortionModelType::THIN_PRISM12,
    DistortionModelType::TILTED14,
    DistortionModelType::OPENCV_FISHEYE4,
    DistortionModelType::KB4,
    DistortionModelType::KB8,
    DistortionModelType::EQUIDISTANT,
    DistortionModelType::EQUISOLID,
    DistortionModelType::STEREOGRAPHIC,
    DistortionModelType::ORTHOGRAPHIC,
    DistortionModelType::OMNIDIRECTIONAL,
  };
  for (const DistortionModelType type : all)
  {
    EXPECT_EQ(parseDistortionModelType(toString(type)), type)
      << "round trip broken for \"" << toString(type) << "\"";
  }
}

TEST(Types, ProjectionModelTypeToString)
{
  EXPECT_STREQ(toString(ProjectionModelType::PINHOLE), "pinhole");
  EXPECT_STREQ(toString(ProjectionModelType::FISHEYE_THETA), "fisheye_theta");
  EXPECT_STREQ(toString(ProjectionModelType::OMNIDIRECTIONAL), "omnidirectional");
  EXPECT_STREQ(toString(ProjectionModelType::DOUBLE_SPHERE), "double_sphere");
  EXPECT_STREQ(toString(ProjectionModelType::EUCM), "eucm");
  EXPECT_STREQ(toString(ProjectionModelType::UNKNOWN), "unknown");
}

// ---------------------------------------------------------------------------
// Float32 <-> Float64 conversion round-trip
// ---------------------------------------------------------------------------

TEST(Types, CameraModelToCameraModel64RoundTrip)
{
  CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 510.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.01f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = 1.5f;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = -0.1f;
  m.distortion.coeffs[1] = 0.02f;
  m.distortion.count = 4U;

  CameraModel64 m64 = toCameraModel64(m);
  EXPECT_DOUBLE_EQ(m64.intrinsics.fx, 500.0);
  EXPECT_DOUBLE_EQ(m64.intrinsics.fy, 510.0);
  EXPECT_NEAR(m64.intrinsics.skew, 0.01, 1e-7);
  EXPECT_EQ(m64.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_NEAR(m64.projection.theta_max, 1.5, 1e-6);
  EXPECT_EQ(m64.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(m64.distortion.space, DistortionSpace::ANGLE);
  EXPECT_EQ(m64.distortion.count, 4U);

  CameraModel back = fromCameraModel64(m64);
  EXPECT_FLOAT_EQ(back.intrinsics.fx, m.intrinsics.fx);
  EXPECT_FLOAT_EQ(back.intrinsics.fy, m.intrinsics.fy);
  EXPECT_FLOAT_EQ(back.intrinsics.skew, m.intrinsics.skew);
  EXPECT_EQ(back.projection.type, m.projection.type);
  EXPECT_FLOAT_EQ(back.projection.theta_max, m.projection.theta_max);
  EXPECT_EQ(back.distortion.type, m.distortion.type);
  EXPECT_EQ(back.distortion.count, m.distortion.count);
}

TEST(Types, DefaultsAreSane)
{
  CameraModel m{};
  EXPECT_EQ(m.intrinsics.fx, 1.0f);
  EXPECT_EQ(m.intrinsics.fy, 1.0f);
  EXPECT_EQ(m.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  EXPECT_FALSE(m.distortion.has_tilt);
  EXPECT_FALSE(m.distortion.has_thin_prism);
  // Tilt matrices default to identity.
  EXPECT_FLOAT_EQ(m.distortion.tilt_matrix[0], 1.0f);
  EXPECT_FLOAT_EQ(m.distortion.tilt_matrix[4], 1.0f);
  EXPECT_FLOAT_EQ(m.distortion.tilt_matrix[8], 1.0f);
}
