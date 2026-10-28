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

// exportCameraInfo: a model is written to a ROS CameraInfo only when a ROS
// distortion model holds the same lens, and every model written reads back
// (makeCameraModel) as a model that projects every ray to the same pixel.

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace
{

using namespace camxiom;

struct NamedModel
{
  const char *name;
  CameraModel model;
};

CameraModel fromCameraInfo(const std::string &distortion_model, const std::vector<double> &d)
{
  CameraInfo info;
  info.k = {400.0, 0.0, 320.0, 0.0, 410.0, 240.0, 0.0, 0.0, 1.0};
  info.distortion_model = distortion_model;
  info.d = d;
  return makeCameraModel(info);
}

std::vector<double> sequence(const std::size_t count, const double start, const double step)
{
  std::vector<double> values(count);
  for (std::size_t i = 0; i < count; ++i)
  {
    values[i] = start + step * static_cast<double>(i);
  }
  return values;
}

// Rays over the field the narrowest of these models covers (about 50 degrees
// off axis), in every direction.
std::vector<Eigen::Vector3f> sampleRays()
{
  std::vector<Eigen::Vector3f> rays;
  for (int i = 0; i <= 6; ++i)
  {
    const float theta = 0.15f * static_cast<float>(i);
    for (int j = 0; j < 8; ++j)
    {
      const float phi = 0.785f * static_cast<float>(j);
      rays.emplace_back(
        std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)
      );
    }
  }
  return rays;
}

void expectSameLens(const CameraModel &a, const CameraModel &b, const char *what)
{
  ASSERT_EQ(a.projection.type, b.projection.type) << what;
  for (const Eigen::Vector3f &ray : sampleRays())
  {
    const PixelResult pa = rayToPixel(a, ray);
    const PixelResult pb = rayToPixel(b, ray);
    ASSERT_EQ(pa.status, pb.status) << what;
    if (pa.status == StatusCode::OK)
    {
      EXPECT_NEAR(pa.pixel.u, pb.pixel.u, 1e-3f) << what;
      EXPECT_NEAR(pa.pixel.v, pb.pixel.v, 1e-3f) << what;
    }
  }
}

CameraInfo sentinel()
{
  CameraInfo info;
  info.distortion_model = "untouched";
  info.d = {9.0};
  info.k[0] = 9.0;
  info.width = 7U;
  info.height = 5U;
  return info;
}

void expectUntouched(const CameraInfo &info, const char *what)
{
  EXPECT_EQ(info.distortion_model, "untouched") << what;
  EXPECT_EQ(info.d, std::vector<double>{9.0}) << what;
  EXPECT_EQ(info.k[0], 9.0) << what;
}

}  // namespace

// Every model ROS can hold: the name, the coefficients (zero-padded only to
// the length the name fixes), K / R / P, and a read-back that projects like
// the original.
TEST(CameraInfoExportRos, WritesEveryLensRosCanHold)
{
  struct Case
  {
    NamedModel source;
    const char *ros_name;
    std::vector<double> d;
  };
  std::vector<double> rational = sequence(8U, -0.2, 0.03);
  std::vector<double> prism_zero = rational;
  prism_zero.resize(12U, 0.0);
  std::vector<double> tilt_zero = rational;
  tilt_zero.resize(14U, 0.0);
  const std::vector<double> kb4{0.02, -0.004, 0.0008, -0.0001};
  std::vector<double> kb8_zero_tail = kb4;
  kb8_zero_tail.resize(8U, 0.0);
  const std::vector<Case> cases{
    {{"pinhole none", fromCameraInfo("none", {})}, "plumb_bob", {0.0, 0.0, 0.0, 0.0, 0.0}},
    {{"radtan4", fromCameraInfo("radtan4", {-0.2, 0.05, 0.001, -0.002})},
     "plumb_bob",
     {-0.2, 0.05, 0.001, -0.002, 0.0}},
    {{"radtan5", fromCameraInfo("radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01})},
     "plumb_bob",
     {-0.2, 0.05, 0.001, -0.002, 0.01}},
    {{"rational8", fromCameraInfo("rational8", rational)}, "rational_polynomial", rational},
    {{"thin_prism12 with zero prism", fromCameraInfo("thin_prism12", prism_zero)},
     "rational_polynomial",
     rational},
    {{"tilted14 with zero prism and tilt", fromCameraInfo("tilted14", tilt_zero)},
     "rational_polynomial",
     rational},
    {{"equidistant kb4", fromCameraInfo("kb4", kb4)}, "equidistant", kb4},
    {{"ideal equidistant", fromCameraInfo("ideal_equidistant", {})},
     "equidistant",
     {0.0, 0.0, 0.0, 0.0}},
    {{"kb8 with zero k5..k8", fromCameraInfo("kb8", kb8_zero_tail)}, "equidistant", kb4},
  };

  for (const Case &c : cases)
  {
    const char *what = c.source.name;
    ASSERT_EQ(validateCameraModel(c.source.model), StatusCode::OK) << what;
    CameraInfo out = sentinel();
    ASSERT_EQ(exportCameraInfo(c.source.model, out), StatusCode::OK) << what;
    EXPECT_EQ(out.distortion_model, c.ros_name) << what;
    ASSERT_EQ(out.d.size(), c.d.size()) << what;
    for (std::size_t i = 0; i < c.d.size(); ++i)
    {
      EXPECT_NEAR(out.d[i], c.d[i], 1e-7) << what << " d[" << i << "]";
    }
    const std::array<double, 9> k{400.0, 0.0, 320.0, 0.0, 410.0, 240.0, 0.0, 0.0, 1.0};
    const std::array<double, 9> r{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 12> p{400.0, 0.0, 320.0, 0.0, 0.0, 410.0,
                                   240.0, 0.0, 0.0,   0.0, 1.0, 0.0};
    EXPECT_EQ(out.k, k) << what;
    EXPECT_EQ(out.r, r) << what;
    EXPECT_EQ(out.p, p) << what;
    EXPECT_EQ(out.width, 7U) << what;  // the image size is the caller's
    EXPECT_EQ(out.height, 5U) << what;

    const CameraModel back = makeCameraModel(out);
    ASSERT_EQ(validateCameraModel(back), StatusCode::OK) << what;
    expectSameLens(c.source.model, back, what);
  }
}

// A RADTAN4 model may carry k3 in a fifth coefficient (projection evaluates
// it); the export keeps it, so the ROS file is the same lens.
TEST(CameraInfoExportRos, KeepsTheFifthCoefficientOfRadtan4)
{
  CameraModel source;
  source.intrinsics.fx = 400.0f;
  source.intrinsics.fy = 410.0f;
  source.intrinsics.cx = 320.0f;
  source.intrinsics.cy = 240.0f;
  source.projection.type = ProjectionModelType::PINHOLE;
  source.distortion.type = DistortionModelType::RADTAN4;
  source.distortion.count = 5U;
  source.distortion.coeffs = {-0.2f, 0.05f, 0.001f, -0.002f, 0.01f};
  ASSERT_EQ(validateCameraModel(source), StatusCode::OK);
  CameraInfo out;
  ASSERT_EQ(exportCameraInfo(source, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
  ASSERT_EQ(out.d.size(), 5U);
  EXPECT_NEAR(out.d[4], 0.01, 1e-7);
  const CameraModel back = makeCameraModel(out);
  ASSERT_EQ(validateCameraModel(back), StatusCode::OK);
  expectSameLens(source, back, "radtan4 with k3");
}

// ROS defines K without skew and its receivers project without it, so a
// skewed camera is refused rather than written as a different lens.
TEST(CameraInfoExportRos, RefusesASkewedCamera)
{
  CameraModel source;
  source.intrinsics.fx = 400.0f;
  source.intrinsics.fy = 410.0f;
  source.intrinsics.cx = 320.0f;
  source.intrinsics.cy = 240.0f;
  source.intrinsics.skew = 2.0f;
  source.projection.type = ProjectionModelType::PINHOLE;
  source.distortion.type = DistortionModelType::RADTAN5;
  source.distortion.count = 5U;
  source.distortion.coeffs = {-0.2f, 0.05f, 0.001f, -0.002f, 0.01f};
  ASSERT_EQ(validateCameraModel(source), StatusCode::OK);
  CameraInfo out = sentinel();
  EXPECT_EQ(exportCameraInfo(source, out), StatusCode::INVALID_MODEL);
  expectUntouched(out, "skewed camera");
  CameraInfo out64 = sentinel();
  EXPECT_EQ(exportCameraInfo64(toCameraModel64(source), out64), StatusCode::INVALID_MODEL);
  expectUntouched(out64, "skewed camera, double");
}

// A model ROS cannot hold is refused and nothing is written: a non-zero prism
// or tilt term, KB8 terms past k4, and every projection other than pinhole and
// the equidistant law.
TEST(CameraInfoExportRos, RefusesLensesRosCannotHold)
{
  std::vector<double> prism = sequence(8U, -0.2, 0.03);
  prism.resize(12U, 0.0);
  prism[9] = 0.001;
  std::vector<double> tilt = sequence(8U, -0.2, 0.03);
  tilt.resize(14U, 0.0);
  tilt[13] = 0.01;
  std::vector<double> kb8{0.02, -0.004, 0.0008, -0.0001, 0.0, 0.0, 1e-5, 0.0};

  CameraModel equirect;
  ASSERT_EQ(makeEquirectangularModel(ImageSize{2048, 1024}, equirect), StatusCode::OK);
  CameraModel cylindrical;
  ASSERT_EQ(
    makeCylindricalModel(ImageSize{1024, 512}, -2.0, 2.0, -1.0, 1.0, cylindrical), StatusCode::OK
  );

  const std::vector<NamedModel> refused{
    {"thin_prism12 with a prism term", fromCameraInfo("thin_prism12", prism)},
    {"tilted14 with a tilt term", fromCameraInfo("tilted14", tilt)},
    {"kb8 with a k7 term", fromCameraInfo("kb8", kb8)},
    {"equisolid", fromCameraInfo("equisolid", {})},
    {"stereographic", fromCameraInfo("stereographic", {})},
    {"orthographic", fromCameraInfo("orthographic", {})},
    {"omnidirectional", fromCameraInfo("omnidirectional", {0.8, -0.2, 0.05, 0.001, -0.001})},
    {"omnidirectional without correction", fromCameraInfo("omnidirectional", {0.8})},
    {"double sphere", fromCameraInfo("double_sphere", {-0.2, 0.6})},
    {"eucm", fromCameraInfo("eucm", {0.6, 1.1})},
    {"equirectangular", equirect},
    {"cylindrical", cylindrical},
  };
  for (const NamedModel &m : refused)
  {
    ASSERT_EQ(validateCameraModel(m.model), StatusCode::OK) << m.name;
    CameraInfo out = sentinel();
    EXPECT_EQ(exportCameraInfo(m.model, out), StatusCode::INVALID_MODEL) << m.name;
    expectUntouched(out, m.name);
  }

  CameraModel invalid = fromCameraInfo("radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01});
  invalid.intrinsics.fx = std::nanf("");
  CameraInfo out = sentinel();
  EXPECT_NE(exportCameraInfo(invalid, out), StatusCode::OK);
  expectUntouched(out, "invalid model");
}

// The double entry point writes the double coefficients as they are.
TEST(CameraInfoExportRos, DoublePrecisionKeepsEveryDigit)
{
  CameraInfo in;
  in.k = {400.1234567890123, 0.0, 320.5, 0.0, 410.9876543210987, 240.25, 0.0, 0.0, 1.0};
  in.distortion_model = "plumb_bob";
  in.d = {-0.12345678901234, 0.0123456789012, 1e-4, -2e-4, 0.001};
  const CameraModel64 model = makeCameraModel64(in);
  ASSERT_EQ(validateCameraModel64(model), StatusCode::OK);
  CameraInfo out;
  ASSERT_EQ(exportCameraInfo64(model, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
  EXPECT_EQ(out.d, in.d);
  EXPECT_EQ(out.k, in.k);
  EXPECT_EQ(makeCameraModel64(out).distortion.coeffs, model.distortion.coeffs);
}
