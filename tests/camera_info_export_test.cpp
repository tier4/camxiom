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

// exportCameraInfo, both targets. ROS: a model is written only when a ROS
// distortion model holds the same lens, and every model written reads back
// (makeCameraModel) as one that projects every ray to the same pixel.
// camxiom's own description: every valid model is written and reads back as
// the same model.

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
#include <utility>
#include <vector>

namespace
{

using namespace camxiom;

struct NamedModel
{
  const char *name;
  CameraModel model;
};

// camxiom's description of a model with fixed intrinsics.
CamxiomCameraInfo describe(
  const std::string &law, const std::vector<double> &law_parameters, const std::string &correction,
  const std::vector<double> &coefficients
)
{
  CamxiomCameraInfo info;
  info.projection_model = law;
  info.projection_parameters = law_parameters;
  info.distortion_model = correction;
  info.distortion_parameters = coefficients;
  info.intrinsic_parameters = {400.0, 410.0, 320.0, 240.0, 0.0};
  return info;
}

CameraModel model(
  const std::string &law, const std::vector<double> &law_parameters, const std::string &correction,
  const std::vector<double> &coefficients
)
{
  return makeCameraModel(describe(law, law_parameters, correction, coefficients));
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
    {{"pinhole none", model("pinhole", {}, "none", {})}, "plumb_bob", {0.0, 0.0, 0.0, 0.0, 0.0}},
    {{"radtan4", model("pinhole", {}, "radtan4", {-0.2, 0.05, 0.001, -0.002})},
     "plumb_bob",
     {-0.2, 0.05, 0.001, -0.002, 0.0}},
    {{"radtan5", model("pinhole", {}, "radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01})},
     "plumb_bob",
     {-0.2, 0.05, 0.001, -0.002, 0.01}},
    {{"rational8", model("pinhole", {}, "rational8", rational)}, "rational_polynomial", rational},
    {{"thin_prism12 with zero prism", model("pinhole", {}, "thin_prism12", prism_zero)},
     "rational_polynomial",
     rational},
    {{"tilted14 with zero prism and tilt", model("pinhole", {}, "tilted14", tilt_zero)},
     "rational_polynomial",
     rational},
    {{"equidistant kb4", model("equidistant", {}, "kb4", kb4)}, "equidistant", kb4},
    {{"ideal equidistant", model("equidistant", {}, "none", {})},
     "equidistant",
     {0.0, 0.0, 0.0, 0.0}},
    {{"kb8 with zero k5..k8", model("equidistant", {}, "kb8", kb8_zero_tail)}, "equidistant", kb4},
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
    {"thin_prism12 with a prism term", model("pinhole", {}, "thin_prism12", prism)},
    {"tilted14 with a tilt term", model("pinhole", {}, "tilted14", tilt)},
    {"kb8 with a k7 term", model("equidistant", {}, "kb8", kb8)},
    {"equisolid", model("equisolid", {}, "none", {})},
    {"stereographic", model("stereographic", {}, "none", {})},
    {"orthographic", model("orthographic", {}, "none", {})},
    {"omnidirectional", model("omnidirectional", {0.8}, "radtan4", {-0.2, 0.05, 0.001, -0.001})},
    {"omnidirectional without correction", model("omnidirectional", {0.8}, "none", {})},
    {"double sphere", model("double_sphere", {-0.2, 0.6}, "none", {})},
    {"eucm", model("eucm", {0.6, 1.1}, "none", {})},
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

  CameraModel invalid = model("pinhole", {}, "radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01});
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

namespace
{

// Every law and correction combination a model can have, with values that
// are not round in float.
std::vector<std::pair<const char *, CamxiomCameraInfo>> everyModel()
{
  std::vector<double> rational = sequence(8U, -0.21, 0.031);
  std::vector<double> prism = sequence(12U, -0.21, 0.031);
  std::vector<double> tilt = sequence(14U, -0.21, 0.017);
  const std::vector<double> kb4{0.021, -0.0043, 0.00081, -0.00012};
  const std::vector<double> kb8{0.021, -0.0043, 0.00081, -0.00012, 1.3e-5, -2.1e-6, 3.0e-7, 0.0};
  return {
    {"pinhole none", describe("pinhole", {}, "none", {})},
    {"pinhole radtan4", describe("pinhole", {}, "radtan4", {-0.21, 0.051, 0.0013, -0.0021})},
    {"pinhole radtan5", describe("pinhole", {}, "radtan5", {-0.21, 0.051, 0.0013, -0.0021, 0.011})},
    {"pinhole rational8", describe("pinhole", {}, "rational8", rational)},
    {"pinhole thin_prism12", describe("pinhole", {}, "thin_prism12", prism)},
    {"pinhole tilted14", describe("pinhole", {}, "tilted14", tilt)},
    {"equidistant none", describe("equidistant", {}, "none", {})},
    {"equidistant kb4", describe("equidistant", {}, "kb4", kb4)},
    {"equidistant kb4 at zero", describe("equidistant", {}, "kb4", {0.0, 0.0, 0.0, 0.0})},
    {"equidistant kb8", describe("equidistant", {}, "kb8", kb8)},
    {"equisolid", describe("equisolid", {}, "none", {})},
    {"stereographic", describe("stereographic", {}, "none", {})},
    {"orthographic", describe("orthographic", {}, "none", {})},
    {"omnidirectional none", describe("omnidirectional", {0.83}, "none", {})},
    {"omnidirectional radtan4",
     describe("omnidirectional", {0.83}, "radtan4", {-0.21, 0.051, 0.0013, -0.0021})},
    {"double sphere none", describe("double_sphere", {-0.21, 0.61}, "none", {})},
    {"double sphere radtan5",
     describe(
       "double_sphere", {-0.21, 0.61}, "radtan5", {-0.021, 0.0051, 0.0013, -0.0021, 0.0011}
     )},
    {"eucm none", describe("eucm", {0.61, 1.13}, "none", {})},
    {"eucm rational8", describe("eucm", {0.61, 1.13}, "rational8", sequence(8U, -0.021, 0.0031))},
    {"equirectangular", describe("equirectangular", {}, "none", {})},
    {"cylindrical", describe("cylindrical", {}, "none", {})},
  };
}

template <typename T>
void expectSameModel(const CameraModelT<T> &a, const CameraModelT<T> &b, const char *what)
{
  EXPECT_EQ(a.intrinsics.fx, b.intrinsics.fx) << what;
  EXPECT_EQ(a.intrinsics.fy, b.intrinsics.fy) << what;
  EXPECT_EQ(a.intrinsics.cx, b.intrinsics.cx) << what;
  EXPECT_EQ(a.intrinsics.cy, b.intrinsics.cy) << what;
  EXPECT_EQ(a.intrinsics.skew, b.intrinsics.skew) << what;
  EXPECT_EQ(a.projection.type, b.projection.type) << what;
  EXPECT_EQ(a.projection.xi, b.projection.xi) << what;
  EXPECT_EQ(a.projection.alpha, b.projection.alpha) << what;
  EXPECT_EQ(a.projection.beta, b.projection.beta) << what;
  EXPECT_EQ(a.projection.theta_max, b.projection.theta_max) << what;
  EXPECT_EQ(a.distortion.type, b.distortion.type) << what;
  EXPECT_EQ(a.distortion.count, b.distortion.count) << what;
  EXPECT_EQ(a.distortion.coeffs, b.distortion.coeffs) << what;
}

}  // namespace

// Every valid model is written, field by field under the toString names, and
// reads back as the same model, in float and in double.
TEST(CameraInfoExportCamxiom, WritesAndReadsBackEveryModel)
{
  for (const auto &[what, description] : everyModel())
  {
    const CameraModel source = makeCameraModel(description);
    ASSERT_EQ(validateCameraModel(source), StatusCode::OK) << what;
    CamxiomCameraInfo out;
    out.image_width = 7U;
    out.image_height = 5U;
    ASSERT_EQ(exportCameraInfo(source, out), StatusCode::OK) << what;
    EXPECT_EQ(out.projection_model, description.projection_model) << what;
    EXPECT_EQ(out.distortion_model, description.distortion_model) << what;
    EXPECT_EQ(out.projection_parameters.size(), description.projection_parameters.size()) << what;
    EXPECT_EQ(out.distortion_parameters.size(), description.distortion_parameters.size()) << what;
    EXPECT_EQ(out.image_width, 7U) << what;  // the image size is the caller's
    EXPECT_EQ(out.image_height, 5U) << what;
    expectSameModel(makeCameraModel(out), source, what);

    const CameraModel64 source64 = makeCameraModel64(description);
    ASSERT_EQ(validateCameraModel64(source64), StatusCode::OK) << what;
    CamxiomCameraInfo out64;
    ASSERT_EQ(exportCameraInfo64(source64, out64), StatusCode::OK) << what;
    // Double values are written as they are.
    EXPECT_EQ(out64.projection_parameters, description.projection_parameters) << what;
    EXPECT_EQ(out64.distortion_parameters, description.distortion_parameters) << what;
    EXPECT_EQ(out64.intrinsic_parameters, description.intrinsic_parameters) << what;
    expectSameModel(makeCameraModel64(out64), source64, what);
  }
}

// A RADTAN4 model may carry k3 in a fifth coefficient, which projection
// evaluates; it is written as the radtan5 lens it is, not truncated.
TEST(CameraInfoExportCamxiom, WritesAFiveTermRadtan4AsRadtan5)
{
  CameraModel source;
  source.intrinsics.fx = 400.0f;
  source.intrinsics.fy = 400.0f;
  source.projection.type = ProjectionModelType::PINHOLE;
  source.distortion.type = DistortionModelType::RADTAN4;
  source.distortion.count = 5U;
  source.distortion.coeffs = {-0.2f, 0.05f, 0.001f, -0.002f, 0.01f};
  ASSERT_EQ(validateCameraModel(source), StatusCode::OK);
  CamxiomCameraInfo out;
  ASSERT_EQ(exportCameraInfo(source, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "radtan5");
  EXPECT_EQ(out.distortion_parameters, (std::vector<double>{-0.2, 0.05, 0.001, -0.002, 0.01}));
  const CameraModel back = makeCameraModel(out);
  ASSERT_EQ(validateCameraModel(back), StatusCode::OK);
  EXPECT_EQ(back.distortion.coeffs, source.distortion.coeffs);
  expectSameLens(source, back, "radtan4 with k3");
  // The ray (1, 0, 1) shows k3: 404 px with it, 400 px without.
  const PixelResult pixel = rayToPixel(back, Eigen::Vector3f(1.0f, 0.0f, 1.0f));
  ASSERT_EQ(pixel.status, StatusCode::OK);
  EXPECT_NEAR(pixel.pixel.u, rayToPixel(source, Eigen::Vector3f(1.0f, 0.0f, 1.0f)).pixel.u, 1e-3f);
}

// A float model is written as the shortest decimal that reads back as the
// same float: 0.021 rather than 0.020999999716877937.
TEST(CameraInfoExportCamxiom, FloatValuesAreWrittenAsShortDecimals)
{
  const CameraModel source =
    makeCameraModel(describe("pinhole", {}, "radtan5", {-0.21, 0.051, 0.0013, -0.0021, 0.011}));
  CamxiomCameraInfo out;
  ASSERT_EQ(exportCameraInfo(source, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_parameters, (std::vector<double>{-0.21, 0.051, 0.0013, -0.0021, 0.011}));
  EXPECT_EQ(out.intrinsic_parameters, (std::array<double, 5>{400.0, 410.0, 320.0, 240.0, 0.0}));
}

// A description that is not a valid model reads as one validation rejects:
// unknown names, parameter counts that do not match the law or the
// correction, and combinations no model has.
TEST(CameraInfoExportCamxiom, RejectsDescriptionsThatAreNotModels)
{
  const std::vector<std::pair<const char *, CamxiomCameraInfo>> invalid{
    {"unknown law", describe("fisheye_theta", {}, "none", {})},
    {"unknown correction", describe("pinhole", {}, "radtan6", {})},
    {"empty law", describe("", {}, "none", {})},
    {"missing law parameter", describe("double_sphere", {-0.2}, "none", {})},
    {"extra law parameter", describe("pinhole", {0.5}, "none", {})},
    {"too few coefficients", describe("pinhole", {}, "radtan5", {-0.2, 0.05, 0.001, -0.002})},
    {"coefficients without a correction", describe("equisolid", {}, "none", {0.0, 0.0, 0.0, 0.0})},
    {"kb4 on pinhole", describe("pinhole", {}, "kb4", {0.02, -0.004, 0.0008, -0.0001})},
    {"radtan5 on the equidistant law",
     describe("equidistant", {}, "radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01})},
    {"correction on equirectangular",
     describe("equirectangular", {}, "radtan4", {-0.2, 0.05, 0.001, -0.002})},
  };
  for (const auto &[what, description] : invalid)
  {
    EXPECT_NE(validateCameraModel(makeCameraModel(description)), StatusCode::OK) << what;
    EXPECT_NE(validateCameraModel64(makeCameraModel64(description)), StatusCode::OK) << what;
  }
}

TEST(CameraInfoExportCamxiom, RefusesAnInvalidModelWithoutWriting)
{
  CameraModel invalid = model("pinhole", {}, "radtan5", {-0.2, 0.05, 0.001, -0.002, 0.01});
  invalid.intrinsics.fy = std::nanf("");
  CamxiomCameraInfo out;
  out.projection_model = "untouched";
  EXPECT_NE(exportCameraInfo(invalid, out), StatusCode::OK);
  EXPECT_EQ(out.projection_model, "untouched");
  EXPECT_TRUE(out.distortion_model.empty());
}
