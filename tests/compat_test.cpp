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

// Tests for the ROS-free / OpenCV-free CameraInfo parser (src/compat/factory.cpp).
//
// makeCameraModel() reads a CameraInfo's K / D / distortion_model. These tests
// characterize its string parsing, the intrinsics mapping, double precision
// and the deterministic error paths. Core test (no Ceres/ROS/OpenCV). The
// export side is tested in camera_info_export_test.cpp.

#include "camxiom/compat.hpp"

#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

using namespace camxiom;

// Canonical row-major K layout makeCameraModel() reads (fx=k[0], skew=k[1],
// cx=k[2], fy=k[4], cy=k[5]); exportCameraInfo writes the same layout.
std::array<double, 9> makeK(double fx, double fy, double cx, double cy, double skew = 0.0)
{
  return {fx, skew, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
}

// ===========================================================================
// makeCameraModel(CameraInfo) — distortion-model string parsing + intrinsics
// ===========================================================================

TEST(CompatMakeCameraModel, IntrinsicsMapping)
{
  CameraInfo info;
  info.k = makeK(525.0, 524.0, 319.5, 239.5, 0.3);
  info.distortion_model = "plumb_bob";
  info.d = {-0.2, 0.05, 0.001, -0.001, 0.01};

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, DistortionModelType::RADTAN5);
  EXPECT_NEAR(model.intrinsics.fx, 525.0f, 1e-3);
  EXPECT_NEAR(model.intrinsics.fy, 524.0f, 1e-3);
  EXPECT_NEAR(model.intrinsics.cx, 319.5f, 1e-3);
  EXPECT_NEAR(model.intrinsics.cy, 239.5f, 1e-3);
  EXPECT_NEAR(model.intrinsics.skew, 0.3f, 1e-3);
  ASSERT_EQ(model.distortion.count, 5U);
  EXPECT_NEAR(model.distortion.coeffs[0], -0.2f, 1e-5);
}

TEST(CompatMakeCameraModel, RationalPolynomial)
{
  CameraInfo info;
  info.k = makeK(500.0, 500.0, 320.0, 240.0);
  info.distortion_model = "rational_polynomial";
  info.d = {-0.2, 0.05, 0.001, -0.001, 0.01, 0.001, 0.0005, 0.0002};

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, DistortionModelType::RATIONAL8);
}

TEST(CompatMakeCameraModel, EquidistantFisheye)
{
  CameraInfo info;
  info.k = makeK(300.0, 300.0, 320.0, 240.0);
  info.distortion_model = "equidistant";
  info.d = {0.01, 0.001, 0.0001, -0.0002};

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EQUIDISTANT);
  EXPECT_EQ(camxiom::distortionSpaceOf(model.distortion.type), DistortionSpace::ANGLE);
}

TEST(CompatMakeCameraModel, NamesSelectTheLawAndTheCorrection)
{
  struct Case
  {
    const char *name;
    std::vector<double> d;
    ProjectionModelType law;
    DistortionModelType distortion;
  };
  const std::vector<double> kb4 = {0.01, 0.001, 0.0001, -0.0002};
  const std::vector<double> kb8 = {0.01, 0.001, 0.0001, -0.0002, 0, 0, 0, 0};
  const Case cases[] = {
    {"equidistant", kb4, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {"fisheye", kb4, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {"opencv_fisheye4", kb4, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {"kb4", kb4, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {"kannala_brandt", kb4, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4},
    {"kb8", kb8, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB8},
    {"kannala_brandt8", kb8, ProjectionModelType::EQUIDISTANT, DistortionModelType::KB8},
    {"ideal_equidistant", {}, ProjectionModelType::EQUIDISTANT, DistortionModelType::NONE},
    {"equisolid", {}, ProjectionModelType::EQUISOLID, DistortionModelType::NONE},
    {"equisolid_angle", {}, ProjectionModelType::EQUISOLID, DistortionModelType::NONE},
    {"stereographic", {}, ProjectionModelType::STEREOGRAPHIC, DistortionModelType::NONE},
    {"stereographic_angle", {}, ProjectionModelType::STEREOGRAPHIC, DistortionModelType::NONE},
    {"orthographic", {}, ProjectionModelType::ORTHOGRAPHIC, DistortionModelType::NONE},
    {"orthographic_angle", {}, ProjectionModelType::ORTHOGRAPHIC, DistortionModelType::NONE},
    {"plumb_bob",
     {-0.1, 0.01, 0, 0, 0},
     ProjectionModelType::PINHOLE,
     DistortionModelType::RADTAN5},
  };
  for (const Case &c : cases)
  {
    CameraInfo info;
    info.k = makeK(300.0, 300.0, 320.0, 240.0);
    info.distortion_model = c.name;
    info.d = c.d;
    const CameraModel model = makeCameraModel(info);
    EXPECT_EQ(validateCameraModel(model), StatusCode::OK) << c.name;
    EXPECT_EQ(model.projection.type, c.law) << c.name;
    EXPECT_EQ(model.distortion.type, c.distortion) << c.name;
    EXPECT_EQ(model.distortion.count, c.d.size()) << c.name;
  }
}

TEST(CompatMakeCameraModel, ZeroKb4StaysKb4)
{
  // A zero KB4 is numerically the plain equidistant law, but it is a
  // correction with free coefficients (e.g. a calibration seed): it is
  // never folded into NONE.
  CameraInfo info;
  info.k = makeK(300.0, 300.0, 320.0, 240.0);
  info.distortion_model = "equidistant";
  info.d = {0.0, 0.0, 0.0, 0.0};
  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EQUIDISTANT);
  EXPECT_EQ(model.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(model.distortion.count, 4U);
}

TEST(CompatMakeCameraModel, PlainLawsDropZeroPaddingAndRejectCoefficients)
{
  CameraInfo info;
  info.k = makeK(300.0, 300.0, 320.0, 240.0);
  info.distortion_model = "equisolid";
  info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
  CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EQUISOLID);
  EXPECT_EQ(model.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(model.distortion.count, 0U);

  info.d[1] = 0.01;
  model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::INVALID_MODEL);
}

TEST(MakeDistortionFree, KeepsTheProjectionLaw)
{
  CameraInfo info;
  info.k = makeK(300.0, 300.0, 320.0, 240.0);
  info.distortion_model = "stereographic";
  const CameraModel stereographic = makeCameraModel(info);
  ASSERT_EQ(validateCameraModel(stereographic), StatusCode::OK);
  const CameraModel stereographic_free = makeDistortionFree(stereographic);
  EXPECT_EQ(stereographic_free.projection.type, ProjectionModelType::STEREOGRAPHIC);
  EXPECT_EQ(stereographic_free.projection.theta_max, stereographic.projection.theta_max);
  EXPECT_EQ(stereographic_free.distortion.type, DistortionModelType::NONE);

  info.distortion_model = "equidistant";
  info.d = {0.01, 0.001, 0.0001, -0.0002};
  const CameraModel kb4 = makeCameraModel(info);
  ASSERT_EQ(kb4.distortion.type, DistortionModelType::KB4);
  const CameraModel kb4_free = makeDistortionFree(kb4);
  EXPECT_EQ(kb4_free.projection.type, ProjectionModelType::EQUIDISTANT);
  EXPECT_EQ(kb4_free.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(kb4_free.distortion.count, 0U);
  EXPECT_EQ(validateCameraModel(kb4_free), StatusCode::OK);

  info.distortion_model = "plumb_bob";
  info.d = {-0.1, 0.01, 0.0, 0.0, 0.0};
  const CameraModel pinhole_free = makeDistortionFree(makeCameraModel(info));
  EXPECT_EQ(pinhole_free.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(pinhole_free.distortion.type, DistortionModelType::NONE);
}

TEST(CompatMakeCameraModel, DoubleSphereString)
{
  CameraInfo info;
  info.k = makeK(200.0, 200.0, 320.0, 240.0);
  info.distortion_model = "double_sphere";
  info.d = {-0.2, 0.5};  // xi, alpha

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_NEAR(model.projection.xi, -0.2f, 1e-5);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);
}

TEST(CompatMakeCameraModel, EucmString)
{
  CameraInfo info;
  info.k = makeK(200.0, 200.0, 320.0, 240.0);
  info.distortion_model = "eucm";
  info.d = {0.5, 1.0};  // alpha, beta

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EUCM);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);
  EXPECT_NEAR(model.projection.beta, 1.0f, 1e-5);
}

TEST(CompatMakeCameraModel, OmnidirectionalString)
{
  CameraInfo info;
  info.k = makeK(200.0, 200.0, 320.0, 240.0);
  info.distortion_model = "omnidirectional";
  info.d = {1.0};  // xi

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::OMNIDIRECTIONAL);
  EXPECT_NEAR(model.projection.xi, 1.0f, 1e-5);
}

TEST(CompatMakeCameraModel, EmptyStringNoDistortionIsPinhole)
{
  CameraInfo info;
  info.k = makeK(500.0, 500.0, 320.0, 240.0);
  info.distortion_model = "";
  info.d = {};

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(model.distortion.count, 0U);
}

// ===========================================================================
// Error / degenerate paths (deterministic status codes)
// ===========================================================================

TEST(CompatErrors, MakeCameraModelDoubleSphereMissingAlphaIsInvalid)
{
  // double_sphere needs xi + alpha; supplying only xi leaves alpha unset
  // (forced NaN) so the produced model must fail validation.
  CameraInfo info;
  info.k = makeK(200.0, 200.0, 320.0, 240.0);
  info.distortion_model = "double_sphere";
  info.d = {-0.2};  // missing alpha

  const CameraModel model = makeCameraModel(info);
  EXPECT_NE(validateCameraModel(model), StatusCode::OK);
}

TEST(CompatErrors, MakeCameraModelTooManyCoefficientsIsUnknown)
{
  CameraInfo info;
  info.k = makeK(500.0, 500.0, 320.0, 240.0);
  info.distortion_model = "plumb_bob";
  info.d.assign(15, 0.0);  // exceeds the 14-slot coefficient array

  const CameraModel model = makeCameraModel(info);
  EXPECT_EQ(model.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_NE(validateCameraModel(model), StatusCode::OK);
}

}  // namespace

TEST(CompatDouble, PreservesCameraInfoAcrossAllFamilies)
{
  const std::vector<std::pair<std::string, std::vector<double>>> cases = {
    {"plumb_bob", {-0.123456789012345, 0.0123456789012345, 0.000123456789, 0, 0}},
    {"rational_polynomial", {-0.1, 0.02, 0.000123456789, 0, 0, 0.001234567890123, 0, 0}},
    {"equidistant", {-0.0123456789012345, 0.00123456789012345, 0, 0}},
    {"omnidirectional", {1.123456789012345, -0.0123456789012345, 0, 0, 0}},
    {"double_sphere", {-0.123456789012345, 0.543210987654321}},
    {"eucm", {0.543210987654321, 1.123456789012345}}};
  for (const auto &entry : cases)
  {
    SCOPED_TRACE(entry.first);
    CameraInfo info;
    info.k = makeK(1234.56789012345, 1230.98765432109, 639.12345678901, 479.98765432109);
    info.distortion_model = entry.first;
    info.d = entry.second;
    auto model = makeCameraModel64(info);
    ASSERT_EQ(validateCameraModel64(model), StatusCode::OK);
    EXPECT_EQ(model.intrinsics.fx, info.k[0]);
    EXPECT_NE(model.intrinsics.fx, static_cast<double>(static_cast<float>(info.k[0])));
    // The ROS names hold these lenses exactly in double; the projection
    // parameters of the sphere models have no ROS name.
    CameraInfo out;
    out.width = 1280;
    out.height = 960;
    const StatusCode status = exportCameraInfo64(model, out);
    if (model.projection.type != ProjectionModelType::PINHOLE && model.projection.type != ProjectionModelType::EQUIDISTANT)
    {
      EXPECT_EQ(status, StatusCode::INVALID_MODEL);
    }
    else
    {
      ASSERT_EQ(status, StatusCode::OK);
      EXPECT_EQ(out.k, info.k);
      EXPECT_EQ(out.d, info.d);
      EXPECT_EQ(out.width, 1280u);
      EXPECT_EQ(out.height, 960u);
      const auto reloaded = makeCameraModel64(out);
      EXPECT_EQ(reloaded.intrinsics.fx, model.intrinsics.fx);
      EXPECT_EQ(reloaded.distortion.coeffs, model.distortion.coeffs);
    }
    updateThetaMax64(model);
    EXPECT_EQ(validateCameraModel64(model), StatusCode::OK);
    EXPECT_EQ(makeDistortionFree64(model).intrinsics.fx, info.k[0]);
  }
}
