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

// Tests for the ROS-free / OpenCV-free compat interchange layer
// (src/compat/factory.cpp + src/<model>/compat.cpp).
//
// makeCameraModel() is the single canonical K/D/distortion-model-string parser
// that every adapter (ROS, OpenCV) funnels through, and the per-model
// import/export*Model functions are the profile-aware conversions used by the
// interop layers. Until now none of this had a direct test (LOADMAP T3).
//
// Strategy: for every model family, round-trip an external description through
// import -> CameraModel -> export and assert it comes back (K + coefficients +
// model-specific scalars), plus characterize the makeCameraModel() string
// parsing and the deterministic error paths. Core test (no Ceres/ROS/OpenCV).

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

// Canonical row-major K layout used by both makeCameraModel() (reads
// fx=k[0], skew=k[1], cx=k[2], fy=k[4], cy=k[5]) and the export path
// (fillProjectionFromIntrinsics writes exactly this layout back).
std::array<double, 9> makeK(double fx, double fy, double cx, double cy, double skew = 0.0)
{
  return {fx, skew, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
}

void expectKNear(
  const std::array<double, 9> &got, const std::array<double, 9> &want, double tol = 1e-3
)
{
  for (std::size_t i = 0; i < 9; ++i)
  {
    EXPECT_NEAR(got[i], want[i], tol) << "K[" << i << "]";
  }
}

void expectDNear(const std::vector<double> &got, const std::vector<double> &want, double tol = 1e-5)
{
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
  {
    EXPECT_NEAR(got[i], want[i], tol) << "D[" << i << "]";
  }
}

// ===========================================================================
// import -> export round-trips (per model family)
// ===========================================================================

TEST(CompatRoundTrip, PinholeRadTan5)
{
  PinholeExternalModel ext;
  ext.K = makeK(525.0, 524.0, 319.5, 239.5);
  ext.D = {-0.2, 0.05, 0.001, -0.001, 0.01};

  CameraModel model;
  ASSERT_EQ(
    importPinholeModel(ext, PinholeCompatProfile::OPENCV_CALIB3D_D5, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);

  PinholeExternalModel out;
  ASSERT_EQ(
    exportPinholeModel(model, PinholeCompatProfile::OPENCV_CALIB3D_D5, out), StatusCode::OK
  );
  expectKNear(out.K, ext.K);
  expectDNear(out.D, ext.D);
}

TEST(CompatRoundTrip, PinholeRational8)
{
  PinholeExternalModel ext;
  ext.K = makeK(525.0, 524.0, 319.5, 239.5);
  ext.D = {-0.2, 0.05, 0.001, -0.001, 0.01, 0.001, 0.0005, 0.0002};

  CameraModel model;
  ASSERT_EQ(
    importPinholeModel(ext, PinholeCompatProfile::OPENCV_CALIB3D_D8, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, DistortionModelType::RATIONAL8);

  PinholeExternalModel out;
  ASSERT_EQ(
    exportPinholeModel(model, PinholeCompatProfile::OPENCV_CALIB3D_D8, out), StatusCode::OK
  );
  expectKNear(out.K, ext.K);
  expectDNear(out.D, ext.D);
}

TEST(CompatRoundTrip, FisheyeKB4)
{
  FisheyeExternalModel ext;
  ext.K = makeK(300.0, 300.0, 320.0, 240.0);
  ext.D = {0.01, 0.001, 0.0001, -0.0002};
  ext.distortion_type = DistortionModelType::KB4;

  CameraModel model;
  ASSERT_EQ(importFisheyeModel(ext, FisheyeCompatProfile::CANONICAL, model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_EQ(model.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(model.distortion.space, DistortionSpace::ANGLE);

  FisheyeExternalModel out;
  ASSERT_EQ(exportFisheyeModel(model, FisheyeCompatProfile::CANONICAL, out), StatusCode::OK);
  expectKNear(out.K, ext.K);
  expectDNear(out.D, ext.D);
  EXPECT_EQ(out.distortion_type, DistortionModelType::KB4);
}

TEST(CompatRoundTrip, OmnidirectionalMei)
{
  OmnidirectionalExternalModel ext;
  ext.K = makeK(200.0, 200.0, 320.0, 240.0);
  ext.D = {};
  ext.xi = 1.0;

  CameraModel model;
  ASSERT_EQ(
    importOmnidirectionalModel(ext, OmnidirectionalCompatProfile::CANONICAL, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::OMNIDIRECTIONAL);
  EXPECT_NEAR(model.projection.xi, 1.0f, 1e-5);

  OmnidirectionalExternalModel out;
  ASSERT_EQ(
    exportOmnidirectionalModel(model, OmnidirectionalCompatProfile::CANONICAL, out), StatusCode::OK
  );
  expectKNear(out.K, ext.K);
  EXPECT_NEAR(out.xi, ext.xi, 1e-5);
  EXPECT_TRUE(out.D.empty());
}

TEST(CompatRoundTrip, DoubleSphere)
{
  DoubleSphereExternalModel ext;
  ext.K = makeK(200.0, 200.0, 320.0, 240.0);
  ext.D = {};
  ext.xi = -0.2;
  ext.alpha = 0.5;

  CameraModel model;
  ASSERT_EQ(
    importDoubleSphereModel(ext, DoubleSphereCompatProfile::BASALT_D0, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_NEAR(model.projection.xi, -0.2f, 1e-5);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);

  DoubleSphereExternalModel out;
  ASSERT_EQ(
    exportDoubleSphereModel(model, DoubleSphereCompatProfile::BASALT_D0, out), StatusCode::OK
  );
  expectKNear(out.K, ext.K);
  EXPECT_NEAR(out.xi, ext.xi, 1e-5);
  EXPECT_NEAR(out.alpha, ext.alpha, 1e-5);
}

TEST(CompatRoundTrip, Eucm)
{
  EucmExternalModel ext;
  ext.K = makeK(200.0, 200.0, 320.0, 240.0);
  ext.D = {};
  ext.alpha = 0.5;
  ext.beta = 1.0;

  CameraModel model;
  ASSERT_EQ(importEucmModel(ext, EucmCompatProfile::KALIBR_D0, model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EUCM);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);
  EXPECT_NEAR(model.projection.beta, 1.0f, 1e-5);

  EucmExternalModel out;
  ASSERT_EQ(exportEucmModel(model, EucmCompatProfile::KALIBR_D0, out), StatusCode::OK);
  expectKNear(out.K, ext.K);
  EXPECT_NEAR(out.alpha, ext.alpha, 1e-5);
  EXPECT_NEAR(out.beta, ext.beta, 1e-5);
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
  EXPECT_EQ(model.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_EQ(model.distortion.space, DistortionSpace::ANGLE);
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

TEST(CompatErrors, ImportPinholeProfileCountMismatch)
{
  PinholeExternalModel ext;
  ext.K = makeK(500.0, 500.0, 320.0, 240.0);
  ext.D = {-0.2, 0.05, 0.001, -0.001};  // 4 coeffs, but D5 profile expects 5

  CameraModel model;
  EXPECT_EQ(
    importPinholeModel(ext, PinholeCompatProfile::OPENCV_CALIB3D_D5, model),
    StatusCode::INVALID_INPUT
  );
}

TEST(CompatErrors, ImportPinholeNonFiniteK)
{
  PinholeExternalModel ext;
  ext.K = makeK(500.0, 500.0, 320.0, 240.0);
  ext.K[0] = std::numeric_limits<double>::quiet_NaN();
  ext.D = {};

  CameraModel model;
  EXPECT_EQ(
    importPinholeModel(ext, PinholeCompatProfile::CANONICAL, model), StatusCode::INVALID_INPUT
  );
}

TEST(CompatErrors, ImportPinholeTooManyCoefficients)
{
  PinholeExternalModel ext;
  ext.K = makeK(500.0, 500.0, 320.0, 240.0);
  ext.D.assign(15, 0.0);  // > 14 max

  CameraModel model;
  EXPECT_EQ(
    importPinholeModel(ext, PinholeCompatProfile::CANONICAL, model), StatusCode::INVALID_INPUT
  );
}

TEST(CompatErrors, ImportEucmNonPositiveBeta)
{
  EucmExternalModel ext;
  ext.K = makeK(200.0, 200.0, 320.0, 240.0);
  ext.alpha = 0.5;
  ext.beta = 0.0;  // beta must be > 0

  CameraModel model;
  EXPECT_EQ(importEucmModel(ext, EucmCompatProfile::KALIBR_D0, model), StatusCode::INVALID_INPUT);
}

TEST(CompatErrors, ImportFisheyeTooManyCanonicalCoefficients)
{
  FisheyeExternalModel ext;
  ext.K = makeK(300.0, 300.0, 320.0, 240.0);
  ext.D.assign(9, 0.0);  // CANONICAL max is 8
  ext.distortion_type = DistortionModelType::KB8;

  CameraModel model;
  EXPECT_EQ(
    importFisheyeModel(ext, FisheyeCompatProfile::CANONICAL, model), StatusCode::INVALID_INPUT
  );
}

TEST(CompatErrors, ExportWrongModelFamilyRejected)
{
  // Build a valid fisheye model, then try to export it through the pinhole /
  // EUCM paths: the projection-type guard must reject it.
  FisheyeExternalModel fisheye_ext;
  fisheye_ext.K = makeK(300.0, 300.0, 320.0, 240.0);
  fisheye_ext.D = {0.01, 0.001, 0.0001, -0.0002};
  fisheye_ext.distortion_type = DistortionModelType::KB4;

  CameraModel fisheye_model;
  ASSERT_EQ(
    importFisheyeModel(fisheye_ext, FisheyeCompatProfile::CANONICAL, fisheye_model), StatusCode::OK
  );

  PinholeExternalModel pinhole_out;
  EXPECT_EQ(
    exportPinholeModel(fisheye_model, PinholeCompatProfile::CANONICAL, pinhole_out),
    StatusCode::INVALID_MODEL
  );

  EucmExternalModel eucm_out;
  EXPECT_EQ(
    exportEucmModel(fisheye_model, EucmCompatProfile::CANONICAL, eucm_out),
    StatusCode::INVALID_MODEL
  );
}

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

// ===========================================================================
// export*CameraInfo(model) -> CameraInfo POD — the canonical CameraInfo
// packing (src/compat/camera_info_export.cpp). The ROS layer delegates here,
// so this block is the dependency-free guard on the packing itself:
// per-family distortion_model string / d layout, identity r, p = [K | 0],
// and width/height left untouched.
// ===========================================================================

void expectIdentityR(const CameraInfo &info)
{
  const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (std::size_t i = 0; i < 9; ++i)
  {
    EXPECT_NEAR(info.r[i], identity[i], 1e-9) << "r[" << i << "]";
  }
}

void expectProjectionMatrix(
  const CameraInfo &info, double fx, double fy, double cx, double cy, double tol = 1e-3
)
{
  EXPECT_NEAR(info.p[0], fx, tol) << "p[0] (fx)";
  EXPECT_NEAR(info.p[2], cx, tol) << "p[2] (cx)";
  EXPECT_NEAR(info.p[5], fy, tol) << "p[5] (fy)";
  EXPECT_NEAR(info.p[6], cy, tol) << "p[6] (cy)";
  EXPECT_NEAR(info.p[10], 1.0, 1e-9) << "p[10]";
  EXPECT_NEAR(info.p[3], 0.0, 1e-9) << "p[3] (Tx)";
  EXPECT_NEAR(info.p[7], 0.0, 1e-9) << "p[7] (Ty)";
}

TEST(CompatCameraInfoExport, PinholeCanonical)
{
  CameraInfo in;
  in.k = makeK(525.0, 524.0, 319.5, 239.5);
  in.distortion_model = "plumb_bob";
  in.d = {-0.2, 0.05, 0.001, -0.001, 0.01};
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  out.width = 640U;  // the export must not touch the image geometry
  out.height = 480U;
  ASSERT_EQ(exportPinholeCameraInfo(model, PinholeCompatProfile::CANONICAL, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
  expectKNear(out.k, in.k);
  expectDNear(out.d, in.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 525.0, 524.0, 319.5, 239.5);
  EXPECT_EQ(out.width, 640U);
  EXPECT_EQ(out.height, 480U);
}

TEST(CompatCameraInfoExport, FisheyeKb4Canonical)
{
  CameraInfo in;
  in.k = makeK(300.0, 300.0, 320.0, 240.0);
  in.distortion_model = "kb4";
  in.d = {0.01, 0.001, 0.0001, -0.0002};
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  ASSERT_EQ(exportFisheyeCameraInfo(model, FisheyeCompatProfile::CANONICAL, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "kb4");
  expectKNear(out.k, in.k);
  expectDNear(out.d, in.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 300.0, 300.0, 320.0, 240.0);
}

TEST(CompatCameraInfoExport, OmnidirectionalCanonicalPacksXiFirst)
{
  CameraInfo in;
  in.k = makeK(200.0, 200.0, 320.0, 240.0);
  in.distortion_model = "omnidirectional";
  in.d = {1.0};  // xi only
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  ASSERT_EQ(
    exportOmnidirectionalCameraInfo(model, OmnidirectionalCompatProfile::CANONICAL, out),
    StatusCode::OK
  );
  EXPECT_EQ(out.distortion_model, "omnidirectional");
  expectKNear(out.k, in.k);
  ASSERT_GE(out.d.size(), 1U);
  EXPECT_NEAR(out.d[0], 1.0, 1e-5);
  expectIdentityR(out);
}

TEST(CompatCameraInfoExport, DoubleSpherePacksXiAlpha)
{
  CameraInfo in;
  in.k = makeK(200.0, 200.0, 320.0, 240.0);
  in.distortion_model = "double_sphere";
  in.d = {-0.2, 0.5};
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  ASSERT_EQ(
    exportDoubleSphereCameraInfo(model, DoubleSphereCompatProfile::BASALT_D0, out), StatusCode::OK
  );
  EXPECT_EQ(out.distortion_model, "double_sphere");
  expectKNear(out.k, in.k);
  ASSERT_EQ(out.d.size(), 2U);
  EXPECT_NEAR(out.d[0], -0.2, 1e-5);
  EXPECT_NEAR(out.d[1], 0.5, 1e-5);
  expectIdentityR(out);
}

TEST(CompatCameraInfoExport, EucmPacksAlphaBeta)
{
  CameraInfo in;
  in.k = makeK(200.0, 200.0, 320.0, 240.0);
  in.distortion_model = "eucm";
  in.d = {0.5, 1.0};
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  ASSERT_EQ(exportEucmCameraInfo(model, EucmCompatProfile::KALIBR_D0, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "eucm");
  expectKNear(out.k, in.k);
  ASSERT_EQ(out.d.size(), 2U);
  EXPECT_NEAR(out.d[0], 0.5, 1e-5);
  EXPECT_NEAR(out.d[1], 1.0, 1e-5);
  expectIdentityR(out);
}

TEST(CompatCameraInfoExport, WrongFamilyRejected)
{
  CameraInfo in;
  in.k = makeK(300.0, 300.0, 320.0, 240.0);
  in.distortion_model = "kb4";
  in.d = {0.01, 0.001, 0.0001, -0.0002};
  const CameraModel fisheye_model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(fisheye_model), StatusCode::OK);

  CameraInfo out;
  EXPECT_EQ(
    exportPinholeCameraInfo(fisheye_model, PinholeCompatProfile::CANONICAL, out),
    StatusCode::INVALID_MODEL
  );
}

TEST(CompatCameraInfoExport, RoundTripsThroughMakeCameraModel)
{
  // export -> makeCameraModel must reproduce the family and intrinsics: the
  // canonical strings the exporters emit have to stay parseable by the core.
  CameraInfo in;
  in.k = makeK(200.0, 200.0, 320.0, 240.0);
  in.distortion_model = "double_sphere";
  in.d = {-0.2, 0.5};
  const CameraModel model = makeCameraModel(in);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  CameraInfo out;
  ASSERT_EQ(
    exportDoubleSphereCameraInfo(model, DoubleSphereCompatProfile::BASALT_D0, out), StatusCode::OK
  );
  const CameraModel reparsed = makeCameraModel(out);
  EXPECT_EQ(validateCameraModel(reparsed), StatusCode::OK);
  EXPECT_EQ(reparsed.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_NEAR(reparsed.projection.xi, model.projection.xi, 1e-6);
  EXPECT_NEAR(reparsed.projection.alpha, model.projection.alpha, 1e-6);
  EXPECT_NEAR(reparsed.intrinsics.fx, model.intrinsics.fx, 1e-4);
  EXPECT_NEAR(reparsed.intrinsics.cy, model.intrinsics.cy, 1e-4);
}

}  // namespace
