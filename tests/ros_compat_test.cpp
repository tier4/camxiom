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

// Tests for the optional ROS interoperability layer (src/compat/ros.cpp).
//
// This layer is a thin sensor_msgs adapter: every conversion copies the
// message's k/d/distortion_model/r/p/size fields into the ROS-free
// camxiom::CameraInfo POD and delegates the actual K/D parsing to the core
// compat layer (makeCameraModel / import*Model / export*Model). Until now
// nothing in the suite included <sensor_msgs/...>, so this bridge had zero
// regression coverage (LOADMAP T5). The POD path itself is covered by
// compat_test (T3); here we guard the sensor_msgs <-> CameraModel field wiring.
//
// Strategy: (1) makeCameraModel(msg) dispatches by distortion_model string;
// (2) per family, a hand-authored CameraInfo message round-trips
// message -> import -> CameraModel -> export -> message with K/D preserved and
// the rectification/projection matrices filled; (3) deterministic error paths.
//
// Only built when the ROS compat layer was compiled in (CMake gate
// CAMXIOM_WITH_ROS); the __has_include guard keeps the TU harmless otherwise.

#include <gtest/gtest.h>

#if __has_include(<sensor_msgs/msg/camera_info.hpp>)

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/ros.hpp"
#include "camxiom/types.hpp"

#include <sensor_msgs/msg/camera_info.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{

using namespace camxiom;

using RosCameraInfo = sensor_msgs::msg::CameraInfo;

// Canonical row-major K (fx=k[0], skew=k[1], cx=k[2], fy=k[4], cy=k[5]).
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
    EXPECT_NEAR(got[i], want[i], tol) << "k[" << i << "]";
  }
}

void expectDNear(const std::vector<double> &got, const std::vector<double> &want, double tol = 1e-4)
{
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
  {
    EXPECT_NEAR(got[i], want[i], tol) << "d[" << i << "]";
  }
}

// export*CameraInfo always writes an identity rectification and a 3x4
// projection matrix built from the pinhole intrinsics (fx,cx / fy,cy).
void expectIdentityR(const RosCameraInfo &msg)
{
  const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (std::size_t i = 0; i < 9; ++i)
  {
    EXPECT_NEAR(msg.r[i], identity[i], 1e-9) << "r[" << i << "]";
  }
}

void expectProjectionMatrix(
  const RosCameraInfo &msg, double fx, double fy, double cx, double cy, double tol = 1e-3
)
{
  EXPECT_NEAR(msg.p[0], fx, tol) << "p[0] (fx)";
  EXPECT_NEAR(msg.p[2], cx, tol) << "p[2] (cx)";
  EXPECT_NEAR(msg.p[5], fy, tol) << "p[5] (fy)";
  EXPECT_NEAR(msg.p[6], cy, tol) << "p[6] (cy)";
  EXPECT_NEAR(msg.p[10], 1.0, 1e-9) << "p[10]";
  EXPECT_NEAR(msg.p[3], 0.0, 1e-9) << "p[3] (Tx)";
  EXPECT_NEAR(msg.p[7], 0.0, 1e-9) << "p[7] (Ty)";
}

}  // namespace

// ===========================================================================
// makeCameraModel(sensor_msgs::CameraInfo): distortion_model string dispatch.
// This is the ROS-message analogue of the POD test in compat_test; it proves
// toCameraInfo() faithfully copies the message into the core parser.
// ===========================================================================

TEST(RosCompatMakeCameraModel, PlumbBobIsPinholeRadtan5)
{
  RosCameraInfo msg;
  msg.k = makeK(525.0, 524.0, 319.5, 239.5, 0.3);
  msg.distortion_model = "plumb_bob";
  msg.d = {-0.2, 0.05, 0.001, -0.001, 0.01};
  msg.width = 640U;
  msg.height = 480U;

  const CameraModel model = makeCameraModel(msg);
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

TEST(RosCompatMakeCameraModel, DoubleSphereString)
{
  RosCameraInfo msg;
  msg.k = makeK(200.0, 200.0, 320.0, 240.0);
  msg.distortion_model = "double_sphere";
  msg.d = {-0.2, 0.5};  // xi, alpha

  const CameraModel model = makeCameraModel(msg);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_NEAR(model.projection.xi, -0.2f, 1e-5);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);
}

TEST(RosCompatMakeCameraModel, EmptyStringIsPinholeNoDistortion)
{
  RosCameraInfo msg;
  msg.k = makeK(500.0, 500.0, 320.0, 240.0);
  msg.distortion_model = "";
  msg.d = {};

  const CameraModel model = makeCameraModel(msg);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(model.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(model.distortion.count, 0U);
}

// ===========================================================================
// message -> import -> CameraModel -> export -> message round-trips.
// ===========================================================================

TEST(RosCompatRoundTrip, PinholeRadtan5)
{
  RosCameraInfo msg;
  msg.k = makeK(525.0, 524.0, 319.5, 239.5);
  msg.d = {-0.2, 0.05, 0.001, -0.001, 0.01};

  CameraModel model;
  ASSERT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::OPENCV_CALIB3D_D5, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);

  RosCameraInfo out;
  ASSERT_EQ(
    exportPinholeCameraInfo(model, PinholeCompatProfile::OPENCV_CALIB3D_D5, out), StatusCode::OK
  );
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 525.0, 524.0, 319.5, 239.5);
  EXPECT_FALSE(out.distortion_model.empty());
}

TEST(RosCompatRoundTrip, FisheyeOpencvD4)
{
  RosCameraInfo msg;
  msg.k = makeK(300.0, 300.0, 320.0, 240.0);
  msg.d = {0.01, 0.001, 0.0001, -0.0002};

  CameraModel model;
  ASSERT_EQ(
    importFisheyeCameraInfo(msg, FisheyeCompatProfile::OPENCV_FISHEYE_D4, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_EQ(model.distortion.space, DistortionSpace::ANGLE);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);

  RosCameraInfo out;
  ASSERT_EQ(
    exportFisheyeCameraInfo(model, FisheyeCompatProfile::OPENCV_FISHEYE_D4, out), StatusCode::OK
  );
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 300.0, 300.0, 320.0, 240.0);
}

TEST(RosCompatRoundTrip, OmnidirectionalMei)
{
  RosCameraInfo msg;
  msg.k = makeK(200.0, 200.0, 320.0, 240.0);
  msg.distortion_model = "omnidirectional";
  msg.d = {1.0};  // xi

  CameraModel model;
  ASSERT_EQ(
    importOmnidirectionalCameraInfo(msg, OmnidirectionalCompatProfile::CANONICAL, model),
    StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::OMNIDIRECTIONAL);
  EXPECT_NEAR(model.projection.xi, 1.0f, 1e-5);
  EXPECT_EQ(validateCameraModel(model), StatusCode::OK);

  RosCameraInfo out;
  ASSERT_EQ(
    exportOmnidirectionalCameraInfo(model, OmnidirectionalCompatProfile::CANONICAL, out),
    StatusCode::OK
  );
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  EXPECT_EQ(out.distortion_model, "omnidirectional");
  expectIdentityR(out);
  expectProjectionMatrix(out, 200.0, 200.0, 320.0, 240.0);
}

TEST(RosCompatRoundTrip, DoubleSphereBasalt)
{
  RosCameraInfo msg;
  msg.k = makeK(200.0, 200.0, 320.0, 240.0);
  msg.d = {-0.2, 0.5};  // xi, alpha

  CameraModel model;
  ASSERT_EQ(
    importDoubleSphereCameraInfo(msg, DoubleSphereCompatProfile::BASALT_D0, model), StatusCode::OK
  );
  EXPECT_EQ(model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_NEAR(model.projection.xi, -0.2f, 1e-5);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);

  RosCameraInfo out;
  ASSERT_EQ(
    exportDoubleSphereCameraInfo(model, DoubleSphereCompatProfile::BASALT_D0, out), StatusCode::OK
  );
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  EXPECT_EQ(out.distortion_model, "double_sphere");
  expectIdentityR(out);
  expectProjectionMatrix(out, 200.0, 200.0, 320.0, 240.0);
}

TEST(RosCompatRoundTrip, Eucm)
{
  RosCameraInfo msg;
  msg.k = makeK(200.0, 200.0, 320.0, 240.0);
  msg.d = {0.5, 1.0};  // alpha, beta

  CameraModel model;
  ASSERT_EQ(importEucmCameraInfo(msg, EucmCompatProfile::KALIBR_D0, model), StatusCode::OK);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EUCM);
  EXPECT_NEAR(model.projection.alpha, 0.5f, 1e-5);
  EXPECT_NEAR(model.projection.beta, 1.0f, 1e-5);

  RosCameraInfo out;
  ASSERT_EQ(exportEucmCameraInfo(model, EucmCompatProfile::KALIBR_D0, out), StatusCode::OK);
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  EXPECT_EQ(out.distortion_model, "eucm");
  expectIdentityR(out);
  expectProjectionMatrix(out, 200.0, 200.0, 320.0, 240.0);
}

// ===========================================================================
// Deterministic error paths.
// ===========================================================================

TEST(RosCompatErrors, RejectsNonFiniteK)
{
  RosCameraInfo msg;
  msg.k = makeK(500.0, 500.0, 320.0, 240.0);
  msg.k[0] = std::numeric_limits<double>::quiet_NaN();
  msg.d = {};

  CameraModel model;
  EXPECT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::CANONICAL, model), StatusCode::INVALID_INPUT
  );
}

TEST(RosCompatErrors, RejectsNonFiniteD)
{
  RosCameraInfo msg;
  msg.k = makeK(500.0, 500.0, 320.0, 240.0);
  msg.d = {std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0, 0.0};

  CameraModel model;
  EXPECT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::OPENCV_CALIB3D_D5, model),
    StatusCode::INVALID_INPUT
  );
}

TEST(RosCompatErrors, RejectsProfileCountMismatch)
{
  RosCameraInfo msg;
  msg.k = makeK(500.0, 500.0, 320.0, 240.0);
  msg.d = {-0.2, 0.05, 0.001, -0.001};  // 4 coeffs, D5 profile expects 5

  CameraModel model;
  EXPECT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::OPENCV_CALIB3D_D5, model),
    StatusCode::INVALID_INPUT
  );
}

// ---------------------------------------------------------------------------
// binning / ROI adjustment. sensor_msgs/CameraInfo specifies K and P in
// FULL-RESOLUTION coordinates while the published image is the ROI crop
// decimated by the binning factors; the conversion used to ignore both, so a
// binning=2 camera silently produced a model whose intrinsics were off by 2x.
// ---------------------------------------------------------------------------

RosCameraInfo makeFullResPinholeMsg()
{
  RosCameraInfo msg;
  msg.k = makeK(1000.0, 1000.0, 640.0, 480.0);
  msg.distortion_model = "plumb_bob";
  msg.d = {0.0, 0.0, 0.0, 0.0, 0.0};
  msg.width = 1280U;
  msg.height = 960U;
  return msg;
}

TEST(RosCompatBinningRoi, BinningScalesIntrinsics)
{
  RosCameraInfo msg = makeFullResPinholeMsg();
  msg.binning_x = 2U;
  msg.binning_y = 2U;

  const CameraModel model = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_NEAR(model.intrinsics.fx, 500.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.fy, 500.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cx, 320.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cy, 240.0, 1e-3);
}

TEST(RosCompatBinningRoi, RoiShiftsPrincipalPoint)
{
  RosCameraInfo msg = makeFullResPinholeMsg();
  msg.roi.x_offset = 100U;
  msg.roi.y_offset = 50U;
  msg.roi.width = 640U;
  msg.roi.height = 480U;

  const CameraModel model = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_NEAR(model.intrinsics.fx, 1000.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.fy, 1000.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cx, 540.0, 1e-3);  // 640 - 100
  EXPECT_NEAR(model.intrinsics.cy, 430.0, 1e-3);  // 480 - 50
}

TEST(RosCompatBinningRoi, BinnedRoiCombination)
{
  RosCameraInfo msg = makeFullResPinholeMsg();
  msg.binning_x = 2U;
  msg.binning_y = 2U;
  msg.roi.x_offset = 100U;
  msg.roi.y_offset = 50U;
  msg.roi.width = 640U;
  msg.roi.height = 480U;

  // Per the spec the ROI is expressed in full-resolution coordinates and the
  // crop is decimated afterwards: shift first, then scale.
  const CameraModel model = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_NEAR(model.intrinsics.fx, 500.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cx, 270.0, 1e-3);  // (640 - 100) / 2
  EXPECT_NEAR(model.intrinsics.cy, 215.0, 1e-3);  // (480 - 50) / 2
}

TEST(RosCompatBinningRoi, ExportResetsBinningAndRoi)
{
  // In-place republish pattern: import a binned+cropped message, then export
  // the model back into the SAME message object. Import folds binning/ROI
  // into K, so export must reset the metadata — otherwise the next importer
  // applies the adjustment a second time and silently halves the focal
  // length.
  RosCameraInfo msg = makeFullResPinholeMsg();
  msg.binning_x = 2U;
  msg.binning_y = 2U;
  msg.roi.x_offset = 100U;
  msg.roi.y_offset = 50U;
  msg.roi.width = 640U;
  msg.roi.height = 480U;

  CameraModel model;
  ASSERT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::OPENCV_CALIB3D_D5, model), StatusCode::OK
  );
  EXPECT_NEAR(model.intrinsics.fx, 500.0, 1e-3);

  ASSERT_EQ(
    exportPinholeCameraInfo(model, PinholeCompatProfile::OPENCV_CALIB3D_D5, msg), StatusCode::OK
  );
  EXPECT_EQ(msg.binning_x, 0U);
  EXPECT_EQ(msg.binning_y, 0U);
  EXPECT_EQ(msg.roi.x_offset, 0U);
  EXPECT_EQ(msg.roi.y_offset, 0U);
  EXPECT_EQ(msg.roi.width, 0U);
  EXPECT_EQ(msg.roi.height, 0U);

  // The republished message must round-trip to the SAME model — not one
  // with the binning folded in twice.
  CameraModel again;
  ASSERT_EQ(
    importPinholeCameraInfo(msg, PinholeCompatProfile::OPENCV_CALIB3D_D5, again), StatusCode::OK
  );
  EXPECT_NEAR(again.intrinsics.fx, model.intrinsics.fx, 1e-6);
  EXPECT_NEAR(again.intrinsics.fy, model.intrinsics.fy, 1e-6);
  EXPECT_NEAR(again.intrinsics.cx, model.intrinsics.cx, 1e-6);
  EXPECT_NEAR(again.intrinsics.cy, model.intrinsics.cy, 1e-6);
}

TEST(RosCompatBinningRoi, DefaultBinningAndFullRoiAreNoOp)
{
  // binning 0/1 and a zero (= full) ROI must leave K untouched — the shape
  // every existing publisher uses.
  for (const unsigned int binning : {0U, 1U})
  {
    RosCameraInfo msg = makeFullResPinholeMsg();
    msg.binning_x = binning;
    msg.binning_y = binning;

    const CameraModel model = camxiom::makeCameraModel(msg);
    ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
    EXPECT_NEAR(model.intrinsics.fx, 1000.0, 1e-3);
    EXPECT_NEAR(model.intrinsics.cx, 640.0, 1e-3);
    EXPECT_NEAR(model.intrinsics.cy, 480.0, 1e-3);
  }
}

TEST(RosCompatBinningRoi, ProfileImportUsesAdjustedK)
{
  // The profile-based imports read K through the same adjustment (they used
  // to copy the raw message K directly).
  RosCameraInfo msg;
  msg.k = makeK(600.0, 600.0, 640.0, 480.0);
  msg.d = {0.01, 0.001, 0.0001, -0.0002};
  msg.width = 1280U;
  msg.height = 960U;
  msg.binning_x = 2U;
  msg.binning_y = 2U;

  CameraModel model;
  ASSERT_EQ(
    importFisheyeCameraInfo(msg, FisheyeCompatProfile::OPENCV_FISHEYE_D4, model), StatusCode::OK
  );
  EXPECT_NEAR(model.intrinsics.fx, 300.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cx, 320.0, 1e-3);
  EXPECT_NEAR(model.intrinsics.cy, 240.0, 1e-3);
}

#else  // sensor_msgs not available

TEST(RosCompat, SkippedWithoutSensorMsgs)
{
  GTEST_SKIP() << "sensor_msgs headers not available; ROS compat layer not built";
}

#endif  // __has_include(<sensor_msgs/msg/camera_info.hpp>)
