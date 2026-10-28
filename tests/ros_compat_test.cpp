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
// compat layer (makeCameraModel / exportCameraInfo). Until now
// nothing in the suite included <sensor_msgs/...>, so this bridge had zero
// regression coverage (LOADMAP T5). The POD path itself is covered by
// compat_test (T3); here we guard the sensor_msgs <-> CameraModel field wiring.
//
// Strategy: (1) makeCameraModel(msg) dispatches by distortion_model string;
// (2) a hand-authored CameraInfo message round-trips message -> CameraModel ->
// exportCameraInfo -> message with K/D preserved and the rectification /
// projection matrices filled, and a lens ROS cannot hold is refused without
// touching the message; (3) deterministic error paths.
//
// Only built when the ROS compat layer was compiled in (CMake gate
// CAMXIOM_WITH_ROS); the __has_include guard keeps the TU harmless otherwise.

#include <gtest/gtest.h>

#if __has_include(<sensor_msgs/msg/camera_info.hpp>)

#include "camxiom/camera_info_yaml.hpp"
#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "camxiom/ros.hpp"
#include "camxiom/types.hpp"

#include <sensor_msgs/msg/camera_info.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
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

// exportCameraInfo always writes an identity rectification and a 3x4
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

// Only the ROS names exportCameraInfo writes are read; any other name (here
// camxiom's former double_sphere packing, and an empty name) gives a model
// validation rejects.
TEST(RosCompatMakeCameraModel, OtherNamesAreRefused)
{
  for (const char *name : {"double_sphere", ""})
  {
    RosCameraInfo msg;
    msg.k = makeK(200.0, 200.0, 320.0, 240.0);
    msg.distortion_model = name;
    msg.d = {-0.2, 0.5};
    EXPECT_NE(validateCameraModel(makeCameraModel(msg)), StatusCode::OK) << name;
  }
}

// ===========================================================================
// message -> CameraModel -> exportCameraInfo -> message round-trips.
// ===========================================================================

TEST(RosCompatRoundTrip, PinholeRadtan5)
{
  RosCameraInfo msg;
  msg.k = makeK(525.0, 524.0, 319.5, 239.5);
  msg.distortion_model = "plumb_bob";
  msg.d = {-0.2, 0.05, 0.001, -0.001, 0.01};

  const CameraModel model = camxiom::makeCameraModel(msg);
  EXPECT_EQ(model.projection.type, ProjectionModelType::PINHOLE);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  RosCameraInfo out;
  out.width = 640U;  // the export must not touch the image geometry
  ASSERT_EQ(exportCameraInfo(model, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "plumb_bob");
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 525.0, 524.0, 319.5, 239.5);
  EXPECT_EQ(out.width, 640U);
}

TEST(RosCompatRoundTrip, EquidistantKb4)
{
  RosCameraInfo msg;
  msg.k = makeK(300.0, 300.0, 320.0, 240.0);
  msg.distortion_model = "equidistant";
  msg.d = {0.01, 0.001, 0.0001, -0.0002};

  const CameraModel model = camxiom::makeCameraModel(msg);
  EXPECT_EQ(model.projection.type, ProjectionModelType::EQUIDISTANT);
  EXPECT_EQ(model.distortion.type, DistortionModelType::KB4);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  RosCameraInfo out;
  ASSERT_EQ(exportCameraInfo(model, out), StatusCode::OK);
  EXPECT_EQ(out.distortion_model, "equidistant");
  expectKNear(out.k, msg.k);
  expectDNear(out.d, msg.d);
  expectIdentityR(out);
  expectProjectionMatrix(out, 300.0, 300.0, 320.0, 240.0);
}

// The sphere models read from camxiom's own names, but ROS has no distortion
// model for them: the export refuses and leaves the message as it was.
// The sphere models have no ROS distortion model: the export refuses and
// leaves the message as it was.
TEST(RosCompatExport, RefusesLensesRosCannotHold)
{
  const std::vector<std::pair<std::string, std::vector<double>>> cases{
    {"omnidirectional", {1.0}}, {"double_sphere", {-0.2, 0.5}}, {"eucm", {0.5, 1.0}}};
  for (const auto &[law, parameters] : cases)
  {
    CamxiomCameraInfo description;
    description.projection_model = law;
    description.projection_parameters = parameters;
    description.distortion_model = "none";
    description.intrinsic_parameters = {200.0, 200.0, 320.0, 240.0, 0.0};
    const CameraModel model = camxiom::makeCameraModel(description);
    ASSERT_EQ(validateCameraModel(model), StatusCode::OK) << law;

    RosCameraInfo out;
    out.distortion_model = "untouched";
    out.binning_x = 2U;
    EXPECT_EQ(exportCameraInfo(model, out), StatusCode::INVALID_MODEL) << law;
    EXPECT_EQ(out.distortion_model, "untouched") << law;
    EXPECT_TRUE(out.d.empty()) << law;
    EXPECT_EQ(out.binning_x, 2U) << law;
  }
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

  EXPECT_NE(validateCameraModel(camxiom::makeCameraModel(msg)), StatusCode::OK);
}

TEST(RosCompatErrors, RejectsNonFiniteD)
{
  RosCameraInfo msg;
  msg.k = makeK(500.0, 500.0, 320.0, 240.0);
  msg.distortion_model = "plumb_bob";
  msg.d = {std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0, 0.0};

  EXPECT_NE(validateCameraModel(camxiom::makeCameraModel(msg)), StatusCode::OK);
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

  const CameraModel model = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  EXPECT_NEAR(model.intrinsics.fx, 500.0, 1e-3);

  ASSERT_EQ(exportCameraInfo(model, msg), StatusCode::OK);
  EXPECT_EQ(msg.binning_x, 0U);
  EXPECT_EQ(msg.binning_y, 0U);
  EXPECT_EQ(msg.roi.x_offset, 0U);
  EXPECT_EQ(msg.roi.y_offset, 0U);
  EXPECT_EQ(msg.roi.width, 0U);
  EXPECT_EQ(msg.roi.height, 0U);

  // The republished message must round-trip to the SAME model — not one
  // with the binning folded in twice.
  const CameraModel again = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(again), StatusCode::OK);
  EXPECT_NEAR(again.intrinsics.fx, model.intrinsics.fx, 1e-6);
  EXPECT_NEAR(again.intrinsics.fy, model.intrinsics.fy, 1e-6);
  EXPECT_NEAR(again.intrinsics.cx, model.intrinsics.cx, 1e-6);
  EXPECT_NEAR(again.intrinsics.cy, model.intrinsics.cy, 1e-6);
}

// The same binned, cropped CameraInfo read as a message and as a ROS YAML
// file gives the same model.
TEST(RosCompatBinningRoi, YamlAndMessageGiveTheSameModel)
{
  RosCameraInfo msg = makeFullResPinholeMsg();
  msg.binning_x = 2U;
  msg.binning_y = 2U;
  msg.roi.x_offset = 100U;
  msg.roi.y_offset = 50U;
  msg.roi.width = 640U;
  msg.roi.height = 480U;
  const CameraModel from_message = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(from_message), StatusCode::OK);

  const std::string yaml =
    "image_width: 1280\n"
    "image_height: 960\n"
    "camera_matrix:\n"
    "  rows: 3\n"
    "  cols: 3\n"
    "  data: [1000.0, 0.0, 640.0, 0.0, 1000.0, 480.0, 0.0, 0.0, 1.0]\n"
    "distortion_model: plumb_bob\n"
    "distortion_coefficients:\n"
    "  rows: 1\n"
    "  cols: 5\n"
    "  data: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
    "binning_x: 2\n"
    "binning_y: 2\n"
    "roi:\n"
    "  x_offset: 100\n"
    "  y_offset: 50\n"
    "  height: 480\n"
    "  width: 640\n"
    "  do_rectify: false\n";
  CameraModel from_yaml;
  camxiom::ImageSize size;
  ASSERT_EQ(camxiom::parseCameraModelYaml(yaml, from_yaml, size), StatusCode::OK);
  EXPECT_EQ(from_yaml.intrinsics.fx, from_message.intrinsics.fx);
  EXPECT_EQ(from_yaml.intrinsics.fy, from_message.intrinsics.fy);
  EXPECT_EQ(from_yaml.intrinsics.cx, from_message.intrinsics.cx);
  EXPECT_EQ(from_yaml.intrinsics.cy, from_message.intrinsics.cy);
  EXPECT_EQ(from_yaml.distortion.type, from_message.distortion.type);
  EXPECT_EQ(size.width, 320);
  EXPECT_EQ(size.height, 240);
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

TEST(RosCompatBinningRoi, FisheyeImportUsesAdjustedK)
{
  // A fisheye message reads K through the same adjustment.
  RosCameraInfo msg;
  msg.k = makeK(600.0, 600.0, 640.0, 480.0);
  msg.distortion_model = "equidistant";
  msg.d = {0.01, 0.001, 0.0001, -0.0002};
  msg.width = 1280U;
  msg.height = 960U;
  msg.binning_x = 2U;
  msg.binning_y = 2U;

  const CameraModel model = camxiom::makeCameraModel(msg);
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
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
