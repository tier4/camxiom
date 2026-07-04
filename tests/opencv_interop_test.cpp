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

// Tests for the optional OpenCV interoperability layer (src/opencv/*).
//
// This layer is a cv::Mat drop-in facade over the OpenCV-free camxiom core:
// every projection/undistortion is delegated to the unified core geometry
// (rayToPixel(Batch) / pixelToRay(Batch) / buildRemapMap), OpenCV is only used
// for cv::Mat plumbing, cv::remap and cv::solvePnP. Until now nothing in the
// test suite included <camxiom/opencv.hpp>, so this ~1150-line layer had zero
// regression coverage (LOADMAP T1).
//
// Strategy: exercise every public entry point and, where a core analogue
// exists, assert the interop result matches the core geometry numerically
// (i.e. that the facade stays a thin wrapper). Only built when the OpenCV
// compat layer was compiled in (CMake gate CAMXIOM_WITH_OPENCV), so the
// symbols are guaranteed to exist.

#include <gtest/gtest.h>

#if __has_include(<opencv2/core.hpp>)

#include "camxiom/model.hpp"
#include "camxiom/opencv.hpp"
#include "camxiom/opencv/pnp.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/remap.hpp"
#include "camxiom/types.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

namespace cvx = camxiom::opencv;

using camxiom::CameraModel;
using camxiom::DistortionModelType;
using camxiom::DistortionSpace;
using camxiom::ProjectionModelType;
using camxiom::RemapResult;
using camxiom::StatusCode;

// ---------------------------------------------------------------------------
// Camera model factories (aligned with remap_test.cpp / projection_smoke_test)
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
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makePinholeRadTan()
{
  CameraModel m = makePinholeNoDistortion();
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.20f;   // k1
  m.distortion.coeffs[1] = 0.05f;    // k2
  m.distortion.coeffs[2] = 0.001f;   // p1
  m.distortion.coeffs[3] = -0.001f;  // p2
  m.distortion.coeffs[4] = 0.0f;     // k3
  m.distortion.count = 5U;
  return m;
}

CameraModel makeKB4(float theta_max = 1.2f)
{
  CameraModel m;
  m.intrinsics.fx = 300.0f;
  m.intrinsics.fy = 300.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = theta_max;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = 0.01f;
  m.distortion.coeffs[1] = 0.001f;
  m.distortion.coeffs[2] = 0.0f;
  m.distortion.coeffs[3] = 0.0f;
  m.distortion.count = 4U;
  return m;
}

cv::Matx33d rodrigues(const cv::Vec3d &rvec)
{
  cv::Matx33d R;
  cv::Rodrigues(rvec, R);
  return R;
}

// A smooth (wrap-free) gray ramp so cv::remap interpolation stays well-behaved.
cv::Mat makeGradientImage(int width, int height)
{
  cv::Mat img(height, width, CV_8UC1);
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const int val = 30 + (x * 120) / std::max(1, width) + (y * 60) / std::max(1, height);
      img.at<unsigned char>(y, x) = static_cast<unsigned char>(val);
    }
  }
  return img;
}

// ===========================================================================
// projectPoints — must equal manual (R,t) transform + core rayToPixel
// ===========================================================================

TEST(OpenCVInteropPoints, ProjectPointsMatchesCoreRayToPixel)
{
  const CameraModel model = makePinholeRadTan();

  const std::vector<cv::Point3f> object_points{
    {0.0f, 0.0f, 0.0f},
    {0.1f, 0.05f, 0.0f},
    {-0.08f, 0.12f, 0.02f},
    {0.15f, -0.1f, -0.03f},
    {-0.2f, -0.18f, 0.05f}};
  const cv::Vec3d rvec(0.03, -0.05, 0.02);
  const cv::Vec3d tvec(0.1, -0.05, 2.0);

  std::vector<cv::Point2f> image_points;
  const int valid = cvx::projectPoints(model, object_points, rvec, tvec, image_points);
  ASSERT_EQ(valid, static_cast<int>(object_points.size()));
  ASSERT_EQ(image_points.size(), object_points.size());

  const cv::Matx33d R = rodrigues(rvec);
  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    const cv::Point3f &p = object_points[i];
    const Eigen::Vector3f ray(
      static_cast<float>(R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + tvec[0]),
      static_cast<float>(R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + tvec[1]),
      static_cast<float>(R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + tvec[2])
    );
    const auto expected = camxiom::rayToPixel(model, ray);
    ASSERT_EQ(expected.status, StatusCode::OK);
    EXPECT_NEAR(image_points[i].x, expected.pixel.u, 1e-3);
    EXPECT_NEAR(image_points[i].y, expected.pixel.v, 1e-3);
  }
}

// ===========================================================================
// distortPoints / undistortPoints — mutual inverses (normalized <-> pixel)
// ===========================================================================

TEST(OpenCVInteropPoints, DistortUndistortRoundTripPinhole)
{
  const CameraModel model = makePinholeRadTan();

  const std::vector<cv::Point2f> normalized{
    {0.0f, 0.0f}, {0.10f, 0.05f}, {-0.12f, 0.08f}, {0.2f, -0.15f}};

  std::vector<cv::Point2f> pixels;
  ASSERT_EQ(cvx::distortPoints(model, normalized, pixels), static_cast<int>(normalized.size()));

  std::vector<cv::Point2f> recovered;
  ASSERT_EQ(
    cvx::undistortPoints(model, pixels, recovered, cv::Mat(), cv::Mat()),
    static_cast<int>(normalized.size())
  );

  ASSERT_EQ(recovered.size(), normalized.size());
  for (std::size_t i = 0; i < normalized.size(); ++i)
  {
    EXPECT_NEAR(recovered[i].x, normalized[i].x, 1e-3);
    EXPECT_NEAR(recovered[i].y, normalized[i].y, 1e-3);
  }
}

TEST(OpenCVInteropPoints, DistortUndistortRoundTripFisheye)
{
  const CameraModel model = makeKB4();

  const std::vector<cv::Point2f> normalized{
    {0.0f, 0.0f}, {0.15f, 0.10f}, {-0.2f, 0.05f}, {0.25f, -0.2f}};

  std::vector<cv::Point2f> pixels;
  ASSERT_EQ(cvx::distortPoints(model, normalized, pixels), static_cast<int>(normalized.size()));

  std::vector<cv::Point2f> recovered;
  ASSERT_EQ(
    cvx::undistortPoints(model, pixels, recovered, cv::Mat(), cv::Mat()),
    static_cast<int>(normalized.size())
  );

  for (std::size_t i = 0; i < normalized.size(); ++i)
  {
    EXPECT_NEAR(recovered[i].x, normalized[i].x, 2e-3);
    EXPECT_NEAR(recovered[i].y, normalized[i].y, 2e-3);
  }
}

// undistortPoints with a P matrix equal to K must map pixel -> pixel for a
// distortion-free pinhole (normalized coords are re-projected by P).
TEST(OpenCVInteropPoints, UndistortPointsWithProjectionMatrixRecoversPixels)
{
  const CameraModel model = makePinholeNoDistortion();
  const double fx = model.intrinsics.fx;
  const double fy = model.intrinsics.fy;
  const double cx = model.intrinsics.cx;
  const double cy = model.intrinsics.cy;

  const std::vector<cv::Point2f> pixels{
    {319.5f, 239.5f}, {400.0f, 300.0f}, {200.0f, 150.0f}, {450.0f, 120.0f}};

  const cv::Mat P = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

  std::vector<cv::Point2f> out;
  ASSERT_EQ(
    cvx::undistortPoints(model, pixels, out, cv::Mat(), P), static_cast<int>(pixels.size())
  );

  for (std::size_t i = 0; i < pixels.size(); ++i)
  {
    EXPECT_NEAR(out[i].x, pixels[i].x, 1e-2);
    EXPECT_NEAR(out[i].y, pixels[i].y, 1e-2);
  }
}

TEST(OpenCVInteropPoints, InvalidModelAndEmptyInput)
{
  const CameraModel bad{};  // default projection type is UNKNOWN -> invalid
  const std::vector<cv::Point3f> obj{{0.0f, 0.0f, 0.0f}};
  const std::vector<cv::Point2f> pix2{{0.0f, 0.0f}};
  std::vector<cv::Point2f> out;

  EXPECT_EQ(cvx::projectPoints(bad, obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 1), out), -1);
  EXPECT_EQ(cvx::undistortPoints(bad, pix2, out, cv::Mat(), cv::Mat()), -1);
  EXPECT_EQ(cvx::distortPoints(bad, pix2, out), -1);

  // Valid model, empty input -> 0 and cleared output.
  const CameraModel model = makePinholeNoDistortion();
  const std::vector<cv::Point3f> empty3;
  const std::vector<cv::Point2f> empty2;
  out.assign(3, cv::Point2f(1.0f, 1.0f));
  EXPECT_EQ(cvx::projectPoints(model, empty3, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 1), out), 0);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(cvx::distortPoints(model, empty2, out), 0);
  EXPECT_TRUE(out.empty());
}

// ===========================================================================
// buildRemapMapCV — must delegate to core buildRemapMap (same RemapResult)
// ===========================================================================

TEST(OpenCVInteropRemap, BuildRemapMapCVMatchesCore)
{
  const CameraModel src = makePinholeRadTan();
  const CameraModel dst = makePinholeNoDistortion();
  const int width = 64;
  const int height = 48;

  cv::Mat map1;
  cv::Mat map2;
  const RemapResult cv_res = cvx::buildRemapMapCV(src, dst, width, height, map1, map2);

  std::vector<float> mx(static_cast<std::size_t>(width * height));
  std::vector<float> my(static_cast<std::size_t>(width * height));
  const RemapResult core_res =
    camxiom::buildRemapMap(src, dst, width, height, mx.data(), my.data());

  EXPECT_EQ(static_cast<int>(cv_res.status), static_cast<int>(core_res.status));
  EXPECT_EQ(cv_res.valid_count, core_res.valid_count);
  EXPECT_EQ(cv_res.total_count, core_res.total_count);
  EXPECT_EQ(cv_res.status, StatusCode::OK);

  ASSERT_FALSE(map1.empty());
  ASSERT_FALSE(map2.empty());
  EXPECT_EQ(map1.rows, height);
  EXPECT_EQ(map1.cols, width);
}

TEST(OpenCVInteropRemap, BuildRemapMapCVRejectsInvalidSize)
{
  const CameraModel model = makePinholeNoDistortion();
  cv::Mat map1;
  cv::Mat map2;
  const RemapResult res = cvx::buildRemapMapCV(model, model, 0, 48, map1, map2);
  EXPECT_EQ(res.status, StatusCode::INVALID_INPUT);
  EXPECT_TRUE(map1.empty());
  EXPECT_TRUE(map2.empty());
}

// buildUndistortRemapMap semantics: for an already distortion-free pinhole the
// map is the identity. Verified on the raw float map (no fixed-point / interp
// noise) to precisely characterize the undistort path.
TEST(OpenCVInteropRemap, UndistortMapIsIdentityForDistortionFreePinhole)
{
  const CameraModel model = makePinholeNoDistortion();
  const int width = 32;
  const int height = 24;

  std::vector<float> mx(static_cast<std::size_t>(width * height));
  std::vector<float> my(static_cast<std::size_t>(width * height));
  const RemapResult res =
    camxiom::buildUndistortRemapMap(model, width, height, mx.data(), my.data());
  ASSERT_EQ(res.status, StatusCode::OK);

  for (int v = 0; v < height; ++v)
  {
    for (int u = 0; u < width; ++u)
    {
      const std::size_t idx = static_cast<std::size_t>(v * width + u);
      ASSERT_TRUE(std::isfinite(mx[idx]));
      ASSERT_TRUE(std::isfinite(my[idx]));
      EXPECT_NEAR(mx[idx], static_cast<float>(u), 1e-2);
      EXPECT_NEAR(my[idx], static_cast<float>(v), 1e-2);
    }
  }
}

// ===========================================================================
// RemapCache — build / apply / cache lifecycle
// ===========================================================================

TEST(OpenCVInteropRemapCache, BuildApplyClearLifecycle)
{
  const CameraModel src = makePinholeRadTan();
  const CameraModel dst = makePinholeNoDistortion();
  const int width = 64;
  const int height = 48;

  cvx::RemapCache cache;
  EXPECT_FALSE(cache.isValid());

  const RemapResult res = cache.build(src, dst, width, height);
  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_TRUE(cache.isValid());
  EXPECT_EQ(cache.width(), width);
  EXPECT_EQ(cache.height(), height);
  EXPECT_FALSE(cache.map1().empty());
  EXPECT_FALSE(cache.map2().empty());

  const cv::Mat image = makeGradientImage(width, height);
  cv::Mat out;
  ASSERT_TRUE(cache.apply(image, out));
  EXPECT_EQ(out.rows, height);
  EXPECT_EQ(out.cols, width);
  EXPECT_EQ(out.type(), image.type());

  cache.clear();
  EXPECT_FALSE(cache.isValid());
  EXPECT_TRUE(cache.map1().empty());
  cv::Mat out2;
  EXPECT_FALSE(cache.apply(image, out2));  // no map -> false
}

TEST(OpenCVInteropRemapCache, BuildUndistortAndRectify)
{
  const CameraModel src = makePinholeRadTan();
  // Rectify needs the image to actually cover the FOV around the principal
  // point, so use the model's native 640x480 resolution (matches remap_test).
  const int width = 640;
  const int height = 480;

  cvx::RemapCache undistort_cache;
  const RemapResult ures = undistort_cache.buildUndistort(src, width, height);
  EXPECT_EQ(ures.status, StatusCode::OK);
  EXPECT_TRUE(undistort_cache.isValid());

  cvx::RemapCache rectify_cache;
  camxiom::RectifiedOutputModelOptions options;
  options.output_size = camxiom::ImageSize{width, height};
  const camxiom::RectifyRemapResult rres =
    rectify_cache.buildRectify(src, camxiom::ImageSize{width, height}, options);
  EXPECT_EQ(rres.remap_result.status, StatusCode::OK);
  EXPECT_TRUE(rectify_cache.isValid());
  EXPECT_EQ(rres.output_model.projection.type, ProjectionModelType::PINHOLE);
}

// ===========================================================================
// Image-level convenience ops
// ===========================================================================

TEST(OpenCVInteropImage, UndistortRemapRectifyDistortSmoke)
{
  const CameraModel src = makePinholeRadTan();
  const CameraModel pinhole = makePinholeNoDistortion();
  // Native 640x480 resolution so rectify has FOV coverage around the principal
  // point (the model's cx/cy are set for 640x480).
  const cv::Mat image = makeGradientImage(640, 480);

  cv::Mat undistorted;
  ASSERT_TRUE(cvx::undistortImage(image, undistorted, src));
  EXPECT_EQ(undistorted.size(), image.size());
  EXPECT_EQ(undistorted.type(), image.type());

  cv::Mat remapped;
  ASSERT_TRUE(cvx::remapImage(image, remapped, src, pinhole));
  EXPECT_EQ(remapped.size(), image.size());

  cv::Mat rectified;
  CameraModel out_model{};
  ASSERT_TRUE(
    cvx::rectifyImage(image, rectified, src, camxiom::RectifiedOutputModelOptions{}, &out_model)
  );
  EXPECT_EQ(rectified.size(), image.size());
  EXPECT_EQ(out_model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_GT(out_model.intrinsics.fx, 0.0f);

  cv::Mat distorted;
  ASSERT_TRUE(cvx::distortImage(image, distorted, pinhole, src));
  EXPECT_EQ(distorted.size(), image.size());
}

TEST(OpenCVInteropImage, EmptyInputReturnsFalse)
{
  const CameraModel src = makePinholeRadTan();
  const cv::Mat empty;
  cv::Mat dst;
  EXPECT_FALSE(cvx::undistortImage(empty, dst, src));
  EXPECT_FALSE(cvx::remapImage(empty, dst, src, makePinholeNoDistortion()));
  EXPECT_FALSE(cvx::rectifyImage(empty, dst, src));
  EXPECT_FALSE(cvx::distortImage(empty, dst, makePinholeNoDistortion(), src));
}

// ===========================================================================
// K/D cv::Mat drop-in APIs (pinhole / fisheye / omnidirectional namespaces)
// ===========================================================================

TEST(OpenCVInteropDropIn, PinholeProjectPointsMatchesManualPinhole)
{
  const double fx = 520.0;
  const double fy = 520.0;
  const double cx = 320.0;
  const double cy = 240.0;
  const cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  const cv::Mat D;  // no distortion

  const std::vector<cv::Point3f> object_points{
    {0.0f, 0.0f, 0.0f}, {0.1f, 0.05f, 0.0f}, {-0.1f, 0.08f, 0.02f}, {0.12f, -0.09f, 0.0f}};
  const cv::Vec3d rvec(0.0, 0.0, 0.0);  // identity rotation
  const cv::Vec3d tvec(0.05, -0.02, 1.5);

  std::vector<cv::Point2f> image_points;
  ASSERT_EQ(
    cvx::pinhole::projectPoints(object_points, rvec, tvec, K, D, image_points),
    static_cast<int>(object_points.size())
  );

  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    const cv::Point3f &p = object_points[i];
    const double xc = p.x + tvec[0];
    const double yc = p.y + tvec[1];
    const double zc = p.z + tvec[2];
    const double u = fx * (xc / zc) + cx;
    const double v = fy * (yc / zc) + cy;
    EXPECT_NEAR(image_points[i].x, u, 1e-2);
    EXPECT_NEAR(image_points[i].y, v, 1e-2);
  }
}

TEST(OpenCVInteropDropIn, PinholeUndistortDistortRoundTrip)
{
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0);
  const cv::Mat D = (cv::Mat_<double>(1, 5) << -0.18, 0.04, 0.001, -0.001, 0.0);

  const std::vector<cv::Point2f> normalized{{0.0f, 0.0f}, {0.08f, 0.05f}, {-0.1f, 0.07f}};

  std::vector<cv::Point2f> pixels;
  ASSERT_EQ(
    cvx::pinhole::distortPoints(normalized, pixels, K, D), static_cast<int>(normalized.size())
  );

  std::vector<cv::Point2f> recovered;
  ASSERT_EQ(
    cvx::pinhole::undistortPoints(pixels, recovered, K, D, cv::Mat(), cv::Mat()),
    static_cast<int>(normalized.size())
  );

  for (std::size_t i = 0; i < normalized.size(); ++i)
  {
    EXPECT_NEAR(recovered[i].x, normalized[i].x, 2e-3);
    EXPECT_NEAR(recovered[i].y, normalized[i].y, 2e-3);
  }
}

TEST(OpenCVInteropDropIn, FisheyeAndOmniProjectPointsSmoke)
{
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);
  const cv::Mat D4 = (cv::Mat_<double>(1, 4) << 0.01, 0.001, 0.0, 0.0);
  const std::vector<cv::Point3f> object_points{
    {0.0f, 0.0f, 0.0f}, {0.1f, 0.05f, 0.0f}, {-0.1f, 0.08f, 0.0f}, {0.05f, -0.06f, 0.0f}};
  const cv::Vec3d rvec(0.0, 0.0, 0.0);
  const cv::Vec3d tvec(0.02, -0.01, 1.2);

  std::vector<cv::Point2f> fisheye_points;
  ASSERT_EQ(
    cvx::fisheye::projectPoints(object_points, rvec, tvec, K, D4, fisheye_points),
    static_cast<int>(object_points.size())
  );
  for (const auto &pt : fisheye_points)
  {
    EXPECT_TRUE(std::isfinite(pt.x));
    EXPECT_TRUE(std::isfinite(pt.y));
  }

  std::vector<cv::Point2f> omni_points;
  ASSERT_EQ(
    cvx::omnidirectional::projectPoints(object_points, rvec, tvec, K, D4, 1.0, omni_points),
    static_cast<int>(object_points.size())
  );
  for (const auto &pt : omni_points)
  {
    EXPECT_TRUE(std::isfinite(pt.x));
    EXPECT_TRUE(std::isfinite(pt.y));
  }
}

TEST(OpenCVInteropDropIn, InvalidCameraMatrixReturnsError)
{
  const cv::Mat bad_K = (cv::Mat_<double>(2, 2) << 1.0, 0.0, 0.0, 1.0);  // wrong shape
  const cv::Mat D;
  const std::vector<cv::Point3f> obj{{0.0f, 0.0f, 0.0f}};
  std::vector<cv::Point2f> out;
  EXPECT_EQ(
    cvx::pinhole::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 1), bad_K, D, out), -1
  );
}

// ===========================================================================
// solvePnP / reprojectRmse — unified PnP over the interop layer
// ===========================================================================

TEST(OpenCVInteropPnp, SolvePnpRecoversPoseAndLowRmse)
{
  const CameraModel model = makePinholeRadTan();

  // Planar target (z = 0) grid in the object frame.
  std::vector<cv::Point3f> object_points;
  for (int r = 0; r < 5; ++r)
  {
    for (int c = 0; c < 5; ++c)
    {
      object_points.emplace_back(
        static_cast<float>(c) * 0.05f - 0.1f, static_cast<float>(r) * 0.05f - 0.1f, 0.0f
      );
    }
  }

  const cv::Vec3d rvec_gt(0.04, -0.08, 0.02);
  const cv::Vec3d tvec_gt(0.03, 0.01, 1.4);

  std::vector<cv::Point2f> image_points;
  ASSERT_EQ(
    cvx::projectPoints(model, object_points, rvec_gt, tvec_gt, image_points),
    static_cast<int>(object_points.size())
  );

  // Reprojection RMSE at the ground-truth pose is ~0.
  EXPECT_LT(cvx::reprojectRmse(model, object_points, image_points, rvec_gt, tvec_gt), 1e-2);

  cv::Vec3d rvec_est;
  cv::Vec3d tvec_est;
  ASSERT_TRUE(cvx::solvePnP(model, object_points, image_points, rvec_est, tvec_est));

  // Recovered pose reprojects with low error.
  EXPECT_LT(cvx::reprojectRmse(model, object_points, image_points, rvec_est, tvec_est), 0.5);

  EXPECT_NEAR(tvec_est[0], tvec_gt[0], 5e-2);
  EXPECT_NEAR(tvec_est[1], tvec_gt[1], 5e-2);
  EXPECT_NEAR(tvec_est[2], tvec_gt[2], 5e-2);
}

TEST(OpenCVInteropPnp, RejectsTooFewPoints)
{
  const CameraModel model = makePinholeNoDistortion();
  const std::vector<cv::Point3f> obj{{0, 0, 0}, {0.1f, 0, 0}, {0, 0.1f, 0}};  // only 3
  const std::vector<cv::Point2f> img{{320, 240}, {360, 240}, {320, 280}};
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  EXPECT_FALSE(cvx::solvePnP(model, obj, img, rvec, tvec));
  EXPECT_EQ(
    cvx::reprojectRmse(model, obj, {}, rvec, tvec), std::numeric_limits<double>::infinity()
  );
}

// ===========================================================================
// initCameraMatrix2D — recovers a sane pinhole K from synthetic planar views
// ===========================================================================

TEST(OpenCVInteropInit, InitCameraMatrix2DRecoversPositiveFocal)
{
  const double fx = 600.0;
  const double fy = 600.0;
  const double cx = 320.0;
  const double cy = 240.0;
  const cv::Size image_size(640, 480);

  // Planar board grid (z = 0).
  std::vector<cv::Point3f> board;
  for (int r = 0; r < 6; ++r)
  {
    for (int c = 0; c < 8; ++c)
    {
      board.emplace_back(static_cast<float>(c) * 0.03f, static_cast<float>(r) * 0.03f, 0.0f);
    }
  }

  const std::vector<cv::Vec3d> rvecs{{0.0, 0.0, 0.0}, {0.2, -0.1, 0.05}, {-0.15, 0.25, -0.05}};
  const std::vector<cv::Vec3d> tvecs{{-0.1, -0.08, 0.8}, {-0.05, -0.1, 0.9}, {-0.12, -0.06, 0.85}};

  std::vector<std::vector<cv::Point3f>> object_points_per_view;
  std::vector<std::vector<cv::Point2f>> image_points_per_view;
  for (std::size_t view = 0; view < rvecs.size(); ++view)
  {
    const cv::Matx33d R = rodrigues(rvecs[view]);
    const cv::Vec3d &t = tvecs[view];
    std::vector<cv::Point2f> image_points;
    image_points.reserve(board.size());
    for (const cv::Point3f &p : board)
    {
      const double xc = R(0, 0) * p.x + R(0, 1) * p.y + R(0, 2) * p.z + t[0];
      const double yc = R(1, 0) * p.x + R(1, 1) * p.y + R(1, 2) * p.z + t[1];
      const double zc = R(2, 0) * p.x + R(2, 1) * p.y + R(2, 2) * p.z + t[2];
      image_points.emplace_back(
        static_cast<float>(fx * xc / zc + cx), static_cast<float>(fy * yc / zc + cy)
      );
    }
    object_points_per_view.push_back(board);
    image_points_per_view.push_back(std::move(image_points));
  }

  CameraModel model_out{};
  ASSERT_TRUE(
    cvx::initCameraMatrix2D(object_points_per_view, image_points_per_view, image_size, model_out)
  );
  EXPECT_EQ(model_out.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_GT(model_out.intrinsics.fx, 0.0f);
  EXPECT_GT(model_out.intrinsics.fy, 0.0f);
  EXPECT_TRUE(std::isfinite(model_out.intrinsics.fx));
  // cv::initCameraMatrix2D fixes the principal point at the image center.
  EXPECT_NEAR(model_out.intrinsics.cx, static_cast<float>(image_size.width) * 0.5f, 2.0f);
  EXPECT_NEAR(model_out.intrinsics.cy, static_cast<float>(image_size.height) * 0.5f, 2.0f);
}

TEST(OpenCVInteropInit, InitCameraMatrix2DRejectsMalformedInput)
{
  CameraModel model_out{};
  // Mismatched view counts.
  const std::vector<std::vector<cv::Point3f>> obj{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}};
  const std::vector<std::vector<cv::Point2f>> img;  // empty -> size mismatch
  EXPECT_FALSE(cvx::initCameraMatrix2D(obj, img, cv::Size(640, 480), model_out));
}

// ---------------------------------------------------------------------------
// Robustness: in-place image calls and non-continuous point Mats
// ---------------------------------------------------------------------------

TEST(OpenCVInteropImage, UndistortImageInPlace)
{
  // cv::remap cannot run in place: passing the same Mat as src and dst used
  // to make it read pixels it had already overwritten, silently corrupting
  // the output. The wrappers must detour through a temporary.
  const CameraModel model = makePinholeRadTan();

  cv::Mat src(96, 128, CV_8UC1);
  for (int r = 0; r < src.rows; ++r)
  {
    for (int c = 0; c < src.cols; ++c)
    {
      src.at<std::uint8_t>(r, c) = static_cast<std::uint8_t>((r * 31 + c * 7) & 0xFF);
    }
  }

  cv::Mat expected;
  ASSERT_TRUE(cvx::undistortImage(src, expected, model));

  cv::Mat in_place = src.clone();
  ASSERT_TRUE(cvx::undistortImage(in_place, in_place, model));

  ASSERT_EQ(in_place.size(), expected.size());
  ASSERT_EQ(in_place.type(), expected.type());
  EXPECT_EQ(cv::countNonZero(in_place != expected), 0);
}

TEST(OpenCVInteropPoints, NonContinuousPointMatMatchesContinuous)
{
  // reshape(1, N) throws cv::Exception on non-continuous views; the extract
  // helpers must clone first so an ROI / submat input just works and gives
  // the same result as a continuous buffer.
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
  const cv::Mat D;  // no distortion
  const cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  tvec.at<double>(2) = 1.0;

  // Two-column 3-channel point Mat; col(0) is a non-continuous view.
  cv::Mat big(4, 2, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
  for (int i = 0; i < big.rows; ++i)
  {
    big.at<cv::Vec3f>(i, 0) =
      cv::Vec3f(0.1f * static_cast<float>(i), -0.05f * static_cast<float>(i), 1.0f);
  }
  const cv::Mat roi = big.col(0);
  ASSERT_FALSE(roi.isContinuous());

  std::vector<cv::Point2f> from_roi;
  ASSERT_EQ(cvx::pinhole::projectPoints(roi, rvec, tvec, K, D, from_roi), 4);

  std::vector<cv::Point2f> from_continuous;
  ASSERT_EQ(cvx::pinhole::projectPoints(roi.clone(), rvec, tvec, K, D, from_continuous), 4);

  for (int i = 0; i < 4; ++i)
  {
    EXPECT_FLOAT_EQ(
      from_roi[static_cast<std::size_t>(i)].x, from_continuous[static_cast<std::size_t>(i)].x
    ) << i;
    EXPECT_FLOAT_EQ(
      from_roi[static_cast<std::size_t>(i)].y, from_continuous[static_cast<std::size_t>(i)].y
    ) << i;
  }
}

}  // namespace

TEST(OpenCVInteropPoints, RejectsMultiChannelMatrices)
{
  // Mat::at<double> does not type-check in Release builds (CV_DbgAssert),
  // so a CV_64FC3 matrix used to be read with the wrong stride and silently
  // produced garbage intrinsics / R / P. All three matrix parsers must
  // reject non-single-channel input instead.
  const cv::Mat K1 =
    (cv::Mat_<double>(3, 3) << 500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0);
  const cv::Mat D = (cv::Mat_<double>(1, 5) << -0.18, 0.04, 0.001, -0.001, 0.0);
  cv::Mat K3;
  cv::merge(std::vector<cv::Mat>{K1, K1, K1}, K3);  // CV_64FC3

  const std::vector<cv::Point2f> pixels{{320.0f, 240.0f}};
  std::vector<cv::Point2f> out;
  EXPECT_EQ(cvx::pinhole::undistortPoints(pixels, out, K3, D, cv::Mat(), cv::Mat()), -1);

  const CameraModel model = makePinholeNoDistortion();
  cv::Mat R3;
  const cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
  cv::merge(std::vector<cv::Mat>{R1, R1, R1}, R3);
  EXPECT_EQ(cvx::undistortPoints(model, pixels, out, R3, cv::Mat()), -1);

  cv::Mat P3;
  cv::merge(std::vector<cv::Mat>{K1, K1, K1}, P3);
  EXPECT_EQ(cvx::undistortPoints(model, pixels, out, cv::Mat(), P3), -1);
}
#else  // OpenCV headers not reachable

TEST(OpenCVInterop, LayerNotCompiled)
{
  GTEST_SKIP() << "camxiom OpenCV interop layer not built (OpenCV not found).";
}

#endif  // __has_include(<opencv2/core.hpp>)
