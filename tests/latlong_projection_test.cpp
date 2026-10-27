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

// The lat-long projections (EQUIRECTANGULAR, CYLINDRICAL) as ordinary camera
// models: the longitude/latitude contract of types.hpp (orientation, seam,
// poles), forward/inverse round trips in both precisions, and parity of every
// path that consumes a CameraModel (batch, ValidatedCameraModel, Jacobian,
// inverse LUT, autodiff template).
//
// CORE test: no Ceres / ROS / OpenCV, built in every configuration.

#include "camxiom/batch.hpp"
#include "camxiom/batch64.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/jacobian.hpp"
#include "camxiom/jacobian64.hpp"
#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/lut.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/projection_template.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "camxiom/validated_model.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

using namespace camxiom;

namespace
{

constexpr double kPi = constants::kPi;

// A 2048 x 1024 full-sphere layout: fx = W / 2pi, fy = H / pi, centred.
template <typename T>
CameraModelT<T> latLongModel(const ProjectionModelType type)
{
  CameraModelT<T> m;
  m.projection.type = type;
  m.projection.theta_max = static_cast<T>(kPi);
  m.intrinsics.fx = static_cast<T>(2048.0 / (2.0 * kPi));
  m.intrinsics.fy =
    static_cast<T>(type == ProjectionModelType::EQUIRECTANGULAR ? 1024.0 / kPi : 300.0);
  m.intrinsics.cx = static_cast<T>(1023.5);
  m.intrinsics.cy = static_cast<T>(511.5);
  return m;
}

const ProjectionModelType kTypes[] = {
  ProjectionModelType::EQUIRECTANGULAR, ProjectionModelType::CYLINDRICAL};

// Directions over the whole sphere, kept 1e-3 rad away from the poles and the
// seam so that both precisions see the same side of every boundary.
std::vector<Eigen::Vector3d> sphereRays(const int count)
{
  std::mt19937 rng(7U);
  std::uniform_real_distribution<double> lon(-kPi + 1e-3, kPi - 1e-3);
  std::uniform_real_distribution<double> lat(-kPi / 2 + 1e-3, kPi / 2 - 1e-3);
  std::vector<Eigen::Vector3d> rays;
  for (int i = 0; i < count; ++i)
  {
    const double l = lon(rng);
    const double p = lat(rng);
    rays.emplace_back(std::cos(p) * std::sin(l), std::sin(p), std::cos(p) * std::cos(l));
  }
  return rays;
}

}  // namespace

TEST(LatLongProjection, NamesRoundTrip)
{
  EXPECT_STREQ(toString(ProjectionModelType::EQUIRECTANGULAR), "equirectangular");
  EXPECT_STREQ(toString(ProjectionModelType::CYLINDRICAL), "cylindrical");
  for (const auto type : kTypes)
  {
    EXPECT_EQ(parseProjectionModelType(toString(type)), type);
    EXPECT_TRUE(isLatLongProjection(type));
    EXPECT_FALSE(isFisheyeProjection(type));
  }
  EXPECT_FALSE(isLatLongProjection(ProjectionModelType::PINHOLE));
}

TEST(LatLongProjection, TakesNoLensCorrection)
{
  for (const auto type : kTypes)
  {
    CameraModel m = latLongModel<float>(type);
    EXPECT_EQ(validateCameraModel(m), StatusCode::OK);
    m.distortion.type = DistortionModelType::RADTAN5;
    m.distortion.count = 5;
    EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
    m.distortion.type = DistortionModelType::KB4;
    m.distortion.count = 4;
    EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
  }
}

TEST(LatLongProjection, IgnoresThetaMax)
{
  for (const auto type : kTypes)
  {
    CameraModel m = latLongModel<float>(type);
    m.projection.theta_max = 0.1f;
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
    // 90 degrees to the right is far outside a 0.1 rad cap.
    const PixelResult px = rayToPixel(m, Eigen::Vector3f(1.0f, 0.0f, 0.0f));
    ASSERT_EQ(px.status, StatusCode::OK);
    EXPECT_NEAR(px.pixel.u, 1023.5f + 512.0f, 1e-3f);
  }
}

TEST(LatLongProjection, Orientation)
{
  for (const auto type : kTypes)
  {
    const CameraModel64 m = latLongModel<double>(type);
    const double cx = m.intrinsics.cx;
    const double cy = m.intrinsics.cy;

    const PixelResult64 ahead = rayToPixel64(m, Eigen::Vector3d(0, 0, 1));
    ASSERT_EQ(ahead.status, StatusCode::OK);
    EXPECT_NEAR(ahead.pixel.u, cx, 1e-9);
    EXPECT_NEAR(ahead.pixel.v, cy, 1e-9);

    // +X is right, +Y is down.
    const PixelResult64 right = rayToPixel64(m, Eigen::Vector3d(1, 0, 0));
    ASSERT_EQ(right.status, StatusCode::OK);
    EXPECT_NEAR(right.pixel.u, cx + 512.0, 1e-9);
    const PixelResult64 up = rayToPixel64(m, Eigen::Vector3d(0, -1, 1));
    ASSERT_EQ(up.status, StatusCode::OK);
    EXPECT_LT(up.pixel.v, cy);

    // A row above the centre looks up.
    const RayResult64 ray = pixelToRay64(m, Pixel2d{cx, cy - 100.0});
    ASSERT_EQ(ray.status, StatusCode::OK);
    EXPECT_LT(ray.ray.direction.y(), 0.0);
    EXPECT_NEAR(ray.ray.direction.x(), 0.0, 1e-12);
    EXPECT_GT(ray.ray.direction.z(), 0.0);
  }
}

TEST(LatLongProjection, EquirectangularRowsAreLatitudes)
{
  const CameraModel64 m = latLongModel<double>(ProjectionModelType::EQUIRECTANGULAR);
  // 30 degrees up, 45 degrees left.
  const double lat = -kPi / 6.0;
  const double lon = -kPi / 4.0;
  const Eigen::Vector3d ray(
    std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon)
  );
  const PixelResult64 px = rayToPixel64(m, ray);
  ASSERT_EQ(px.status, StatusCode::OK);
  EXPECT_NEAR(px.pixel.u, 1023.5 + lon * 2048.0 / (2.0 * kPi), 1e-9);
  EXPECT_NEAR(px.pixel.v, 511.5 + lat * 1024.0 / kPi, 1e-9);
}

TEST(LatLongProjection, CylindricalRowsAreTangents)
{
  const CameraModel64 m = latLongModel<double>(ProjectionModelType::CYLINDRICAL);
  const Eigen::Vector3d ray(0.3, -0.4, 0.8);
  const PixelResult64 px = rayToPixel64(m, ray);
  ASSERT_EQ(px.status, StatusCode::OK);
  const double h = std::hypot(0.3, 0.8);
  EXPECT_NEAR(px.pixel.u, 1023.5 + std::atan2(0.3, 0.8) * 2048.0 / (2.0 * kPi), 1e-9);
  EXPECT_NEAR(px.pixel.v, 511.5 + 300.0 * (-0.4 / h), 1e-9);
}

TEST(LatLongProjection, SeamBelongsToTheLeftEdge)
{
  for (const auto type : kTypes)
  {
    const CameraModel64 m = latLongModel<double>(type);
    const double left = m.intrinsics.cx - kPi * m.intrinsics.fx;  // u = -0.5
    const PixelResult64 behind = rayToPixel64(m, Eigen::Vector3d(0.0, 0.0, -1.0));
    ASSERT_EQ(behind.status, StatusCode::OK);
    EXPECT_NEAR(behind.pixel.u, left, 1e-9);
    // Just right of the seam (seen from behind: +X side) lands on the right
    // edge, just left of it on the left edge.
    const PixelResult64 near_right = rayToPixel64(m, Eigen::Vector3d(1e-3, 0.0, -1.0));
    const PixelResult64 near_left = rayToPixel64(m, Eigen::Vector3d(-1e-3, 0.0, -1.0));
    ASSERT_EQ(near_right.status, StatusCode::OK);
    ASSERT_EQ(near_left.status, StatusCode::OK);
    EXPECT_NEAR(near_right.pixel.u, left + 2048.0 - 1e-3 * m.intrinsics.fx, 1e-6);
    EXPECT_NEAR(near_left.pixel.u, left + 1e-3 * m.intrinsics.fx, 1e-6);

    // The inverse does not wrap: longitude pi itself and beyond is outside.
    EXPECT_EQ(
      pixelToRay64(m, Pixel2d{left + 2048.0, m.intrinsics.cy}).status, StatusCode::OUT_OF_FOV
    );
    EXPECT_EQ(pixelToRay64(m, Pixel2d{left - 1.0, m.intrinsics.cy}).status, StatusCode::OUT_OF_FOV);
    EXPECT_EQ(pixelToRay64(m, Pixel2d{left, m.intrinsics.cy}).status, StatusCode::OK);

    const CameraModel mf = latLongModel<float>(type);
    EXPECT_EQ(rayToPixel(mf, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).status, StatusCode::OK);
    EXPECT_EQ(rayToPixel(mf, Eigen::Vector3f(-0.0f, 0.0f, -1.0f)).status, StatusCode::OK);
    EXPECT_NEAR(rayToPixel(mf, Eigen::Vector3f(0.0f, 0.0f, -1.0f)).pixel.u, -0.5f, 1e-3f);
    EXPECT_NEAR(rayToPixel(mf, Eigen::Vector3f(-0.0f, 0.0f, -1.0f)).pixel.u, -0.5f, 1e-3f);
  }
}

TEST(LatLongProjection, PolesHaveNoLongitude)
{
  for (const auto type : kTypes)
  {
    const CameraModel m = latLongModel<float>(type);
    const CameraModel64 m64 = latLongModel<double>(type);
    for (const double y : {-1.0, 1.0})
    {
      EXPECT_EQ(
        rayToPixel(m, Eigen::Vector3f(0.0f, static_cast<float>(y), 0.0f)).status,
        StatusCode::DOMAIN_ERROR
      );
      EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(0.0, y, 0.0)).status, StatusCode::DOMAIN_ERROR);
      EXPECT_EQ(
        rayToPixelWithJacobian64(m64, Eigen::Vector3d(0.0, y, 0.0)).status, StatusCode::DOMAIN_ERROR
      );
    }
    // A short ray near (but not at) the axis is a direction, not a pole.
    EXPECT_EQ(rayToPixel64(m64, Eigen::Vector3d(1e-9, -1e-7, 1e-9)).status, StatusCode::OK);
  }

  // The pole rows (latitude +-pi/2) have no longitude either: the inverse
  // refuses them like the forward map refuses the ray; past them is outside.
  // (fy = 1, cy = 0 puts the pole exactly on v = -+pi/2.)
  CameraModel64 m = latLongModel<double>(ProjectionModelType::EQUIRECTANGULAR);
  m.intrinsics.fy = 1.0;
  m.intrinsics.cy = 0.0;
  EXPECT_EQ(pixelToRay64(m, Pixel2d{100.0, -constants::kHalfPi}).status, StatusCode::DOMAIN_ERROR);
  EXPECT_EQ(pixelToRay64(m, Pixel2d{100.0, constants::kHalfPi}).status, StatusCode::DOMAIN_ERROR);
  EXPECT_EQ(
    pixelToRay64(m, Pixel2d{100.0, -constants::kHalfPi - 1e-4}).status, StatusCode::OUT_OF_FOV
  );
  EXPECT_EQ(
    pixelToRay64(m, Pixel2d{100.0, constants::kHalfPi + 1e-4}).status, StatusCode::OUT_OF_FOV
  );
  // The float half-pi lies just past the true pole; it is the pole too.
  const CameraModel mf = fromCameraModel64(m);
  EXPECT_EQ(pixelToRay(mf, Pixel2{100.0f, constants::kHalfPiF}).status, StatusCode::DOMAIN_ERROR);
  EXPECT_EQ(pixelToRay(mf, Pixel2{100.0f, -constants::kHalfPiF}).status, StatusCode::DOMAIN_ERROR);
  EXPECT_EQ(
    pixelToRay(mf, Pixel2{100.0f, constants::kHalfPiF + 1e-3f}).status, StatusCode::OUT_OF_FOV
  );

  // The cylinder has no latitude limit short of the poles.
  const CameraModel64 cyl = latLongModel<double>(ProjectionModelType::CYLINDRICAL);
  const RayResult64 steep = pixelToRay64(cyl, Pixel2d{1023.5, -1e6});
  ASSERT_EQ(steep.status, StatusCode::OK);
  EXPECT_LT(steep.ray.direction.y(), -0.9999);
}

namespace
{

// A unit-intrinsics model (pixel = (longitude, y)), so a test can step the
// longitude or latitude by single representable values.
template <typename T>
CameraModelT<T> unitLatLongModel(const ProjectionModelType type)
{
  CameraModelT<T> m = latLongModel<T>(type);
  m.intrinsics.fx = T(1);
  m.intrinsics.fy = T(1);
  m.intrinsics.cx = T(0);
  m.intrinsics.cy = T(0);
  return m;
}

template <typename T>
PixelResultT<T> project(const CameraModelT<T> &m, const Eigen::Matrix<T, 3, 1> &ray)
{
  if constexpr (std::is_same_v<T, float>)
  {
    return rayToPixel(m, ray);
  }
  else
  {
    return rayToPixel64(m, ray);
  }
}

template <typename T>
RayResultT<T> unproject(const CameraModelT<T> &m, const Pixel2T<T> &px)
{
  if constexpr (std::is_same_v<T, float>)
  {
    return pixelToRay(m, px);
  }
  else
  {
    return pixelToRay64(m, px);
  }
}

// Every pixel at and next to the seam, in one precision: the left edge
// (longitude -pi) is the seam ray and projects back onto the left edge;
// values inside stay on their own side; values past either edge are outside.
template <typename T>
void checkSeam(const ProjectionModelType type)
{
  const CameraModelT<T> m = unitLatLongModel<T>(type);
  const T pi = static_cast<T>(kPi);
  const T inf = std::numeric_limits<T>::infinity();
  const auto round_trip = [&](const T lambda) {
    const RayResultT<T> ray = unproject(m, Pixel2T<T>{lambda, T(0.25)});
    EXPECT_EQ(ray.status, StatusCode::OK) << toString(type) << " lambda " << lambda;
    const PixelResultT<T> px = project(m, ray.ray.direction);
    EXPECT_EQ(px.status, StatusCode::OK) << toString(type) << " lambda " << lambda;
    return px.pixel.u;
  };

  // The left edge and the value just inside it stay on the left edge.
  EXPECT_EQ(round_trip(-pi), -pi) << toString(type);
  const T just_inside_left = std::nextafter(-pi, T(0));
  EXPECT_NEAR(round_trip(just_inside_left), just_inside_left, 4 * (just_inside_left + pi));
  // Rounding slack below the left edge is the seam; well past it is outside.
  EXPECT_EQ(round_trip(std::nextafter(-pi, -inf)), -pi) << toString(type);
  EXPECT_EQ(unproject(m, Pixel2T<T>{-pi - T(1e-3), T(0)}).status, StatusCode::OUT_OF_FOV);
  // The right edge is open: just inside stays right, the edge itself is out.
  const T just_inside_right = std::nextafter(pi, T(0));
  EXPECT_GT(round_trip(just_inside_right), T(3)) << toString(type);
  EXPECT_EQ(unproject(m, Pixel2T<T>{pi, T(0)}).status, StatusCode::OUT_OF_FOV);
}

// Latitudes stepping onto the pole: an OK unprojection always projects again
// and comes back to the same row, a pole is DOMAIN_ERROR both ways.
template <typename T>
void checkPoleApproach()
{
  const CameraModelT<T> m = unitLatLongModel<T>(ProjectionModelType::EQUIRECTANGULAR);
  const T half_pi = static_cast<T>(kPi / 2);
  for (const T sign : {T(-1), T(1)})
  {
    T phi = sign * half_pi;
    for (int step = 0; step < 64; ++step)
    {
      const RayResultT<T> ray = unproject(m, Pixel2T<T>{T(0.5), phi});
      const PixelResultT<T> px = project(m, ray.ray.direction);
      if (ray.status == StatusCode::OK)
      {
        ASSERT_EQ(px.status, StatusCode::OK) << "phi " << phi;
        EXPECT_NEAR(px.pixel.v, phi, T(1e-3)) << "phi " << phi;
      }
      else
      {
        EXPECT_EQ(ray.status, StatusCode::DOMAIN_ERROR) << "phi " << phi;
      }
      phi = std::nextafter(phi, T(0));
    }
    EXPECT_EQ(unproject(m, Pixel2T<T>{T(0.5), sign * half_pi}).status, StatusCode::DOMAIN_ERROR);
    EXPECT_EQ(unproject(m, Pixel2T<T>{T(0.5), sign * (half_pi - T(1e-3))}).status, StatusCode::OK);
  }
}

}  // namespace

TEST(LatLongProjection, SeamIsTheSameInBothPrecisions)
{
  for (const auto type : kTypes)
  {
    checkSeam<float>(type);
    checkSeam<double>(type);
  }
}

TEST(LatLongProjection, PoleRowsAgreeWithTheForwardMap)
{
  checkPoleApproach<float>();
  checkPoleApproach<double>();
}

// With a long focal length the rounding slack at the seam and the poles is
// capped at a hundredth of a pixel: whole pixels outside the image stay
// outside instead of collapsing onto the limit.
TEST(LatLongProjection, LimitSlackStaysBelowAPixel)
{
  CameraModel64 m = latLongModel<double>(ProjectionModelType::EQUIRECTANGULAR);
  m.intrinsics.fx = 1e6;
  m.intrinsics.fy = 1e6;
  m.intrinsics.cx = kPi * 1e6 + 5.0;        // the seam is at u = 5
  m.intrinsics.cy = 0.5 * kPi * 1e6 + 5.0;  // the upper pole is at v = 5
  ASSERT_EQ(validateCameraModel64(m), StatusCode::OK);

  for (const double u : {0.0, 1.0, 2.0, 3.0, 4.0, 4.9})
  {
    EXPECT_EQ(pixelToRay64(m, Pixel2d{u, m.intrinsics.cy}).status, StatusCode::OUT_OF_FOV)
      << "u " << u;
  }
  for (const double u : {5.0, 5.0 - 0.005})
  {
    const RayResult64 ray = pixelToRay64(m, Pixel2d{u, m.intrinsics.cy});
    ASSERT_EQ(ray.status, StatusCode::OK) << "u " << u;
    const PixelResult64 px = rayToPixel64(m, ray.ray.direction);
    ASSERT_EQ(px.status, StatusCode::OK) << "u " << u;
    EXPECT_NEAR(px.pixel.u, 5.0, 1e-6) << "u " << u;
  }

  for (const double v : {0.0, 1.0, 2.0, 3.0, 4.0, 4.9})
  {
    EXPECT_EQ(pixelToRay64(m, Pixel2d{m.intrinsics.cx, v}).status, StatusCode::OUT_OF_FOV)
      << "v " << v;
  }
  EXPECT_EQ(pixelToRay64(m, Pixel2d{m.intrinsics.cx, 5.0}).status, StatusCode::DOMAIN_ERROR);
}

TEST(LatLongProjection, FullSphereEdgesRoundTrip)
{
  // The image edges of a whole-sphere layout: column -0.5 is the seam and
  // comes back as column -0.5 in both precisions; the last column stays on
  // the right.
  for (const auto type : kTypes)
  {
    const CameraModel64 m64 = latLongModel<double>(type);
    const CameraModel m = latLongModel<float>(type);
    const RayResult64 left64 = pixelToRay64(m64, Pixel2d{-0.5, 511.5});
    ASSERT_EQ(left64.status, StatusCode::OK) << toString(type);
    EXPECT_NEAR(rayToPixel64(m64, left64.ray.direction).pixel.u, -0.5, 1e-9);
    const RayResult left = pixelToRay(m, Pixel2{-0.5f, 511.5f});
    ASSERT_EQ(left.status, StatusCode::OK) << toString(type);
    EXPECT_NEAR(rayToPixel(m, left.ray.direction).pixel.u, -0.5f, 1e-3f);
    // The float model widened to double keeps its left edge too (its float
    // fx puts that edge a hair past the seam, about 4e-5 px).
    const CameraModel64 widened = toCameraModel64(m);
    const RayResult64 left_widened = pixelToRay64(widened, Pixel2d{-0.5, 511.5});
    ASSERT_EQ(left_widened.status, StatusCode::OK) << toString(type);
    EXPECT_NEAR(rayToPixel64(widened, left_widened.ray.direction).pixel.u, -0.5, 1e-3);
    for (const float u : {2047.0f, 2047.4f})
    {
      const RayResult right = pixelToRay(m, Pixel2{u, 511.5f});
      ASSERT_EQ(right.status, StatusCode::OK) << toString(type);
      EXPECT_NEAR(rayToPixel(m, right.ray.direction).pixel.u, u, 1e-3f);
      const RayResult64 right64 = pixelToRay64(m64, Pixel2d{u, 511.5});
      ASSERT_EQ(right64.status, StatusCode::OK) << toString(type);
      EXPECT_NEAR(rayToPixel64(m64, right64.ray.direction).pixel.u, u, 1e-9);
    }
  }
}

TEST(LatLongProjection, RoundTripBothPrecisions)
{
  for (const auto type : kTypes)
  {
    const CameraModel64 m64 = latLongModel<double>(type);
    const CameraModel m = latLongModel<float>(type);
    for (const Eigen::Vector3d &ray : sphereRays(2000))
    {
      const PixelResult64 px64 = rayToPixel64(m64, ray);
      ASSERT_EQ(px64.status, StatusCode::OK);
      const RayResult64 back64 = pixelToRay64(m64, px64.pixel);
      ASSERT_EQ(back64.status, StatusCode::OK);
      EXPECT_LT((back64.ray.direction - ray).norm(), 1e-12);

      const PixelResult px = rayToPixel(m, ray.cast<float>());
      ASSERT_EQ(px.status, StatusCode::OK);
      // Near the poles the cylinder's v grows like tan(phi); allow the float
      // relative precision on top of the absolute tolerance.
      EXPECT_NEAR(px.pixel.u, px64.pixel.u, 2e-3);
      EXPECT_NEAR(px.pixel.v, px64.pixel.v, 2e-3 + 1e-6 * std::abs(px64.pixel.v));
      const RayResult back = pixelToRay(m, px.pixel);
      ASSERT_EQ(back.status, StatusCode::OK);
      EXPECT_LT((back.ray.direction.cast<double>() - ray).norm(), 1e-5);
    }
  }
}

TEST(LatLongProjection, JacobianMatchesFiniteDifferences)
{
  for (const auto type : kTypes)
  {
    CameraModel64 m = latLongModel<double>(type);
    m.intrinsics.skew = 3.0;
    for (const Eigen::Vector3d &unit : sphereRays(300))
    {
      const Eigen::Vector3d ray = 2.5 * unit;  // not unit length on purpose
      const ProjectionJacobian64 jac = rayToPixelWithJacobian64(m, ray);
      ASSERT_EQ(jac.status, StatusCode::OK);
      const PixelResult64 px = rayToPixel64(m, ray);
      EXPECT_EQ(jac.pixel.u, px.pixel.u);
      EXPECT_EQ(jac.pixel.v, px.pixel.v);
      for (int k = 0; k < 3; ++k)
      {
        const double h = 1e-6;
        Eigen::Vector3d plus = ray;
        Eigen::Vector3d minus = ray;
        plus[k] += h;
        minus[k] -= h;
        const PixelResult64 a = rayToPixel64(m, plus);
        const PixelResult64 b = rayToPixel64(m, minus);
        ASSERT_EQ(a.status, StatusCode::OK);
        ASSERT_EQ(b.status, StatusCode::OK);
        // Rays this close to the seam would straddle it.
        if (std::abs(a.pixel.u - b.pixel.u) > 100.0) continue;
        const double du = (a.pixel.u - b.pixel.u) / (2 * h);
        const double dv = (a.pixel.v - b.pixel.v) / (2 * h);
        const double scale = 1.0 + std::abs(jac.J(0, k)) + std::abs(jac.J(1, k));
        EXPECT_NEAR(jac.J(0, k), du, 1e-5 * scale);
        EXPECT_NEAR(jac.J(1, k), dv, 1e-5 * scale);
      }
    }

    const CameraModel mf = fromCameraModel64(m);
    const ProjectionJacobian jf = rayToPixelWithJacobian(mf, Eigen::Vector3f(0.3f, -0.2f, 0.9f));
    const ProjectionJacobian64 jd = rayToPixelWithJacobian64(m, Eigen::Vector3d(0.3, -0.2, 0.9));
    ASSERT_EQ(jf.status, StatusCode::OK);
    EXPECT_LT((jf.J.cast<double>() - jd.J).norm(), 1e-3 * jd.J.norm());
  }
}

TEST(LatLongProjection, BatchMatchesScalar)
{
  for (const auto type : kTypes)
  {
    const CameraModel m = latLongModel<float>(type);
    const CameraModel64 m64 = latLongModel<double>(type);
    std::vector<Eigen::Vector3d> rays = sphereRays(257);
    rays.emplace_back(0.0, -1.0, 0.0);  // a pole
    const int n = static_cast<int>(rays.size());

    Eigen::Matrix3Xf dirs(3, n);
    Eigen::Matrix3Xd dirs64(3, n);
    for (int i = 0; i < n; ++i)
    {
      dirs.col(i) = rays[i].cast<float>();
      dirs64.col(i) = rays[i];
    }
    Eigen::Matrix2Xf px(2, n);
    Eigen::Matrix2Xd px64(2, n);
    std::vector<StatusCode> st(n), st64(n);
    EXPECT_EQ(rayToPixelBatch(m, dirs, px, st.data()), n - 1);
    EXPECT_EQ(rayToPixelBatch64(m64, dirs64, px64, st64.data()), n - 1);
    for (int i = 0; i < n; ++i)
    {
      const PixelResult s = rayToPixel(m, Eigen::Vector3f(dirs.col(i)));
      const PixelResult64 s64 = rayToPixel64(m64, Eigen::Vector3d(dirs64.col(i)));
      EXPECT_EQ(st[i], s.status);
      EXPECT_EQ(st64[i], s64.status);
      if (s.status != StatusCode::OK) continue;
      EXPECT_EQ(px(0, i), s.pixel.u);
      EXPECT_EQ(px(1, i), s.pixel.v);
      EXPECT_EQ(px64(0, i), s64.pixel.u);
      EXPECT_EQ(px64(1, i), s64.pixel.v);
    }

    Eigen::Matrix2Xf pixels(2, n - 1);
    Eigen::Matrix2Xd pixels64(2, n - 1);
    pixels = px.leftCols(n - 1);
    pixels64 = px64.leftCols(n - 1);
    Eigen::Matrix3Xf out(3, n - 1);
    Eigen::Matrix3Xd out64(3, n - 1);
    EXPECT_EQ(pixelToRayBatch(m, pixels, out), n - 1);
    EXPECT_EQ(pixelToRayBatch64(m64, pixels64, out64), n - 1);
    for (int i = 0; i < n - 1; ++i)
    {
      const RayResult s = pixelToRay(m, Pixel2{pixels(0, i), pixels(1, i)});
      const RayResult64 s64 = pixelToRay64(m64, Pixel2d{pixels64(0, i), pixels64(1, i)});
      EXPECT_EQ(Eigen::Vector3f(out.col(i)), s.ray.direction);
      EXPECT_EQ(Eigen::Vector3d(out64.col(i)), s64.ray.direction);
    }
  }
}

TEST(LatLongProjection, ValidatedModelAndLutMatchTheGenericPath)
{
  for (const auto type : kTypes)
  {
    CameraModel m = latLongModel<float>(type);
    m.intrinsics.fx = 128.0f / (2.0f * constants::kPiF);
    m.intrinsics.fy =
      type == ProjectionModelType::EQUIRECTANGULAR ? 64.0f / constants::kPiF : 20.0f;
    m.intrinsics.cx = 63.5f;
    m.intrinsics.cy = 31.5f;
    const std::optional<ValidatedCameraModel> v = ValidatedCameraModel::tryMake(m);
    ASSERT_TRUE(v.has_value());

    InverseLut lut;
    ASSERT_GT(lut.build(m, 128, 64), 0);
    // The LUT interpolates each pixel from the grid node one pixel further
    // on, which lies past the image edge for the last row and column.
    for (int row = 0; row < 63; row += 7)
    {
      for (int col = 0; col < 127; col += 5)
      {
        const Pixel2 p{static_cast<float>(col), static_cast<float>(row)};
        const RayResult generic = pixelToRay(m, p);
        ASSERT_EQ(generic.status, StatusCode::OK);
        const RayResult wrapped = v->pixelToRay(p);
        EXPECT_EQ(wrapped.status, generic.status);
        EXPECT_EQ(wrapped.ray.direction, generic.ray.direction);
        const RayResult from_lut = lut.query(p.u, p.v);
        ASSERT_EQ(from_lut.status, StatusCode::OK);
        EXPECT_LT((from_lut.ray.direction - generic.ray.direction).norm(), 1e-6f);

        const PixelResult forward = rayToPixel(m, generic.ray.direction);
        const PixelResult forward_wrapped = v->rayToPixel(generic.ray.direction);
        EXPECT_EQ(forward_wrapped.status, forward.status);
        EXPECT_EQ(forward_wrapped.pixel.u, forward.pixel.u);
        EXPECT_EQ(forward_wrapped.pixel.v, forward.pixel.v);
      }
    }
  }
}

TEST(LatLongProjection, DistortionFreeCopyIsTheSameModel)
{
  for (const auto type : kTypes)
  {
    const CameraModel m = latLongModel<float>(type);
    const CameraModel free = makeDistortionFree(m);
    EXPECT_EQ(free.projection.type, type);
    EXPECT_EQ(free.intrinsics.fx, m.intrinsics.fx);
    EXPECT_EQ(free.intrinsics.fy, m.intrinsics.fy);
    EXPECT_EQ(free.intrinsics.cx, m.intrinsics.cx);
    EXPECT_EQ(free.intrinsics.cy, m.intrinsics.cy);
    EXPECT_EQ(free.distortion.type, DistortionModelType::NONE);
  }
}

TEST(LatLongProjection, NotCalibrationModels)
{
  namespace tpl = camxiom::projection_template;
  for (const auto type : kTypes)
  {
    const CameraModel64 m = latLongModel<double>(type);
    // The parameter Jacobian and the autodiff template serve calibration only.
    EXPECT_EQ(
      rayToPixelWithFullJacobian64(m, Eigen::Vector3d(0.1, 0.2, 1.0)).status,
      StatusCode::INVALID_MODEL
    );
    double u = 0.0;
    double v = 0.0;
    EXPECT_FALSE(tpl::projectGeneric(m, 0.1, 0.2, 1.0, u, v));

    CameraModel64 updated = m;
    updated.projection.theta_max = 0.5;
    updateThetaMax64(updated);
    EXPECT_EQ(updated.projection.theta_max, kPi);
  }
}
