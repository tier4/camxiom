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

// The ideal fisheye laws (equidistant, equisolid, stereographic, orthographic
// with no lens correction) unproject in closed form. These tests pin the law
// each inverse undoes, the statuses at the edges of the field of view, and
// the batch path against the scalar one.

#include <camxiom/batch.hpp>
#include <camxiom/internal/constants.hpp>
#include <camxiom/model.hpp>
#include <camxiom/projection.hpp>
#include <camxiom/projection64.hpp>
#include <camxiom/types.hpp>
#include <camxiom/types64.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace camxiom;

constexpr double kPi = constants::kPi;

const ProjectionModelType kLaws[] = {
  ProjectionModelType::EQUIDISTANT, ProjectionModelType::EQUISOLID,
  ProjectionModelType::STEREOGRAPHIC, ProjectionModelType::ORTHOGRAPHIC};

// Image-plane radius of a ray theta off the axis, computed independently of
// the library so a wrong scale in either direction shows up.
double lawRadius(const ProjectionModelType law, const double theta)
{
  switch (law)
  {
    case ProjectionModelType::EQUISOLID:
      return 2.0 * std::sin(theta / 2.0);
    case ProjectionModelType::STEREOGRAPHIC:
      return 2.0 * std::tan(theta / 2.0);
    case ProjectionModelType::ORTHOGRAPHIC:
      return std::sin(theta);
    default:
      return theta;
  }
}

CameraModel lawModel(const ProjectionModelType law)
{
  CameraModel m;
  m.projection.type = law;
  m.intrinsics.fx = 300.0f;
  m.intrinsics.fy = 280.0f;
  m.intrinsics.cx = 639.5f;
  m.intrinsics.cy = 479.5f;
  updateThetaMax(m);
  return m;
}

// Pixel at image-plane radius r along azimuth phi (no skew in lawModel).
template <typename Model>
auto pixelAt(const Model &m, const double r, const double phi)
{
  using T = decltype(m.intrinsics.fx);
  return Pixel2T<T>{
    static_cast<T>(m.intrinsics.cx + m.intrinsics.fx * r * std::cos(phi)),
    static_cast<T>(m.intrinsics.cy + m.intrinsics.fy * r * std::sin(phi))};
}

double offAxisAngle(const Eigen::Vector3d &d) { return std::atan2(d.head<2>().norm(), d.z()); }

}  // namespace

// Known points of each law: the inverse returns the angle the law maps to
// that radius (e.g. stereographic 2 tan(theta / 2) = 2 at theta = pi / 2).
TEST(FisheyeLawInverse, UndoesEachLawAtKnownPoints)
{
  struct Point
  {
    ProjectionModelType law;
    double radius;
    double theta;
  };
  const Point points[] = {
    {ProjectionModelType::EQUIDISTANT, 1.0, 1.0},
    {ProjectionModelType::EQUISOLID, std::sqrt(2.0), kPi / 2.0},
    {ProjectionModelType::STEREOGRAPHIC, 2.0, kPi / 2.0},
    {ProjectionModelType::ORTHOGRAPHIC, 0.5, kPi / 6.0}};
  for (const Point &p : points)
  {
    const CameraModel64 m = toCameraModel64(lawModel(p.law));
    const RayResult64 ray = pixelToRay64(m, pixelAt(m, p.radius, 0.3));
    ASSERT_EQ(ray.status, StatusCode::OK) << toString(p.law);
    EXPECT_NEAR(offAxisAngle(ray.ray.direction), p.theta, 1e-12) << toString(p.law);
    EXPECT_NEAR(std::atan2(ray.ray.direction.y(), ray.ray.direction.x()), 0.3, 1e-12);
  }
}

// Over the whole field of view, in both precisions, the inverse recovers the
// angle of the ray the law maps to that radius.
TEST(FisheyeLawInverse, RecoversTheAngleAcrossTheField)
{
  for (const auto law : kLaws)
  {
    const CameraModel mf = lawModel(law);
    const CameraModel64 md = toCameraModel64(mf);
    const double cap = md.projection.theta_max;
    for (int i = 0; i < 400; ++i)
    {
      const double theta = cap * i / 400.0;
      const double phi = 0.017 * i;
      const RayResult64 rd = pixelToRay64(md, pixelAt(md, lawRadius(law, theta), phi));
      ASSERT_EQ(rd.status, StatusCode::OK) << toString(law) << " theta " << theta;
      EXPECT_NEAR(offAxisAngle(rd.ray.direction), theta, 1e-9) << toString(law) << " " << theta;

      // float: the law's slope vanishes at the equisolid and orthographic
      // caps, where a float radius only fixes the angle to ~sqrt(eps); a
      // float direction resolves angles only to a few 1e-8 anywhere.
      const double slope = std::abs(lawRadius(law, theta + 1e-6) - lawRadius(law, theta)) / 1e-6;
      const RayResult rf = pixelToRay(mf, pixelAt(mf, lawRadius(law, theta), phi));
      if (rf.status != StatusCode::OK)
      {
        // Only the last float steps at the cap may round past it.
        EXPECT_EQ(rf.status, StatusCode::OUT_OF_FOV) << toString(law) << " " << theta;
        EXPECT_GT(theta, cap - 1e-3) << toString(law);
        continue;
      }
      const double tolerance = slope > 0.05 ? std::max(2e-5 / slope, 3e-7) : 2e-3;
      EXPECT_NEAR(offAxisAngle(rf.ray.direction.cast<double>()), theta, tolerance)
        << toString(law) << " theta " << theta;
    }
  }
}

// Stereographic rays past about 2 rad used to come back NON_CONVERGED from
// the iterative solve; the closed form answers them like any other ray.
TEST(FisheyeLawInverse, StereographicAnswersWideRays)
{
  const CameraModel mf = lawModel(ProjectionModelType::STEREOGRAPHIC);
  const CameraModel64 md = toCameraModel64(mf);
  for (const double theta : {2.0, 2.1, 2.2, 2.5, 3.0})
  {
    const RayResult64 rd = pixelToRay64(md, pixelAt(md, lawRadius(md.projection.type, theta), 1.0));
    ASSERT_EQ(rd.status, StatusCode::OK) << theta;
    EXPECT_NEAR(offAxisAngle(rd.ray.direction), theta, 1e-9);
    const RayResult rf = pixelToRay(mf, pixelAt(mf, lawRadius(mf.projection.type, theta), 1.0));
    ASSERT_EQ(rf.status, StatusCode::OK) << theta;
    EXPECT_NEAR(offAxisAngle(rf.ray.direction.cast<double>()), theta, 1e-5);
  }
}

// The field of view ends where it did with the iterative solve: radii past
// the law's value at theta_max are OUT_OF_FOV, the centre is the axis, and
// non-finite pixels are invalid input.
TEST(FisheyeLawInverse, KeepsTheEdgeStatuses)
{
  for (const auto law : kLaws)
  {
    for (const double cap : {1.2, -1.0})
    {
      CameraModel64 m = toCameraModel64(lawModel(law));
      if (cap > 0.0)
      {
        m.projection.theta_max = cap;
      }
      const double edge = lawRadius(law, m.projection.theta_max);
      const RayResult64 inside = pixelToRay64(m, pixelAt(m, edge * (1.0 - 1e-9), 0.5));
      ASSERT_EQ(inside.status, StatusCode::OK) << toString(law);
      EXPECT_LE(offAxisAngle(inside.ray.direction), m.projection.theta_max + 1e-12);
      EXPECT_EQ(
        pixelToRay64(m, pixelAt(m, edge * (1.0 + 1e-6), 0.5)).status, StatusCode::OUT_OF_FOV
      ) << toString(law);
      EXPECT_EQ(pixelToRay64(m, pixelAt(m, edge + 1.0, 0.5)).status, StatusCode::OUT_OF_FOV)
        << toString(law);
    }

    const CameraModel64 m = toCameraModel64(lawModel(law));
    const RayResult64 centre = pixelToRay64(m, Pixel2d{m.intrinsics.cx, m.intrinsics.cy});
    ASSERT_EQ(centre.status, StatusCode::OK);
    EXPECT_EQ(centre.ray.direction, Eigen::Vector3d(0.0, 0.0, 1.0));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(pixelToRay64(m, Pixel2d{nan, 1.0}).status, StatusCode::INVALID_INPUT);
    const CameraModel mf = lawModel(law);
    EXPECT_EQ(
      pixelToRay(mf, Pixel2{std::numeric_limits<float>::quiet_NaN(), 1.0f}).status,
      StatusCode::INVALID_INPUT
    );
  }

  // The equisolid circle (radius 2) and the orthographic disc (radius 1).
  const CameraModel64 equisolid = toCameraModel64(lawModel(ProjectionModelType::EQUISOLID));
  EXPECT_EQ(
    pixelToRay64(equisolid, pixelAt(equisolid, 2.0001, 0.0)).status, StatusCode::OUT_OF_FOV
  );
  const CameraModel64 ortho = toCameraModel64(lawModel(ProjectionModelType::ORTHOGRAPHIC));
  const double rim_radius = lawRadius(ortho.projection.type, ortho.projection.theta_max);
  const RayResult64 rim = pixelToRay64(ortho, pixelAt(ortho, rim_radius * (1.0 - 1e-12), 0.0));
  ASSERT_EQ(rim.status, StatusCode::OK);
  EXPECT_NEAR(offAxisAngle(rim.ray.direction), ortho.projection.theta_max, 1e-5);
  EXPECT_EQ(pixelToRay64(ortho, pixelAt(ortho, 1.0001, 0.0)).status, StatusCode::OUT_OF_FOV);
}

// The batch path gives the scalar answer for every law, including pixels at
// and past the edge of the field of view.
TEST(FisheyeLawInverse, BatchMatchesScalar)
{
  for (const auto law : kLaws)
  {
    const CameraModel m = lawModel(law);
    std::vector<float> u;
    std::vector<float> v;
    for (int j = -40; j < 1000; j += 13)
    {
      for (int i = -40; i < 1320; i += 11)
      {
        u.push_back(static_cast<float>(i) + 0.25f);
        v.push_back(static_cast<float>(j) + 0.5f);
      }
    }
    const int count = static_cast<int>(u.size());
    std::vector<float> dirs(3 * u.size());
    std::vector<StatusCode> statuses(u.size());
    (void)pixelToRayBatch(m, u.data(), v.data(), count, dirs.data(), statuses.data());
    for (int i = 0; i < count; ++i)
    {
      const RayResult scalar = pixelToRay(m, Pixel2{u[i], v[i]});
      ASSERT_EQ(statuses[i], scalar.status) << toString(law) << " (" << u[i] << ", " << v[i] << ")";
      if (scalar.status == StatusCode::OK)
      {
        for (int k = 0; k < 3; ++k)
        {
          EXPECT_NEAR(dirs[3 * i + k], scalar.ray.direction[k], 2e-5f)
            << toString(law) << " (" << u[i] << ", " << v[i] << ")";
        }
      }
    }
  }
}

namespace
{

// Unproject pixel (r cos phi, r sin phi) of a unit-intrinsics model and check
// that an OK ray projects back onto it and that a refused one is refused as
// the forward map refuses the back axis or the field of view. A pixel clearly
// inside the rim (must_be_ok) has to come back OK, so the back-axis test
// cannot spread over valid pixels.
template <typename T>
void checkReprojects(
  const CameraModelT<T> &m, const T r, const T phi, const char *what, const bool must_be_ok = false
)
{
  const Pixel2T<T> px{r * std::cos(phi), r * std::sin(phi)};
  RayResultT<T> ray;
  PixelResultT<T> back;
  if constexpr (std::is_same_v<T, float>)
  {
    ray = pixelToRay(m, px);
    back = rayToPixel(m, ray.ray.direction);
  }
  else
  {
    ray = pixelToRay64(m, px);
    back = rayToPixel64(m, ray.ray.direction);
  }
  if (must_be_ok)
  {
    ASSERT_EQ(ray.status, StatusCode::OK) << what << " r " << r;
  }
  if (ray.status != StatusCode::OK)
  {
    EXPECT_TRUE(ray.status == StatusCode::DOMAIN_ERROR || ray.status == StatusCode::OUT_OF_FOV)
      << what << " r " << r << ": " << toString(ray.status);
    return;
  }
  ASSERT_EQ(back.status, StatusCode::OK) << what << " r " << r;
  EXPECT_NEAR(back.pixel.u, px.u, T(1e-3)) << what << " r " << r;
  EXPECT_NEAR(back.pixel.v, px.v, T(1e-3)) << what << " r " << r;
}

// Image radius of the rim (theta_max) in the model's own precision, in the
// same order of operations as the library's forward law.
template <typename T>
T rimRadius(const CameraModelT<T> &m)
{
  const T theta = m.projection.theta_max;
  if (m.projection.type == ProjectionModelType::EQUISOLID)
  {
    return T(2) * std::sin(theta * T(0.5));
  }
  if (m.distortion.type == DistortionModelType::KB4)
  {
    const T t3 = theta * theta * theta;
    return theta + m.distortion.coeffs[0] * t3;
  }
  return theta;
}

template <typename T>
void checkBackAxis(const CameraModelT<T> &m, const char *what)
{
  const T cap_radius = rimRadius(m);
  // The rim radius itself is the back axis: refused, not flipped.
  if constexpr (std::is_same_v<T, float>)
  {
    EXPECT_EQ(pixelToRay(m, Pixel2T<T>{cap_radius, T(0)}).status, StatusCode::DOMAIN_ERROR) << what;
  }
  else
  {
    EXPECT_EQ(pixelToRay64(m, Pixel2T<T>{cap_radius, T(0)}).status, StatusCode::DOMAIN_ERROR)
      << what;
  }
  for (const T phi : {T(0), T(0.4), T(2.5), T(-1.9)})
  {
    T r = cap_radius;
    for (int step = 0; step < 24; ++step)
    {
      checkReprojects(m, r, phi, what);
      r = std::nextafter(r, T(0));
    }
    checkReprojects(m, cap_radius - T(1e-6), phi, what);
    for (const T below : {T(1e-4), T(1e-3), T(1e-2)})
    {
      checkReprojects(m, cap_radius - below, phi, what, /*must_be_ok=*/true);
    }
    checkReprojects(m, std::nextafter(cap_radius, T(10)), phi, what);
  }
}

template <typename T>
CameraModelT<T> backAxisModel(const ProjectionModelType law, const DistortionModelType correction)
{
  CameraModelT<T> m;
  m.projection.type = law;
  m.distortion.type = correction;
  if (correction == DistortionModelType::KB4)
  {
    m.distortion.count = 4U;
    m.distortion.coeffs[0] = static_cast<T>(0.001);
  }
  // Float pi exceeds the true one; double takes the true pi. Both validate.
  m.projection.theta_max =
    std::is_same_v<T, float> ? static_cast<T>(constants::kPiF) : static_cast<T>(constants::kPi);
  return m;
}

}  // namespace

// Laws that reach theta = pi map the back axis to the whole rim circle. The
// inverse refuses the rim like the forward map refuses the back axis, never
// flips a ray across the axis, and every ray it returns projects back.
TEST(FisheyeLawInverse, BackAxisIsRefusedLikeTheForwardMap)
{
  const std::pair<ProjectionModelType, DistortionModelType> laws[] = {
    {ProjectionModelType::EQUIDISTANT, DistortionModelType::NONE},
    {ProjectionModelType::EQUISOLID, DistortionModelType::NONE},
    {ProjectionModelType::EQUIDISTANT, DistortionModelType::KB4}};
  for (const auto &[law, correction] : laws)
  {
    const CameraModel mf = backAxisModel<float>(law, correction);
    const CameraModel64 md = backAxisModel<double>(law, correction);
    ASSERT_EQ(validateCameraModel(mf), StatusCode::OK) << toString(law);
    ASSERT_EQ(validateCameraModel64(md), StatusCode::OK) << toString(law);
    checkBackAxis(mf, toString(law));
    checkBackAxis(md, toString(law));

    // The batch path agrees, its SIMD groups and its scalar tail alike.
    std::vector<float> u;
    std::vector<float> v;
    float r = rimRadius(mf);
    for (int i = 0; i < 37; ++i)
    {
      u.push_back(r * std::cos(0.3f * i));
      v.push_back(r * std::sin(0.3f * i));
      r = i % 3 == 0 ? std::nextafter(r, 0.0f) : r - 1e-4f;
    }
    const int count = static_cast<int>(u.size());
    std::vector<float> dirs(3 * u.size());
    std::vector<StatusCode> statuses(u.size());
    (void)pixelToRayBatch(mf, u.data(), v.data(), count, dirs.data(), statuses.data());
    for (int i = 0; i < count; ++i)
    {
      const RayResult scalar = pixelToRay(mf, Pixel2{u[i], v[i]});
      EXPECT_EQ(statuses[i], scalar.status) << toString(law) << " lane " << i;
    }
  }
}
