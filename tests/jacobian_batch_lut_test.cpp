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

// Characterization tests for the jacobian / batch / LUT subsystems (#1 step 5).
//
// These subsystems previously had no direct gtest coverage (their only in-tree
// consumer is the optional OpenCV layer). Before unifying the hand-duplicated
// float and double implementations onto scalar-templated cores, we pin their
// observable behaviour here so the refactor is provably behaviour-preserving:
//
//   - rayToPixelWithJacobian{,64}: the analytic 2x3 Jacobian must match a
//     central finite-difference of rayToPixel, its pixel must match rayToPixel,
//     and the float result must match the double result (parity).
//   - rayToPixelBatch{,64} / pixelToRayBatch{,64}: the batch results (both the
//     Eigen and the raw-pointer variants) must match the per-point projection
//     API element for element, plus float/double parity.
//   - InverseLut{,64}: a step-1 LUT queried at grid coordinates must reproduce
//     pixelToRay, plus float/double parity.

#include "camxiom/batch.hpp"
#include "camxiom/batch64.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/jacobian.hpp"
#include "camxiom/jacobian64.hpp"
#include "camxiom/jacobian_batch.hpp"
#include "camxiom/jacobian_batch64.hpp"
#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/lut.hpp"
#include "camxiom/lut64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

using namespace camxiom;

namespace
{

CameraModel makePinholeRadtan5()
{
  CameraModel m;
  m.intrinsics.fx = 520.0f;
  m.intrinsics.fy = 519.0f;
  m.intrinsics.cx = 322.0f;
  m.intrinsics.cy = 241.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.28f;
  m.distortion.coeffs[1] = 0.11f;
  m.distortion.coeffs[2] = 0.0006f;
  m.distortion.coeffs[3] = -0.0004f;
  m.distortion.coeffs[4] = -0.02f;
  m.distortion.count = 5U;
  return m;
}

CameraModel makeFisheyeEquidistant()
{
  CameraModel m;
  m.intrinsics.fx = 280.0f;
  m.intrinsics.fy = 280.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi) - 1e-4f;
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeFisheyeKb4()
{
  CameraModel m;
  m.intrinsics.fx = 285.0f;
  m.intrinsics.fy = 285.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = 3.1415926f - 1e-3f;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = -0.012f;
  m.distortion.coeffs[1] = 0.004f;
  m.distortion.coeffs[2] = -0.0007f;
  m.distortion.coeffs[3] = 0.0001f;
  m.distortion.count = 4U;
  return m;
}

CameraModel makeOmnidirectional()
{
  CameraModel m;
  m.intrinsics.fx = 400.0f;
  m.intrinsics.fy = 400.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 0.6f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeDoubleSphere()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 0.5f;
  m.projection.alpha = 0.5f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeEucm()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.alpha = 0.6f;
  m.projection.beta = 1.1f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

// Cached tilt matrices (mirrors src/model/internal.hpp computeTiltMatrices) so
// the TILTED14 model exercises the tilt branch of the distortion Jacobian.
void computeTiltMatrices(
  float tau_x, float tau_y, std::array<float, 9> &tilt, std::array<float, 9> &inv_tilt
)
{
  const float c_tx = std::cos(tau_x);
  const float s_tx = std::sin(tau_x);
  const float c_ty = std::cos(tau_y);
  const float s_ty = std::sin(tau_y);

  const std::array<float, 9> rot_xy{c_ty, s_ty * s_tx, -s_ty * c_tx, 0.0f,       c_tx,
                                    s_tx, s_ty,        -c_ty * s_tx, c_ty * c_tx};

  const float r22 = rot_xy[8];
  const std::array<float, 9> proj_z{r22, 0.0f, -rot_xy[2], 0.0f, r22, -rot_xy[5], 0.0f, 0.0f, 1.0f};

  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      float value = 0.0f;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += proj_z[row * 3 + mid] * rot_xy[mid * 3 + col];
      }
      tilt[row * 3 + col] = value;
    }
  }

  const float inv_r22 = 1.0f / r22;
  const std::array<float, 9> inv_proj_z{
    inv_r22, 0.0f, rot_xy[2] * inv_r22, 0.0f, inv_r22, rot_xy[5] * inv_r22, 0.0f, 0.0f, 1.0f};
  const std::array<float, 9> rot_xy_t{rot_xy[0], rot_xy[3], rot_xy[6], rot_xy[1], rot_xy[4],
                                      rot_xy[7], rot_xy[2], rot_xy[5], rot_xy[8]};
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      float value = 0.0f;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += rot_xy_t[row * 3 + mid] * inv_proj_z[mid * 3 + col];
      }
      inv_tilt[row * 3 + col] = value;
    }
  }
}

CameraModel makePinholeTilted14()
{
  CameraModel m;
  m.intrinsics.fx = 520.0f;
  m.intrinsics.fy = 519.0f;
  m.intrinsics.cx = 322.0f;
  m.intrinsics.cy = 241.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::TILTED14;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.28f;
  m.distortion.coeffs[1] = 0.11f;
  m.distortion.coeffs[2] = 0.0006f;
  m.distortion.coeffs[3] = -0.0004f;
  m.distortion.coeffs[4] = -0.02f;
  m.distortion.coeffs[5] = 0.001f;
  m.distortion.coeffs[6] = 0.0005f;
  m.distortion.coeffs[7] = -0.0002f;
  m.distortion.coeffs[8] = 0.0003f;
  m.distortion.coeffs[9] = -0.0002f;
  m.distortion.coeffs[10] = 0.0001f;
  m.distortion.coeffs[11] = 0.0002f;
  m.distortion.coeffs[12] = 0.02f;
  m.distortion.coeffs[13] = -0.015f;
  m.distortion.count = 14U;
  m.distortion.is_rational = true;
  m.distortion.has_thin_prism = true;
  m.distortion.has_tilt = true;
  computeTiltMatrices(
    m.distortion.coeffs[12], m.distortion.coeffs[13], m.distortion.tilt_matrix,
    m.distortion.inv_tilt_matrix
  );
  return m;
}

std::vector<Eigen::Vector3d> sampleRays()
{
  return {
    Eigen::Vector3d(0.0, 0.0, 1.0),    Eigen::Vector3d(0.15, 0.10, 1.0),
    Eigen::Vector3d(-0.20, 0.12, 1.0), Eigen::Vector3d(0.30, -0.18, 1.0),
    Eigen::Vector3d(0.40, 0.25, 0.9),
  };
}

// ---------------------------------------------------------------------------
// Jacobian: analytic vs central finite difference (double), plus float parity.
// ---------------------------------------------------------------------------
void checkJacobian(const CameraModel &model_f, const Eigen::Vector3d &ray)
{
  const CameraModel64 model = toCameraModel64(model_f);

  const ProjectionJacobian64 jac = rayToPixelWithJacobian64(model, ray);
  const PixelResult64 px = rayToPixel64(model, ray);
  ASSERT_EQ(px.status, StatusCode::OK) << "setup: rayToPixel64 must succeed for the sampled ray";
  ASSERT_EQ(jac.status, StatusCode::OK)
    << "rayToPixelWithJacobian64 failed for model " << toString(model.projection.type);

  // The Jacobian call must reproduce the plain forward pixel.
  EXPECT_NEAR(jac.pixel.u, px.pixel.u, 1e-6);
  EXPECT_NEAR(jac.pixel.v, px.pixel.v, 1e-6);

  // Central finite difference of rayToPixel64 w.r.t. (X, Y, Z).
  const double h = 1e-6;
  for (int c = 0; c < 3; ++c)
  {
    Eigen::Vector3d rp = ray;
    Eigen::Vector3d rm = ray;
    rp[c] += h;
    rm[c] -= h;
    const PixelResult64 pp = rayToPixel64(model, rp);
    const PixelResult64 pm = rayToPixel64(model, rm);
    ASSERT_EQ(pp.status, StatusCode::OK);
    ASSERT_EQ(pm.status, StatusCode::OK);

    const double du = (pp.pixel.u - pm.pixel.u) / (2.0 * h);
    const double dv = (pp.pixel.v - pm.pixel.v) / (2.0 * h);

    EXPECT_NEAR(jac.J(0, c), du, 1e-3 * std::abs(du) + 1e-3)
      << "d u / d[" << c << "] mismatch, model " << toString(model.projection.type);
    EXPECT_NEAR(jac.J(1, c), dv, 1e-3 * std::abs(dv) + 1e-3)
      << "d v / d[" << c << "] mismatch, model " << toString(model.projection.type);
  }

  // Float vs double parity.
  const Eigen::Vector3f ray_f(
    static_cast<float>(ray.x()), static_cast<float>(ray.y()), static_cast<float>(ray.z())
  );
  const ProjectionJacobian jac_f = rayToPixelWithJacobian(model_f, ray_f);
  ASSERT_EQ(jac_f.status, StatusCode::OK);
  for (int r = 0; r < 2; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      EXPECT_NEAR(
        static_cast<double>(jac_f.J(r, c)), jac.J(r, c), 2e-3 * std::abs(jac.J(r, c)) + 1e-2
      ) << "float/double Jacobian parity mismatch ("
        << r << "," << c << "), model " << toString(model.projection.type);
    }
  }
}

TEST(JacobianChar, AnalyticMatchesFiniteDifferenceAndParity)
{
  const std::vector<CameraModel> models{makePinholeRadtan5(),     makePinholeTilted14(),
                                        makeFisheyeEquidistant(), makeOmnidirectional(),
                                        makeDoubleSphere(),       makeEucm()};
  for (const auto &m : models)
  {
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
    for (const auto &ray : sampleRays())
    {
      checkJacobian(m, ray);
    }
  }
}

// ---------------------------------------------------------------------------
// Batch: Eigen + raw-pointer variants vs per-point projection, plus parity.
// ---------------------------------------------------------------------------
TEST(BatchChar, ForwardMatchesPerPoint)
{
  const std::vector<CameraModel> models{
    makePinholeRadtan5(), makeFisheyeEquidistant(), makeOmnidirectional(), makeDoubleSphere(),
    makeEucm()};
  const auto rays = sampleRays();
  const int n = static_cast<int>(rays.size());

  for (const auto &model : models)
  {
    ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
    const CameraModel64 model64 = toCameraModel64(model);

    Eigen::Matrix3Xf R(3, n);
    Eigen::Matrix3Xd Rd(3, n);
    for (int i = 0; i < n; ++i)
    {
      R.col(i) = rays[i].cast<float>();
      Rd.col(i) = rays[i];
    }

    Eigen::Matrix2Xf P(2, n);
    std::vector<StatusCode> st(static_cast<std::size_t>(n));
    const int ok = rayToPixelBatch(model, R, P, st.data());
    EXPECT_GE(ok, 0);

    // Raw-pointer variant must agree with the Eigen variant.
    std::vector<float> rays_xyz(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i)
    {
      rays_xyz[static_cast<std::size_t>(3 * i + 0)] = R(0, i);
      rays_xyz[static_cast<std::size_t>(3 * i + 1)] = R(1, i);
      rays_xyz[static_cast<std::size_t>(3 * i + 2)] = R(2, i);
    }
    std::vector<float> u_out(static_cast<std::size_t>(n));
    std::vector<float> v_out(static_cast<std::size_t>(n));
    std::vector<StatusCode> st_raw(static_cast<std::size_t>(n));
    const int ok_raw =
      rayToPixelBatch(model, rays_xyz.data(), n, u_out.data(), v_out.data(), st_raw.data());
    EXPECT_EQ(ok_raw, ok);

    Eigen::Matrix2Xd Pd(2, n);
    std::vector<StatusCode> std64(static_cast<std::size_t>(n));
    const int ok64 = rayToPixelBatch64(model64, Rd, Pd, std64.data());
    EXPECT_EQ(ok64, ok);

    for (int i = 0; i < n; ++i)
    {
      const PixelResult single = rayToPixel(model, R.col(i));
      EXPECT_EQ(st[static_cast<std::size_t>(i)], single.status);
      EXPECT_EQ(st_raw[static_cast<std::size_t>(i)], single.status);
      if (single.status == StatusCode::OK)
      {
        EXPECT_NEAR(P(0, i), single.pixel.u, 1e-3);
        EXPECT_NEAR(P(1, i), single.pixel.v, 1e-3);
        EXPECT_NEAR(u_out[static_cast<std::size_t>(i)], single.pixel.u, 1e-3);
        EXPECT_NEAR(v_out[static_cast<std::size_t>(i)], single.pixel.v, 1e-3);
        // float/double parity.
        EXPECT_NEAR(static_cast<double>(P(0, i)), Pd(0, i), 1e-2 * std::abs(Pd(0, i)) + 1e-1);
        EXPECT_NEAR(static_cast<double>(P(1, i)), Pd(1, i), 1e-2 * std::abs(Pd(1, i)) + 1e-1);
      }
    }
  }
}

TEST(BatchChar, InverseMatchesPerPoint)
{
  const std::vector<CameraModel> models{
    makePinholeRadtan5(), makeFisheyeEquidistant(), makeOmnidirectional(), makeDoubleSphere(),
    makeEucm()};

  for (const auto &model : models)
  {
    ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

    // Build a set of in-image pixels by projecting sample rays forward.
    std::vector<Pixel2> pixels;
    for (const auto &ray : sampleRays())
    {
      const PixelResult px = rayToPixel(model, ray.cast<float>());
      if (px.status == StatusCode::OK)
      {
        pixels.push_back(px.pixel);
      }
    }
    ASSERT_FALSE(pixels.empty());
    const int n = static_cast<int>(pixels.size());

    Eigen::Matrix2Xf Pin(2, n);
    for (int i = 0; i < n; ++i)
    {
      Pin(0, i) = pixels[static_cast<std::size_t>(i)].u;
      Pin(1, i) = pixels[static_cast<std::size_t>(i)].v;
    }

    Eigen::Matrix3Xf D(3, n);
    std::vector<StatusCode> st(static_cast<std::size_t>(n));
    const int ok = pixelToRayBatch(model, Pin, D, st.data());
    EXPECT_GE(ok, 0);

    for (int i = 0; i < n; ++i)
    {
      const RayResult single = pixelToRay(model, pixels[static_cast<std::size_t>(i)]);
      EXPECT_EQ(st[static_cast<std::size_t>(i)], single.status);
      if (single.status == StatusCode::OK)
      {
        EXPECT_NEAR(D(0, i), single.ray.direction.x(), 1e-4);
        EXPECT_NEAR(D(1, i), single.ray.direction.y(), 1e-4);
        EXPECT_NEAR(D(2, i), single.ray.direction.z(), 1e-4);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// LUT: step-1 grid query reproduces pixelToRay; float/double parity.
// ---------------------------------------------------------------------------
TEST(LutChar, GridQueryMatchesPixelToRayAndParity)
{
  const CameraModel model = makePinholeRadtan5();
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
  const CameraModel64 model64 = toCameraModel64(model);

  const int w = 64;
  const int h = 48;

  InverseLut lut;
  const int valid = lut.build(model, w, h, SolverOptions{}, 1);
  ASSERT_GT(valid, 0);
  ASSERT_TRUE(lut.isValid());

  InverseLut64 lut64;
  const int valid64 = lut64.build(model64, w, h, SolverOptions64{}, 1);
  ASSERT_GT(valid64, 0);

  // At integer grid coordinates (step 1) the bilinear query returns the stored
  // grid value, which is pixelToRay evaluated at that pixel.
  for (int v = 4; v < h; v += 11)
  {
    for (int u = 4; u < w; u += 13)
    {
      const RayResult q = lut.query(static_cast<float>(u), static_cast<float>(v));
      const RayResult direct =
        pixelToRay(model, Pixel2{static_cast<float>(u), static_cast<float>(v)});
      if (direct.status != StatusCode::OK)
      {
        continue;
      }
      ASSERT_EQ(q.status, StatusCode::OK) << "LUT query failed at (" << u << "," << v << ")";
      const float dot = q.ray.direction.normalized().dot(direct.ray.direction.normalized());
      EXPECT_NEAR(dot, 1.0f, 1e-4f) << "LUT vs pixelToRay at (" << u << "," << v << ")";

      // Float vs double LUT parity.
      const RayResult64 q64 = lut64.query(static_cast<double>(u), static_cast<double>(v));
      if (q64.status == StatusCode::OK)
      {
        const Eigen::Vector3d qf = q.ray.direction.cast<double>().normalized();
        const double dot64 = qf.dot(q64.ray.direction.normalized());
        EXPECT_NEAR(dot64, 1.0, 1e-4) << "float/double LUT parity at (" << u << "," << v << ")";
      }
    }
  }
}

TEST(LutChar, BuildRejectsOverflowingTotalGridSize)
{
  // 200000 x 200000 at step 1 passes the per-axis int guards (each axis is
  // ~200001) but the total grid would be ~4e10 cells / ~160 GB of resize():
  // build must follow its reject-with-0 contract, not die on an uncaught
  // bad_alloc.
  const CameraModel model = makePinholeRadtan5();
  ASSERT_EQ(validateCameraModel(model), StatusCode::OK);

  InverseLut lut;
  EXPECT_EQ(lut.build(model, 200000, 200000, SolverOptions{}, 1), 0);
  EXPECT_FALSE(lut.isValid());
}

TEST(LutChar, QueryRejectsExtremeCoordinatesCleanly)
{
  const CameraModel model = makePinholeRadtan5();
  InverseLut lut;
  ASSERT_GT(lut.build(model, 64, 48, SolverOptions{}, 1), 0);

  // Finite but far outside any int-representable grid index: must reject as
  // OUT_OF_FOV before the float->int cast, which would otherwise be UB.
  EXPECT_EQ(lut.query(3e18f, 24.0f).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(lut.query(-3e18f, 24.0f).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(lut.query(32.0f, 3e18f).status, StatusCode::OUT_OF_FOV);
  EXPECT_EQ(lut.query(32.0f, -3e18f).status, StatusCode::OUT_OF_FOV);
}

// ---------------------------------------------------------------------------
// Batched Jacobian: Eigen + raw-pointer variants vs per-point, plus parity.
// ---------------------------------------------------------------------------
TEST(JacobianBatchChar, MatchesPerPoint)
{
  const std::vector<CameraModel> models{makePinholeRadtan5(), makePinholeTilted14(),
                                        makeFisheyeKb4(),     makeOmnidirectional(),
                                        makeDoubleSphere(),   makeEucm()};
  const auto rays = sampleRays();
  const int n = static_cast<int>(rays.size());

  for (const auto &model : models)
  {
    ASSERT_EQ(validateCameraModel(model), StatusCode::OK);
    const CameraModel64 model64 = toCameraModel64(model);

    Eigen::Matrix3Xf R(3, n);
    Eigen::Matrix3Xd Rd(3, n);
    for (int i = 0; i < n; ++i)
    {
      R.col(i) = rays[i].cast<float>();
      Rd.col(i) = rays[i];
    }

    // Eigen variant (float).
    Eigen::Matrix2Xf P(2, n);
    std::vector<Eigen::Matrix<float, 2, 3>> J(static_cast<std::size_t>(n));
    std::vector<StatusCode> st(static_cast<std::size_t>(n));
    const int ok = rayToPixelWithJacobianBatch(model, R, P, J.data(), st.data());
    EXPECT_GE(ok, 0);

    // Raw-pointer variant (float).
    std::vector<float> rays_xyz(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i)
    {
      rays_xyz[static_cast<std::size_t>(3 * i + 0)] = R(0, i);
      rays_xyz[static_cast<std::size_t>(3 * i + 1)] = R(1, i);
      rays_xyz[static_cast<std::size_t>(3 * i + 2)] = R(2, i);
    }
    std::vector<float> u_out(static_cast<std::size_t>(n));
    std::vector<float> v_out(static_cast<std::size_t>(n));
    std::vector<float> jac_raw(static_cast<std::size_t>(6 * n));
    std::vector<StatusCode> st_raw(static_cast<std::size_t>(n));
    const int ok_raw = rayToPixelWithJacobianBatch(
      model, rays_xyz.data(), n, u_out.data(), v_out.data(), jac_raw.data(), st_raw.data()
    );
    EXPECT_EQ(ok_raw, ok);

    // Eigen variant (double) for parity.
    Eigen::Matrix2Xd Pd(2, n);
    std::vector<Eigen::Matrix<double, 2, 3>> Jd(static_cast<std::size_t>(n));
    std::vector<StatusCode> std64(static_cast<std::size_t>(n));
    const int ok_j64 = rayToPixelWithJacobianBatch64(model64, Rd, Pd, Jd.data(), std64.data());
    EXPECT_GE(ok_j64, 0);

    for (int i = 0; i < n; ++i)
    {
      const std::size_t ui = static_cast<std::size_t>(i);
      const ProjectionJacobian single = rayToPixelWithJacobian(model, R.col(i));
      EXPECT_EQ(st[ui], single.status);
      EXPECT_EQ(st_raw[ui], single.status);
      if (single.status != StatusCode::OK)
      {
        continue;
      }
      EXPECT_NEAR(P(0, i), single.pixel.u, 1e-4);
      EXPECT_NEAR(P(1, i), single.pixel.v, 1e-4);
      for (int r = 0; r < 2; ++r)
      {
        for (int c = 0; c < 3; ++c)
        {
          const float tol = 1e-4f * std::abs(single.J(r, c)) + 1e-4f;
          EXPECT_NEAR(J[ui](r, c), single.J(r, c), tol);
          EXPECT_NEAR(jac_raw[static_cast<std::size_t>(6 * i + r * 3 + c)], single.J(r, c), tol);
        }
      }
      const ProjectionJacobian64 single64 = rayToPixelWithJacobian64(model64, Rd.col(i));
      if (single64.status == StatusCode::OK)
      {
        for (int r = 0; r < 2; ++r)
        {
          for (int c = 0; c < 3; ++c)
          {
            EXPECT_NEAR(Jd[ui](r, c), single64.J(r, c), 1e-9 * std::abs(single64.J(r, c)) + 1e-9);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Full Jacobian (double): point Jacobian, xd/yd, and the distortion- and
// projection-parameter derivatives used by calibration, vs finite difference.
// ---------------------------------------------------------------------------
bool xydFromProjection(const CameraModel64 &m, const Eigen::Vector3d &ray, double &xd, double &yd)
{
  const PixelResult64 px = rayToPixel64(m, ray);
  if (px.status != StatusCode::OK)
  {
    return false;
  }
  yd = (px.pixel.v - m.intrinsics.cy) / m.intrinsics.fy;
  xd = (px.pixel.u - m.intrinsics.skew * yd - m.intrinsics.cx) / m.intrinsics.fx;
  return true;
}

double *projParamPtr(CameraModel64 &m, const int k)
{
  if (k == 0)
  {
    return &m.projection.xi;
  }
  if (k == 1)
  {
    return &m.projection.alpha;
  }
  return &m.projection.beta;
}

// Which dxd_dproj / dyd_dproj slots (0=xi, 1=alpha, 2=beta) a model populates.
std::vector<int> activeProjIndices(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::OMNIDIRECTIONAL:
      return {0};
    case ProjectionModelType::DOUBLE_SPHERE:
      return {0, 1};
    case ProjectionModelType::EUCM:
      return {1, 2};
    default:
      return {};
  }
}

void checkFullJacobian(const CameraModel &model_f)
{
  const CameraModel64 model = toCameraModel64(model_f);
  const double h = 1e-6;

  for (const auto &ray : sampleRays())
  {
    const PixelResult64 px = rayToPixel64(model, ray);
    if (px.status != StatusCode::OK)
    {
      continue;
    }

    const FullProjectionJacobian64 full = rayToPixelWithFullJacobian64(model, ray);
    ASSERT_EQ(full.status, StatusCode::OK) << toString(model.projection.type);

    EXPECT_NEAR(full.pixel.u, px.pixel.u, 1e-9);
    EXPECT_NEAR(full.pixel.v, px.pixel.v, 1e-9);

    double xd_ref = 0.0;
    double yd_ref = 0.0;
    ASSERT_TRUE(xydFromProjection(model, ray, xd_ref, yd_ref));
    EXPECT_NEAR(full.xd, xd_ref, 1e-9);
    EXPECT_NEAR(full.yd, yd_ref, 1e-9);

    // J_point must match the standalone point Jacobian (cross-checks the full
    // Jacobian's own local distortion-Jacobian against the shared core).
    const ProjectionJacobian64 pt = rayToPixelWithJacobian64(model, ray);
    ASSERT_EQ(pt.status, StatusCode::OK);
    for (int r = 0; r < 2; ++r)
    {
      for (int c = 0; c < 3; ++c)
      {
        EXPECT_NEAR(full.J_point(r, c), pt.J(r, c), 1e-6 * std::abs(pt.J(r, c)) + 1e-6)
          << "J_point (" << r << "," << c << "), " << toString(model.projection.type);
      }
    }

    // Distortion-coefficient derivatives vs central finite difference.
    for (int i = 0; i < full.dist_count; ++i)
    {
      CameraModel64 mp = model;
      CameraModel64 mm = model;
      mp.distortion.coeffs[static_cast<std::size_t>(i)] += h;
      mm.distortion.coeffs[static_cast<std::size_t>(i)] -= h;
      double xp = 0.0;
      double yp = 0.0;
      double xn = 0.0;
      double yn = 0.0;
      if (!xydFromProjection(mp, ray, xp, yp) || !xydFromProjection(mm, ray, xn, yn))
      {
        continue;
      }
      const double fdx = (xp - xn) / (2.0 * h);
      const double fdy = (yp - yn) / (2.0 * h);
      EXPECT_NEAR(full.dxd_ddist[static_cast<std::size_t>(i)], fdx, 1e-3 * std::abs(fdx) + 1e-5)
        << "dxd/ddist[" << i << "], " << toString(model.projection.type);
      EXPECT_NEAR(full.dyd_ddist[static_cast<std::size_t>(i)], fdy, 1e-3 * std::abs(fdy) + 1e-5)
        << "dyd/ddist[" << i << "], " << toString(model.projection.type);
    }

    // Projection-parameter derivatives (xi/alpha/beta) vs central finite difference.
    for (const int k : activeProjIndices(model.projection.type))
    {
      CameraModel64 mp = model;
      CameraModel64 mm = model;
      *projParamPtr(mp, k) += h;
      *projParamPtr(mm, k) -= h;
      double xp = 0.0;
      double yp = 0.0;
      double xn = 0.0;
      double yn = 0.0;
      if (!xydFromProjection(mp, ray, xp, yp) || !xydFromProjection(mm, ray, xn, yn))
      {
        continue;
      }
      const double fdx = (xp - xn) / (2.0 * h);
      const double fdy = (yp - yn) / (2.0 * h);
      EXPECT_NEAR(full.dxd_dproj[static_cast<std::size_t>(k)], fdx, 1e-3 * std::abs(fdx) + 1e-5)
        << "dxd/dproj[" << k << "], " << toString(model.projection.type);
      EXPECT_NEAR(full.dyd_dproj[static_cast<std::size_t>(k)], fdy, 1e-3 * std::abs(fdy) + 1e-5)
        << "dyd/dproj[" << k << "], " << toString(model.projection.type);
    }
  }
}

TEST(FullJacobianChar, PointJacobianXdYdAndParameterDerivatives)
{
  const std::vector<CameraModel> models{makePinholeRadtan5(), makePinholeTilted14(),
                                        makeFisheyeKb4(),     makeOmnidirectional(),
                                        makeDoubleSphere(),   makeEucm()};
  for (const auto &m : models)
  {
    ASSERT_EQ(validateCameraModel(m), StatusCode::OK);
    checkFullJacobian(m);
  }
}

}  // namespace
