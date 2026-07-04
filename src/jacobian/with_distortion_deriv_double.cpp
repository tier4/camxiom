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

// Full-parameter projection Jacobian, DOUBLE ONLY — and deliberately so (D1).
//
// rayToPixelWithFullJacobian64() returns, on top of the spatial 2x3
// d(u,v)/d(X,Y,Z), the *parameter* derivatives an optimiser needs: the
// distorted normalised point (xd,yd) for d(u,v)/d(fx,fy), the per-coefficient
// d(xd,yd)/d(dist[i]), and d(xd,yd)/d(xi,alpha,beta). Its ONLY consumers are the
// double-precision calibration / PnP optimiser paths:
//   - src/optimizer/pnp/pnp_gauss_newton.cpp   (Ceres-free normal equations)
//   - src/optimizer/pnp/pnp_cost_analytical_batch.cpp (ANALYTICAL Ceres cost)
//   - src/optimizer/pnp/pnp_solver.cpp
// Calibration/optimisation is inherently double precision (Ceres, normal
// equations); nothing in the float32 real-time path consumes a parameter
// Jacobian.
//
// Why this file is NOT scalar-templated like the rest (relation to #1):
// #1 single-sourced the float/double runtime pieces (projection, the SPATIAL
// Jacobian in src/jacobian/jacobian_impl.hpp, validation, ...) precisely because
// BOTH precisions are actually evaluated at runtime, so two hand-kept copies
// were a real drift hazard. This full-parameter Jacobian is evaluated in double
// ONLY, so a concrete double implementation is the correct scope: templatising
// it would add an uninstantiated, untested float instantiation with no caller —
// the opposite of #1's goal (a single source of truth for code that is really
// used twice). The asymmetry (this is the lone double-only jacobian TU) is
// intentional; do NOT add a float variant unless a genuine float consumer of
// the parameter Jacobian appears. See ARCHITECTURE.md §4.

#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"  // validateCameraModel64
#include "camxiom/projection64.hpp"
#include "detail/ds_forward.hpp"
#include "detail/projection_common.hpp"  // hasThetaMaxCap / withinThetaMax
#include "jacobian/full_jacobian64_internal.hpp"

#include <cmath>

namespace camxiom
{
namespace
{

inline constexpr double kEps64 = 1e-15;

inline bool isFinite2d(double x, double y) { return std::isfinite(x) && std::isfinite(y); }
inline bool isFinite3d(double x, double y, double z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

FullProjectionJacobian64 invalidFJ64(StatusCode s)
{
  FullProjectionJacobian64 r;
  r.status = s;
  return r;
}

bool isFiniteJacobian2x3d(const Eigen::Matrix<double, 2, 3> &J)
{
  for (int i = 0; i < 6; ++i)
    if (!std::isfinite(J.data()[i])) return false;
  return true;
}

FullProjectionJacobian64 finalizeFJ64(FullProjectionJacobian64 r)
{
  if (!isFiniteJacobian2x3d(r.J_point)) return invalidFJ64(StatusCode::NUMERIC_ERROR);
  return r;
}

// -----------------------------------------------------------------------
// Plane distortion with full Jacobian (spatial + coefficient)
// -----------------------------------------------------------------------
struct DistJ64
{
  double j00{1.0}, j01{0.0}, j10{0.0}, j11{1.0};
};

struct PlaneDistFullJ64
{
  DistJ64 jac_xy;
  std::array<double, 14> dxd_dc{};
  std::array<double, 14> dyd_dc{};
};

StatusCode distortPlaneNoTiltWithFullJ64(
  const DistortionModel64 &model, double x, double y, double &xd, double &yd, PlaneDistFullJ64 &out
)
{
  const double r2 = x * x + y * y;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const auto &c = model.coeffs;
  const double k1 = c[0], k2 = c[1], p1 = c[2], p2 = c[3], k3 = c[4];

  const double num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double dnum = 2.0 * (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4);

  double radial = num, drad_dx = dnum * x, drad_dy = dnum * y;

  double inv_den = 1.0;
  double inv_den2 = 1.0;
  double num_val = num;
  if (model.is_rational)
  {
    const double k4 = c[5], k5 = c[6], k6 = c[7];
    const double den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    if (std::abs(den) <= kEps64) return StatusCode::DOMAIN_ERROR;
    const double dden = 2.0 * (k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4);
    inv_den = 1.0 / den;
    inv_den2 = inv_den * inv_den;
    radial = num * inv_den;
    drad_dx = (dnum * x * den - num * dden * x) * inv_den2;
    drad_dy = (dnum * y * den - num * dden * y) * inv_den2;
    num_val = num;
  }

  const double x_tan = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_tan = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  double dtx_dx = 2.0 * p1 * y + 6.0 * p2 * x;
  double dtx_dy = 2.0 * p1 * x + 2.0 * p2 * y;
  double dty_dx = 2.0 * p1 * x + 2.0 * p2 * y;
  double dty_dy = 6.0 * p1 * y + 2.0 * p2 * x;

  xd = x * radial + x_tan;
  yd = y * radial + y_tan;

  // ∂(xd,yd)/∂dist coefficients
  out.dxd_dc.fill(0.0);
  out.dyd_dc.fill(0.0);

  if (!model.is_rational)
  {
    // Radtan5: c[0]=k1, c[1]=k2, c[2]=p1, c[3]=p2, c[4]=k3
    out.dxd_dc[0] = x * r2;
    out.dyd_dc[0] = y * r2;
    out.dxd_dc[1] = x * r4;
    out.dyd_dc[1] = y * r4;
    out.dxd_dc[2] = 2.0 * x * y;
    out.dyd_dc[2] = r2 + 2.0 * y * y;
    out.dxd_dc[3] = r2 + 2.0 * x * x;
    out.dyd_dc[3] = 2.0 * x * y;
    out.dxd_dc[4] = x * r6;
    out.dyd_dc[4] = y * r6;
  }
  else
  {
    // Rational8: c[0..4] same but scaled by 1/den, c[5..7] = k4,k5,k6
    out.dxd_dc[0] = x * r2 * inv_den;
    out.dyd_dc[0] = y * r2 * inv_den;
    out.dxd_dc[1] = x * r4 * inv_den;
    out.dyd_dc[1] = y * r4 * inv_den;
    out.dxd_dc[2] = 2.0 * x * y;
    out.dyd_dc[2] = r2 + 2.0 * y * y;
    out.dxd_dc[3] = r2 + 2.0 * x * x;
    out.dyd_dc[3] = 2.0 * x * y;
    out.dxd_dc[4] = x * r6 * inv_den;
    out.dyd_dc[4] = y * r6 * inv_den;
    const double neg_x_num = -x * num_val * inv_den2;
    const double neg_y_num = -y * num_val * inv_den2;
    out.dxd_dc[5] = neg_x_num * r2;
    out.dyd_dc[5] = neg_y_num * r2;
    out.dxd_dc[6] = neg_x_num * r4;
    out.dyd_dc[6] = neg_y_num * r4;
    out.dxd_dc[7] = neg_x_num * r6;
    out.dyd_dc[7] = neg_y_num * r6;
  }

  if (model.has_thin_prism)
  {
    const double s1 = c[8], s2 = c[9], s3 = c[10], s4 = c[11];
    xd += s1 * r2 + s2 * r4;
    yd += s3 * r2 + s4 * r4;
    dtx_dx += 2.0 * x * (s1 + 2.0 * s2 * r2);
    dtx_dy += 2.0 * y * (s1 + 2.0 * s2 * r2);
    dty_dx += 2.0 * x * (s3 + 2.0 * s4 * r2);
    dty_dy += 2.0 * y * (s3 + 2.0 * s4 * r2);
    out.dxd_dc[8] = r2;
    out.dyd_dc[8] = 0.0;
    out.dxd_dc[9] = r4;
    out.dyd_dc[9] = 0.0;
    out.dxd_dc[10] = 0.0;
    out.dyd_dc[10] = r2;
    out.dxd_dc[11] = 0.0;
    out.dyd_dc[11] = r4;
  }

  out.jac_xy.j00 = radial + x * drad_dx + dtx_dx;
  out.jac_xy.j01 = x * drad_dy + dtx_dy;
  out.jac_xy.j10 = y * drad_dx + dty_dx;
  out.jac_xy.j11 = radial + y * drad_dy + dty_dy;
  return StatusCode::OK;
}

StatusCode distortPlaneWithFullJ64(
  const DistortionModel64 &model, double xin, double yin, double &xout, double &yout,
  PlaneDistFullJ64 &out
)
{
  if (model.space == DistortionSpace::NONE || model.type == DistortionModelType::NONE)
  {
    xout = xin;
    yout = yin;
    out.jac_xy = {1.0, 0.0, 0.0, 1.0};
    out.dxd_dc.fill(0.0);
    out.dyd_dc.fill(0.0);
    return StatusCode::OK;
  }
  if (model.space != DistortionSpace::PLANE) return StatusCode::INVALID_MODEL;

  double xnt = 0.0, ynt = 0.0;
  PlaneDistFullJ64 j_no_tilt{};
  StatusCode s = distortPlaneNoTiltWithFullJ64(model, xin, yin, xnt, ynt, j_no_tilt);
  if (s != StatusCode::OK) return s;

  if (!model.has_tilt)
  {
    xout = xnt;
    yout = ynt;
    out = j_no_tilt;
    return StatusCode::OK;
  }

  const auto &T = model.tilt_matrix;
  const double mx = T[0] * xnt + T[1] * ynt + T[2];
  const double my = T[3] * xnt + T[4] * ynt + T[5];
  const double mz = T[6] * xnt + T[7] * ynt + T[8];
  if (std::abs(mz) <= kEps64) return StatusCode::DOMAIN_ERROR;
  const double inv_mz = 1.0 / mz;
  xout = mx * inv_mz;
  yout = my * inv_mz;

  const double dx_dxnt = (T[0] - xout * T[6]) * inv_mz;
  const double dx_dynt = (T[1] - xout * T[7]) * inv_mz;
  const double dy_dxnt = (T[3] - yout * T[6]) * inv_mz;
  const double dy_dynt = (T[4] - yout * T[7]) * inv_mz;

  // Chain spatial jacobian through tilt
  out.jac_xy.j00 = dx_dxnt * j_no_tilt.jac_xy.j00 + dx_dynt * j_no_tilt.jac_xy.j10;
  out.jac_xy.j01 = dx_dxnt * j_no_tilt.jac_xy.j01 + dx_dynt * j_no_tilt.jac_xy.j11;
  out.jac_xy.j10 = dy_dxnt * j_no_tilt.jac_xy.j00 + dy_dynt * j_no_tilt.jac_xy.j10;
  out.jac_xy.j11 = dy_dxnt * j_no_tilt.jac_xy.j01 + dy_dynt * j_no_tilt.jac_xy.j11;

  // Chain dist coefficient jacobian through tilt
  const int nc = static_cast<int>(model.count);
  for (int i = 0; i < nc; ++i)
  {
    out.dxd_dc[static_cast<std::size_t>(i)] =
      dx_dxnt * j_no_tilt.dxd_dc[static_cast<std::size_t>(i)] +
      dx_dynt * j_no_tilt.dyd_dc[static_cast<std::size_t>(i)];
    out.dyd_dc[static_cast<std::size_t>(i)] =
      dy_dxnt * j_no_tilt.dxd_dc[static_cast<std::size_t>(i)] +
      dy_dynt * j_no_tilt.dyd_dc[static_cast<std::size_t>(i)];
  }
  for (int i = nc; i < 14; ++i)
  {
    out.dxd_dc[static_cast<std::size_t>(i)] = 0.0;
    out.dyd_dc[static_cast<std::size_t>(i)] = 0.0;
  }
  return StatusCode::OK;
}

// -----------------------------------------------------------------------
// Apply intrinsics to compose J_point and populate full result
// -----------------------------------------------------------------------
void applyIntrinsicsToFull(
  FullProjectionJacobian64 &r, const IntrinsicsModel64 &intr, double xd, double yd,
  const PlaneDistFullJ64 &dj, double p00, double p01, double p02, double p10, double p11, double p12
)
{
  const double fx = intr.fx, fy = intr.fy, sk = intr.skew;

  r.xd = xd;
  r.yd = yd;
  r.pixel.u = fx * xd + sk * yd + intr.cx;
  r.pixel.v = fy * yd + intr.cy;

  // intrinsics × distortion_spatial → id (2×2)
  const double id00 = fx * dj.jac_xy.j00 + sk * dj.jac_xy.j10;
  const double id01 = fx * dj.jac_xy.j01 + sk * dj.jac_xy.j11;
  const double id10 = fy * dj.jac_xy.j10;
  const double id11 = fy * dj.jac_xy.j11;

  // J_point = id × projection_jacobian
  r.J_point(0, 0) = id00 * p00 + id01 * p10;
  r.J_point(0, 1) = id00 * p01 + id01 * p11;
  r.J_point(0, 2) = id00 * p02 + id01 * p12;
  r.J_point(1, 0) = id10 * p00 + id11 * p10;
  r.J_point(1, 1) = id10 * p01 + id11 * p11;
  r.J_point(1, 2) = id10 * p02 + id11 * p12;
}

}  // namespace

// -----------------------------------------------------------------------
// Pinhole full Jacobian
// -----------------------------------------------------------------------
namespace pinhole
{

static FullProjectionJacobian64 fullJ64(const CameraModel64 &model, const Eigen::Vector3d &ray)
{
  if (model.projection.type != ProjectionModelType::PINHOLE)
    return invalidFJ64(StatusCode::INVALID_MODEL);
  if (!isFinite3d(ray.x(), ray.y(), ray.z())) return invalidFJ64(StatusCode::INVALID_INPUT);
  if (ray.z() <= 0.0) return invalidFJ64(StatusCode::BEHIND_CAMERA);

  const double X = ray.x(), Y = ray.y(), Z = ray.z();
  const double inv_z = 1.0 / Z, inv_z2 = inv_z * inv_z;
  const double x_n = X * inv_z, y_n = Y * inv_z;

  // ∂(xn,yn)/∂(X,Y,Z)
  const double p00 = inv_z, p02 = -X * inv_z2;
  const double p11 = inv_z, p12 = -Y * inv_z2;

  double xd = 0.0, yd = 0.0;
  PlaneDistFullJ64 dj{};
  StatusCode ds = distortPlaneWithFullJ64(model.distortion, x_n, y_n, xd, yd, dj);
  if (ds != StatusCode::OK) return invalidFJ64(ds);

  FullProjectionJacobian64 r;
  r.status = StatusCode::OK;
  r.dist_count = static_cast<int>(model.distortion.count);
  r.dxd_ddist = dj.dxd_dc;
  r.dyd_ddist = dj.dyd_dc;
  applyIntrinsicsToFull(r, model.intrinsics, xd, yd, dj, p00, 0.0, p02, 0.0, p11, p12);
  if (!isFinite2d(r.pixel.u, r.pixel.v)) return invalidFJ64(StatusCode::NUMERIC_ERROR);
  return finalizeFJ64(r);
}

}  // namespace pinhole

// -----------------------------------------------------------------------
// Fisheye full Jacobian
// -----------------------------------------------------------------------
namespace fisheye
{

static FullProjectionJacobian64 fullJ64(const CameraModel64 &model, const Eigen::Vector3d &ray)
{
  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
    return invalidFJ64(StatusCode::INVALID_MODEL);
  if (!isFinite3d(ray.x(), ray.y(), ray.z())) return invalidFJ64(StatusCode::INVALID_INPUT);

  const double X = ray.x(), Y = ray.y(), Z = ray.z();
  const double r_xy = std::sqrt(X * X + Y * Y);
  const double R2 = X * X + Y * Y + Z * Z;
  if (!std::isfinite(R2) || R2 <= kEps64) return invalidFJ64(StatusCode::INVALID_INPUT);
  const double theta = std::atan2(r_xy, Z);
  if (theta < 0.0) return invalidFJ64(StatusCode::DOMAIN_ERROR);
  if (theta > model.projection.theta_max) return invalidFJ64(StatusCode::OUT_OF_FOV);
  if (r_xy <= kEps64 && Z < 0.0) return invalidFJ64(StatusCode::DOMAIN_ERROR);

  double theta_d = 0.0;
  double dtheta_d = 0.0;
  std::array<double, 14> dthetad_dc{};
  const int dist_count = static_cast<int>(model.distortion.count);

  {
    const auto &dm = model.distortion;
    const auto &c = dm.coeffs;
    const double t2 = theta * theta, t4 = t2 * t2, t6 = t4 * t2, t8 = t4 * t4;
    switch (dm.type)
    {
      case DistortionModelType::NONE:
      case DistortionModelType::EQUIDISTANT:
        theta_d = theta;
        dtheta_d = 1.0;
        break;
      case DistortionModelType::OPENCV_FISHEYE4: {
        theta_d = theta * (1.0 + c[0] * t2 + c[1] * t4 + c[2] * t6 + c[3] * t8);
        dtheta_d = 1.0 + 3.0 * c[0] * t2 + 5.0 * c[1] * t4 + 7.0 * c[2] * t6 + 9.0 * c[3] * t8;
        // ∂theta_d/∂c[i] = theta^(2i+3)
        dthetad_dc[0] = theta * t2;
        dthetad_dc[1] = theta * t4;
        dthetad_dc[2] = theta * t6;
        dthetad_dc[3] = theta * t8;
        break;
      }
      case DistortionModelType::KB4: {
        theta_d =
          theta + c[0] * theta * t2 + c[1] * theta * t4 + c[2] * theta * t6 + c[3] * theta * t8;
        dtheta_d = 1.0 + 3.0 * c[0] * t2 + 5.0 * c[1] * t4 + 7.0 * c[2] * t6 + 9.0 * c[3] * t8;
        dthetad_dc[0] = theta * t2;
        dthetad_dc[1] = theta * t4;
        dthetad_dc[2] = theta * t6;
        dthetad_dc[3] = theta * t8;
        break;
      }
      case DistortionModelType::KB8: {
        const double t10 = t8 * t2, t12 = t8 * t4, t14 = t8 * t6, t16 = t8 * t8;
        theta_d = theta + c[0] * theta * t2 + c[1] * theta * t4 + c[2] * theta * t6 +
                  c[3] * theta * t8 + c[4] * theta * t10 + c[5] * theta * t12 + c[6] * theta * t14 +
                  c[7] * theta * t16;
        dtheta_d = 1.0 + 3.0 * c[0] * t2 + 5.0 * c[1] * t4 + 7.0 * c[2] * t6 + 9.0 * c[3] * t8 +
                   11.0 * c[4] * t10 + 13.0 * c[5] * t12 + 15.0 * c[6] * t14 + 17.0 * c[7] * t16;
        dthetad_dc[0] = theta * t2;
        dthetad_dc[1] = theta * t4;
        dthetad_dc[2] = theta * t6;
        dthetad_dc[3] = theta * t8;
        dthetad_dc[4] = theta * t10;
        dthetad_dc[5] = theta * t12;
        dthetad_dc[6] = theta * t14;
        dthetad_dc[7] = theta * t16;
        break;
      }
      case DistortionModelType::EQUISOLID:
        theta_d = 2.0 * std::sin(theta * 0.5);
        dtheta_d = std::cos(theta * 0.5);
        break;
      case DistortionModelType::STEREOGRAPHIC: {
        theta_d = 2.0 * std::tan(theta * 0.5);
        const double c2 = std::cos(theta * 0.5);
        dtheta_d = 1.0 / (c2 * c2);
        break;
      }
      case DistortionModelType::ORTHOGRAPHIC:
        theta_d = std::sin(theta);
        dtheta_d = std::cos(theta);
        break;
      case DistortionModelType::RADTAN4:
      case DistortionModelType::RADTAN5:
      case DistortionModelType::RATIONAL8:
      case DistortionModelType::THIN_PRISM12:
      case DistortionModelType::TILTED14:
      case DistortionModelType::OMNIDIRECTIONAL:
      case DistortionModelType::UNKNOWN:
        return invalidFJ64(StatusCode::INVALID_MODEL);
    }
  }

  const double fx = model.intrinsics.fx, fy = model.intrinsics.fy, sk = model.intrinsics.skew;

  if (r_xy <= kEps64)
  {
    FullProjectionJacobian64 r;
    r.status = StatusCode::OK;
    r.dist_count = dist_count;
    r.xd = 0.0;
    r.yd = 0.0;
    r.pixel.u = model.intrinsics.cx;
    r.pixel.v = model.intrinsics.cy;
    r.J_point(0, 0) = fx * dtheta_d / Z;
    r.J_point(0, 1) = sk * dtheta_d / Z;
    r.J_point(1, 1) = fy * dtheta_d / Z;
    // dist jacobian: all zero at optical axis
    return finalizeFJ64(r);
  }

  const double s = theta_d / r_xy;
  const double xd = X * s, yd_val = Y * s;
  const double inv_R2 = 1.0 / R2;
  const double inv_rxy2 = 1.0 / (r_xy * r_xy);
  const double A = (dtheta_d * Z * inv_R2 - theta_d / r_xy) * inv_rxy2;

  const double q00 = s + X * X * A;
  const double q01 = X * Y * A;
  const double q02 = -X * dtheta_d * inv_R2;
  const double q10 = X * Y * A;
  const double q11 = s + Y * Y * A;
  const double q12 = -Y * dtheta_d * inv_R2;

  FullProjectionJacobian64 r;
  r.status = StatusCode::OK;
  r.dist_count = dist_count;
  r.xd = xd;
  r.yd = yd_val;
  r.pixel.u = fx * xd + sk * yd_val + model.intrinsics.cx;
  r.pixel.v = fy * yd_val + model.intrinsics.cy;
  if (!isFinite2d(r.pixel.u, r.pixel.v)) return invalidFJ64(StatusCode::NUMERIC_ERROR);

  // J_point
  r.J_point(0, 0) = fx * q00 + sk * q10;
  r.J_point(0, 1) = fx * q01 + sk * q11;
  r.J_point(0, 2) = fx * q02 + sk * q12;
  r.J_point(1, 0) = fy * q10;
  r.J_point(1, 1) = fy * q11;
  r.J_point(1, 2) = fy * q12;

  // ∂(xd,yd)/∂dist[i] = (X/r_xy, Y/r_xy) × ∂theta_d/∂dist[i]
  const double inv_rxy = 1.0 / r_xy;
  const double x_over_rxy = X * inv_rxy;
  const double y_over_rxy = Y * inv_rxy;
  for (int i = 0; i < dist_count; ++i)
  {
    r.dxd_ddist[static_cast<std::size_t>(i)] = x_over_rxy * dthetad_dc[static_cast<std::size_t>(i)];
    r.dyd_ddist[static_cast<std::size_t>(i)] = y_over_rxy * dthetad_dc[static_cast<std::size_t>(i)];
  }
  return finalizeFJ64(r);
}

}  // namespace fisheye

// -----------------------------------------------------------------------
// Omnidirectional full Jacobian
// -----------------------------------------------------------------------
namespace omnidirectional
{

static FullProjectionJacobian64 fullJ64(const CameraModel64 &model, const Eigen::Vector3d &ray)
{
  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
    return invalidFJ64(StatusCode::INVALID_MODEL);
  if (!isFinite3d(ray.x(), ray.y(), ray.z())) return invalidFJ64(StatusCode::INVALID_INPUT);

  const double X = ray.x(), Y = ray.y(), Z = ray.z();
  const double xi = model.projection.xi;
  const double rv = std::sqrt(X * X + Y * Y + Z * Z);
  if (!std::isfinite(rv) || rv <= kEps64) return invalidFJ64(StatusCode::INVALID_INPUT);

  const double denom = Z + xi * rv;
  if (!std::isfinite(denom) || denom <= kEps64) return invalidFJ64(StatusCode::OUT_OF_FOV);
  // Injectivity limit (see the omnidirectional projection impl): binding for xi > 1.
  if (rv + xi * Z <= kEps64) return invalidFJ64(StatusCode::OUT_OF_FOV);
  // theta_max contract, matching the scalar forward.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z, rv))
  {
    return invalidFJ64(StatusCode::OUT_OF_FOV);
  }
  const double inv_d = 1.0 / denom;
  const double x_n = X * inv_d, y_n = Y * inv_d;

  const double inv_r = 1.0 / rv;
  const double dd_dX = xi * X * inv_r;
  const double dd_dY = xi * Y * inv_r;
  const double dd_dZ = 1.0 + xi * Z * inv_r;
  const double inv_d2 = inv_d * inv_d;

  const double p00 = (denom - X * dd_dX) * inv_d2;
  const double p01 = (-X * dd_dY) * inv_d2;
  const double p02 = (-X * dd_dZ) * inv_d2;
  const double p10 = (-Y * dd_dX) * inv_d2;
  const double p11 = (denom - Y * dd_dY) * inv_d2;
  const double p12 = (-Y * dd_dZ) * inv_d2;

  double xd = 0.0, yd = 0.0;
  PlaneDistFullJ64 dj{};
  StatusCode ds = distortPlaneWithFullJ64(model.distortion, x_n, y_n, xd, yd, dj);
  if (ds != StatusCode::OK) return invalidFJ64(ds);

  // ∂(xn,yn)/∂xi: xn = X*inv_d, denom = Z + xi*rv
  // ∂denom/∂xi = rv  →  ∂xn/∂xi = -xn*rv*inv_d,  ∂yn/∂xi = -yn*rv*inv_d
  const double dxn_dxi = -x_n * rv * inv_d;
  const double dyn_dxi = -y_n * rv * inv_d;

  FullProjectionJacobian64 r;
  r.status = StatusCode::OK;
  r.dist_count = static_cast<int>(model.distortion.count);
  r.dxd_ddist = dj.dxd_dc;
  r.dyd_ddist = dj.dyd_dc;
  r.proj_param_count = 1;
  r.dxd_dproj[0] = dj.jac_xy.j00 * dxn_dxi + dj.jac_xy.j01 * dyn_dxi;
  r.dyd_dproj[0] = dj.jac_xy.j10 * dxn_dxi + dj.jac_xy.j11 * dyn_dxi;
  applyIntrinsicsToFull(r, model.intrinsics, xd, yd, dj, p00, p01, p02, p10, p11, p12);
  if (!isFinite2d(r.pixel.u, r.pixel.v)) return invalidFJ64(StatusCode::NUMERIC_ERROR);
  return finalizeFJ64(r);
}

}  // namespace omnidirectional

// -----------------------------------------------------------------------
// Double Sphere full Jacobian
// -----------------------------------------------------------------------
namespace double_sphere
{

static FullProjectionJacobian64 fullJ64(const CameraModel64 &model, const Eigen::Vector3d &ray)
{
  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
    return invalidFJ64(StatusCode::INVALID_MODEL);
  if (!isFinite3d(ray.x(), ray.y(), ray.z())) return invalidFJ64(StatusCode::INVALID_INPUT);

  const double X = ray.x(), Y = ray.y(), Z = ray.z();
  const double xi = model.projection.xi;
  const double alpha = model.projection.alpha;

  double d1 = 0.0, r_sq = 0.0, xi_d1_z = 0.0, d2 = 0.0, denom = 0.0;
  const StatusCode fwd =
    detail::computeDsForward(xi, alpha, X, Y, Z, kEps64, d1, r_sq, xi_d1_z, d2, denom);
  if (fwd != StatusCode::OK) return invalidFJ64(fwd);
  // theta_max contract, matching the scalar forward.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z, d1))
  {
    return invalidFJ64(StatusCode::OUT_OF_FOV);
  }
  const double inv_d1 = 1.0 / d1;
  const double inv_d2 = 1.0 / d2;
  const double inv_denom = 1.0 / denom;
  const double inv_denom2 = inv_denom * inv_denom;

  auto ddenom_dV = [&](double V) -> double {
    const double dxi_d1_z_dV = xi * V * inv_d1;
    const double dd2_dV = (V + xi_d1_z * dxi_d1_z_dV) * inv_d2;
    return alpha * dd2_dV + (1.0 - alpha) * dxi_d1_z_dV;
  };
  const double dxi_d1_z_dZ = xi * Z * inv_d1 + 1.0;
  const double dd2_dZ = (xi_d1_z * dxi_d1_z_dZ) * inv_d2;
  const double ddenom_dZ = alpha * dd2_dZ + (1.0 - alpha) * dxi_d1_z_dZ;
  const double ddenom_dX = ddenom_dV(X);
  const double ddenom_dY = ddenom_dV(Y);

  const double p00 = (denom - X * ddenom_dX) * inv_denom2;
  const double p01 = (-X * ddenom_dY) * inv_denom2;
  const double p02 = (-X * ddenom_dZ) * inv_denom2;
  const double p10 = (-Y * ddenom_dX) * inv_denom2;
  const double p11 = (denom - Y * ddenom_dY) * inv_denom2;
  const double p12 = (-Y * ddenom_dZ) * inv_denom2;

  const double x_n = X * inv_denom;
  const double y_n = Y * inv_denom;

  double xd = 0.0, yd = 0.0;
  PlaneDistFullJ64 dj{};
  StatusCode ds = distortPlaneWithFullJ64(model.distortion, x_n, y_n, xd, yd, dj);
  if (ds != StatusCode::OK) return invalidFJ64(ds);

  // ∂denom/∂xi = d1 * (alpha * xi_d1_z * inv_d2 + (1 - alpha))
  const double ddenom_dxi = d1 * (alpha * xi_d1_z * inv_d2 + (1.0 - alpha));
  const double dxn_dxi = -x_n * ddenom_dxi * inv_denom;
  const double dyn_dxi = -y_n * ddenom_dxi * inv_denom;

  // ∂denom/∂alpha = d2 - xi_d1_z
  const double ddenom_dalpha = d2 - xi_d1_z;
  const double dxn_dalpha = -x_n * ddenom_dalpha * inv_denom;
  const double dyn_dalpha = -y_n * ddenom_dalpha * inv_denom;

  FullProjectionJacobian64 r;
  r.status = StatusCode::OK;
  r.dist_count = static_cast<int>(model.distortion.count);
  r.dxd_ddist = dj.dxd_dc;
  r.dyd_ddist = dj.dyd_dc;
  r.proj_param_count = 2;
  r.dxd_dproj[0] = dj.jac_xy.j00 * dxn_dxi + dj.jac_xy.j01 * dyn_dxi;
  r.dyd_dproj[0] = dj.jac_xy.j10 * dxn_dxi + dj.jac_xy.j11 * dyn_dxi;
  r.dxd_dproj[1] = dj.jac_xy.j00 * dxn_dalpha + dj.jac_xy.j01 * dyn_dalpha;
  r.dyd_dproj[1] = dj.jac_xy.j10 * dxn_dalpha + dj.jac_xy.j11 * dyn_dalpha;
  applyIntrinsicsToFull(r, model.intrinsics, xd, yd, dj, p00, p01, p02, p10, p11, p12);
  if (!isFinite2d(r.pixel.u, r.pixel.v)) return invalidFJ64(StatusCode::NUMERIC_ERROR);
  return finalizeFJ64(r);
}

}  // namespace double_sphere

// -----------------------------------------------------------------------
// EUCM full Jacobian
// -----------------------------------------------------------------------
namespace eucm
{

static FullProjectionJacobian64 fullJ64(const CameraModel64 &model, const Eigen::Vector3d &ray)
{
  if (model.projection.type != ProjectionModelType::EUCM)
    return invalidFJ64(StatusCode::INVALID_MODEL);
  if (!isFinite3d(ray.x(), ray.y(), ray.z())) return invalidFJ64(StatusCode::INVALID_INPUT);

  const double X = ray.x(), Y = ray.y(), Z = ray.z();
  const double alpha = model.projection.alpha;
  const double beta = model.projection.beta;

  const double d = std::sqrt(beta * (X * X + Y * Y) + Z * Z);
  if (!std::isfinite(d) || d <= kEps64) return invalidFJ64(StatusCode::INVALID_INPUT);
  const double inv_d = 1.0 / d;

  const double denom = alpha * d + (1.0 - alpha) * Z;
  if (!std::isfinite(denom) || std::abs(denom) <= kEps64)
    return invalidFJ64(StatusCode::DOMAIN_ERROR);

  // Unconditional w-check (at alpha = 0 it degenerates to z > 0), matching
  // the scalar forward; the old alpha gate let alpha ~ 0 mirror rear points.
  {
    const double w = (alpha <= 0.5) ? (alpha / (1.0 - alpha)) : ((1.0 - alpha) / alpha);
    if (Z <= -w * d) return invalidFJ64(StatusCode::OUT_OF_FOV);
  }
  // theta_max contract; d above is beta-weighted, so use the true norm.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z,
                                   std::sqrt(X * X + Y * Y + Z * Z)))
  {
    return invalidFJ64(StatusCode::OUT_OF_FOV);
  }

  const double inv_denom = 1.0 / denom;
  const double inv_denom2 = inv_denom * inv_denom;

  const double dd_dX = alpha * beta * X * inv_d;
  const double dd_dY = alpha * beta * Y * inv_d;
  const double dd_dZ = alpha * Z * inv_d + (1.0 - alpha);

  const double p00 = (denom - X * dd_dX) * inv_denom2;
  const double p01 = (-X * dd_dY) * inv_denom2;
  const double p02 = (-X * dd_dZ) * inv_denom2;
  const double p10 = (-Y * dd_dX) * inv_denom2;
  const double p11 = (denom - Y * dd_dY) * inv_denom2;
  const double p12 = (-Y * dd_dZ) * inv_denom2;

  const double x_n = X * inv_denom;
  const double y_n = Y * inv_denom;

  double xd = 0.0, yd = 0.0;
  PlaneDistFullJ64 dj{};
  StatusCode ds = distortPlaneWithFullJ64(model.distortion, x_n, y_n, xd, yd, dj);
  if (ds != StatusCode::OK) return invalidFJ64(ds);

  // ∂denom/∂alpha = d - Z
  const double ddenom_dalpha = d - Z;
  const double dxn_dalpha = -x_n * ddenom_dalpha * inv_denom;
  const double dyn_dalpha = -y_n * ddenom_dalpha * inv_denom;

  // ∂d/∂beta = (X²+Y²) / (2*d)
  // ∂denom/∂beta = alpha * (X²+Y²) / (2*d)
  const double r2_xy = X * X + Y * Y;
  const double ddenom_dbeta = alpha * r2_xy * 0.5 * inv_d;
  const double dxn_dbeta = -x_n * ddenom_dbeta * inv_denom;
  const double dyn_dbeta = -y_n * ddenom_dbeta * inv_denom;

  FullProjectionJacobian64 r;
  r.status = StatusCode::OK;
  r.dist_count = static_cast<int>(model.distortion.count);
  r.dxd_ddist = dj.dxd_dc;
  r.dyd_ddist = dj.dyd_dc;
  r.proj_param_count = 2;
  // EUCM: proj_params = [xi(unused), alpha, beta] → slots 1, 2
  r.dxd_dproj[1] = dj.jac_xy.j00 * dxn_dalpha + dj.jac_xy.j01 * dyn_dalpha;
  r.dyd_dproj[1] = dj.jac_xy.j10 * dxn_dalpha + dj.jac_xy.j11 * dyn_dalpha;
  r.dxd_dproj[2] = dj.jac_xy.j00 * dxn_dbeta + dj.jac_xy.j01 * dyn_dbeta;
  r.dyd_dproj[2] = dj.jac_xy.j10 * dxn_dbeta + dj.jac_xy.j11 * dyn_dbeta;
  applyIntrinsicsToFull(r, model.intrinsics, xd, yd, dj, p00, p01, p02, p10, p11, p12);
  if (!isFinite2d(r.pixel.u, r.pixel.v)) return invalidFJ64(StatusCode::NUMERIC_ERROR);
  return finalizeFJ64(r);
}

}  // namespace eucm

// -----------------------------------------------------------------------
// Dispatch
// -----------------------------------------------------------------------

namespace detail
{
FullProjectionJacobian64 rayToPixelWithFullJacobian64Unchecked(
  const CameraModel64 &model, const Eigen::Vector3d &ray
)
{
  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::fullJ64(model, ray);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::fullJ64(model, ray);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::fullJ64(model, ray);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::fullJ64(model, ray);
    case ProjectionModelType::EUCM:
      return eucm::fullJ64(model, ray);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return invalidFJ64(StatusCode::INVALID_MODEL);
}
}  // namespace detail

FullProjectionJacobian64 rayToPixelWithFullJacobian64(
  const CameraModel64 &model, const Eigen::Vector3d &ray
)
{
  const StatusCode v = validateCameraModel64(model);
  if (v != StatusCode::OK) return invalidFJ64(v);
  return detail::rayToPixelWithFullJacobian64Unchecked(model, ray);
}

}  // namespace camxiom
