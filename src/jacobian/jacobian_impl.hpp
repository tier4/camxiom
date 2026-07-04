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

#ifndef CAMXIOM__JACOBIAN__JACOBIAN_IMPL_HPP
#define CAMXIOM__JACOBIAN__JACOBIAN_IMPL_HPP

// Scalar-templated projection-Jacobian core (#1 step 5c). Single source of truth
// for the analytic 2x3 Jacobians previously hand-duplicated between the float
// implementation (src/jacobian/float.cpp) and the double implementation
// (src/jacobian/double.cpp). The float side already delegated the distortion
// Jacobian to the shared detail_impl core, but the double side carried a local
// re-implementation (distortPlaneWithJ64 + an inlined distortTheta/derivative
// switch); routing both precisions through this template removes that drift
// hazard.
//
// Everything is expressed via the shared, already-templated cores:
//   - detail_impl::distortPlaneModelWithJacobian<T> (plane_impl.hpp),
//   - detail_impl::distortTheta<T> / distortThetaDerivative<T> (angle_impl.hpp),
//   - detail::computeDsForward<T> (ds_forward.hpp),
// with PlaneTraits<T>::kEpsilon for the per-precision thresholds.

#include "camxiom/jacobian.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "detail/ds_forward.hpp"
#include "detail/internal.hpp"           // validateCameraModelQuery
#include "detail/projection_common.hpp"  // hasThetaMaxCap / withinThetaMax
#include "distortion/angle_impl.hpp"
#include "distortion/plane_impl.hpp"
#include "projection64/internal.hpp"  // validateCameraModelQuery64

#include <Eigen/Core>

#include <cmath>
#include <type_traits>

namespace camxiom::detail_impl
{

template <typename T>
inline ProjectionJacobianT<T> invalidJacobian(const StatusCode status)
{
  ProjectionJacobianT<T> result;
  result.status = status;
  return result;
}

template <typename T>
inline bool isFiniteJacobian2x3(const Eigen::Matrix<T, 2, 3> &J)
{
  return std::isfinite(J(0, 0)) && std::isfinite(J(0, 1)) && std::isfinite(J(0, 2)) &&
         std::isfinite(J(1, 0)) && std::isfinite(J(1, 1)) && std::isfinite(J(1, 2));
}

template <typename T>
inline ProjectionJacobianT<T> finalizeJacobian(ProjectionJacobianT<T> result)
{
  if (!isFiniteJacobian2x3<T>(result.J))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }
  return result;
}

}  // namespace camxiom::detail_impl

namespace camxiom::pinhole::impl
{

template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobian(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidJacobian;

  if (model.projection.type != ProjectionModelType::PINHOLE)
  {
    return invalidJacobian<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }
  if (ray_direction.z() <= T(0))
  {
    return invalidJacobian<T>(StatusCode::BEHIND_CAMERA);
  }

  const T X = ray_direction.x();
  const T Y = ray_direction.y();
  const T Z = ray_direction.z();
  const T inv_z = T(1) / Z;
  const T inv_z2 = inv_z * inv_z;

  const T x_n = X * inv_z;
  const T y_n = Y * inv_z;

  const T p00 = inv_z;
  const T p02 = -X * inv_z2;
  const T p11 = inv_z;
  const T p12 = -Y * inv_z2;

  detail_impl::NormPointT<T> undistorted_xy{x_n, y_n};
  detail_impl::NormPointT<T> distorted_xy{};
  T j00 = T(1);
  T j01 = T(0);
  T j10 = T(0);
  T j11 = T(1);
  const StatusCode dist_status = detail_impl::distortPlaneModelWithJacobian<T>(
    model.distortion, undistorted_xy, distorted_xy, j00, j01, j10, j11
  );
  if (dist_status != StatusCode::OK)
  {
    return invalidJacobian<T>(dist_status);
  }

  const T fx = model.intrinsics.fx;
  const T fy = model.intrinsics.fy;
  const T skew = model.intrinsics.skew;

  ProjectionJacobianT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = fx * distorted_xy.x + skew * distorted_xy.y + model.intrinsics.cx;
  result.pixel.v = fy * distorted_xy.y + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }

  const T id00 = fx * j00 + skew * j10;
  const T id01 = fx * j01 + skew * j11;
  const T id10 = fy * j10;
  const T id11 = fy * j11;

  result.J(0, 0) = id00 * p00;
  result.J(0, 1) = id01 * p11;
  result.J(0, 2) = id00 * p02 + id01 * p12;
  result.J(1, 0) = id10 * p00;
  result.J(1, 1) = id11 * p11;
  result.J(1, 2) = id10 * p02 + id11 * p12;

  return detail_impl::finalizeJacobian<T>(result);
}

}  // namespace camxiom::pinhole::impl

namespace camxiom::fisheye::impl
{

template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobian(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidJacobian;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    return invalidJacobian<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }

  const T X = ray_direction.x();
  const T Y = ray_direction.y();
  const T Z = ray_direction.z();

  const T r_xy = std::sqrt(X * X + Y * Y);
  const T R2 = X * X + Y * Y + Z * Z;
  if (!std::isfinite(R2) || R2 <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }
  const T theta = std::atan2(r_xy, Z);

  if (theta < T(0))
  {
    return invalidJacobian<T>(StatusCode::DOMAIN_ERROR);
  }
  if (theta > model.projection.theta_max)
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }
  if (r_xy <= kEpsilon && Z < T(0))
  {
    return invalidJacobian<T>(StatusCode::DOMAIN_ERROR);
  }

  T theta_d = T(0);
  const StatusCode td_status = detail_impl::distortTheta<T>(model.distortion, theta, theta_d);
  if (td_status != StatusCode::OK)
  {
    return invalidJacobian<T>(td_status);
  }

  T dtheta_d_dtheta = T(0);
  const StatusCode dd_status =
    detail_impl::distortThetaDerivative<T>(model.distortion, theta, dtheta_d_dtheta);
  if (dd_status != StatusCode::OK)
  {
    return invalidJacobian<T>(dd_status);
  }

  const T fx = model.intrinsics.fx;
  const T fy = model.intrinsics.fy;
  const T skew = model.intrinsics.skew;

  if (r_xy <= kEpsilon)
  {
    ProjectionJacobianT<T> result;
    result.status = StatusCode::OK;
    result.pixel.u = model.intrinsics.cx;
    result.pixel.v = model.intrinsics.cy;
    result.J(0, 0) = fx * dtheta_d_dtheta / Z;
    result.J(0, 1) = skew * dtheta_d_dtheta / Z;
    result.J(0, 2) = T(0);
    result.J(1, 0) = T(0);
    result.J(1, 1) = fy * dtheta_d_dtheta / Z;
    result.J(1, 2) = T(0);
    return detail_impl::finalizeJacobian<T>(result);
  }

  const T s = theta_d / r_xy;
  const T x_d = X * s;
  const T y_d = Y * s;

  const T inv_R2 = T(1) / R2;
  const T inv_rxy2 = T(1) / (r_xy * r_xy);
  const T A = (dtheta_d_dtheta * Z * inv_R2 - theta_d / r_xy) * inv_rxy2;

  const T p00 = s + X * X * A;
  const T p01 = X * Y * A;
  const T p02 = -X * dtheta_d_dtheta * inv_R2;
  const T p10 = X * Y * A;
  const T p11 = s + Y * Y * A;
  const T p12 = -Y * dtheta_d_dtheta * inv_R2;

  ProjectionJacobianT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = fx * x_d + skew * y_d + model.intrinsics.cx;
  result.pixel.v = fy * y_d + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }

  result.J(0, 0) = fx * p00 + skew * p10;
  result.J(0, 1) = fx * p01 + skew * p11;
  result.J(0, 2) = fx * p02 + skew * p12;
  result.J(1, 0) = fy * p10;
  result.J(1, 1) = fy * p11;
  result.J(1, 2) = fy * p12;

  return detail_impl::finalizeJacobian<T>(result);
}

}  // namespace camxiom::fisheye::impl

namespace camxiom::omnidirectional::impl
{

template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobian(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidJacobian;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
  {
    return invalidJacobian<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }

  const T X = ray_direction.x();
  const T Y = ray_direction.y();
  const T Z = ray_direction.z();
  const T xi = model.projection.xi;

  const T r = std::sqrt(X * X + Y * Y + Z * Z);
  if (!std::isfinite(r) || r <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }

  const T denom = Z + xi * r;
  if (!std::isfinite(denom) || denom <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }
  // Injectivity limit (see the omnidirectional projection impl): binding for xi > 1.
  if (r + xi * Z <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }
  // theta_max contract, matching the scalar forward.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z, r))
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }

  const T inv_denom = T(1) / denom;
  const T x_n = X * inv_denom;
  const T y_n = Y * inv_denom;

  const T inv_r = T(1) / r;
  const T dd_dX = xi * X * inv_r;
  const T dd_dY = xi * Y * inv_r;
  const T dd_dZ = T(1) + xi * Z * inv_r;

  const T inv_denom2 = inv_denom * inv_denom;
  const T p00 = (denom - X * dd_dX) * inv_denom2;
  const T p01 = (-X * dd_dY) * inv_denom2;
  const T p02 = (-X * dd_dZ) * inv_denom2;
  const T p10 = (-Y * dd_dX) * inv_denom2;
  const T p11 = (denom - Y * dd_dY) * inv_denom2;
  const T p12 = (-Y * dd_dZ) * inv_denom2;

  detail_impl::NormPointT<T> undistorted_xy{x_n, y_n};
  detail_impl::NormPointT<T> distorted_xy{};
  T j00 = T(1);
  T j01 = T(0);
  T j10 = T(0);
  T j11 = T(1);
  const StatusCode dist_status = detail_impl::distortPlaneModelWithJacobian<T>(
    model.distortion, undistorted_xy, distorted_xy, j00, j01, j10, j11
  );
  if (dist_status != StatusCode::OK)
  {
    return invalidJacobian<T>(dist_status);
  }

  const T fx = model.intrinsics.fx;
  const T fy = model.intrinsics.fy;
  const T skew = model.intrinsics.skew;

  ProjectionJacobianT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = fx * distorted_xy.x + skew * distorted_xy.y + model.intrinsics.cx;
  result.pixel.v = fy * distorted_xy.y + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }

  const T id00 = fx * j00 + skew * j10;
  const T id01 = fx * j01 + skew * j11;
  const T id10 = fy * j10;
  const T id11 = fy * j11;

  result.J(0, 0) = id00 * p00 + id01 * p10;
  result.J(0, 1) = id00 * p01 + id01 * p11;
  result.J(0, 2) = id00 * p02 + id01 * p12;
  result.J(1, 0) = id10 * p00 + id11 * p10;
  result.J(1, 1) = id10 * p01 + id11 * p11;
  result.J(1, 2) = id10 * p02 + id11 * p12;

  return detail_impl::finalizeJacobian<T>(result);
}

}  // namespace camxiom::omnidirectional::impl

namespace camxiom::double_sphere::impl
{

template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobian(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidJacobian;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
  {
    return invalidJacobian<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }

  const T X = ray_direction.x();
  const T Y = ray_direction.y();
  const T Z = ray_direction.z();
  const T xi = model.projection.xi;
  const T alpha = model.projection.alpha;

  T d1 = T(0);
  T r_sq = T(0);
  T xi_d1_z = T(0);
  T d2 = T(0);
  T denom = T(0);
  const StatusCode fwd =
    detail::computeDsForward<T>(xi, alpha, X, Y, Z, kEpsilon, d1, r_sq, xi_d1_z, d2, denom);
  if (fwd != StatusCode::OK)
  {
    return invalidJacobian<T>(fwd);
  }
  // theta_max contract, matching the scalar forward.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z, d1))
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }
  const T inv_d1 = T(1) / d1;
  const T inv_d2 = T(1) / d2;
  const T inv_denom = T(1) / denom;

  const T x_n = X * inv_denom;
  const T y_n = Y * inv_denom;

  const T inv_denom2 = inv_denom * inv_denom;

  auto ddenom_dV = [&](const T V) -> T {
    const T dd1_dV = V * inv_d1;
    const T dxi_d1_z_dV = xi * dd1_dV;
    const T dd2_dV = (V + xi_d1_z * dxi_d1_z_dV) * inv_d2;
    return alpha * dd2_dV + (T(1) - alpha) * dxi_d1_z_dV;
  };

  const T dd1_dZ = Z * inv_d1;
  const T dxi_d1_z_dZ = xi * dd1_dZ + T(1);
  const T dd2_dZ = (xi_d1_z * dxi_d1_z_dZ) * inv_d2;
  const T ddenom_dZ = alpha * dd2_dZ + (T(1) - alpha) * dxi_d1_z_dZ;

  const T ddenom_dX = ddenom_dV(X);
  const T ddenom_dY = ddenom_dV(Y);

  const T p00 = (denom - X * ddenom_dX) * inv_denom2;
  const T p01 = (-X * ddenom_dY) * inv_denom2;
  const T p02 = (-X * ddenom_dZ) * inv_denom2;
  const T p10 = (-Y * ddenom_dX) * inv_denom2;
  const T p11 = (denom - Y * ddenom_dY) * inv_denom2;
  const T p12 = (-Y * ddenom_dZ) * inv_denom2;

  detail_impl::NormPointT<T> undistorted_xy{x_n, y_n};
  detail_impl::NormPointT<T> distorted_xy{};
  T j00 = T(1);
  T j01 = T(0);
  T j10 = T(0);
  T j11 = T(1);
  const StatusCode dist_status = detail_impl::distortPlaneModelWithJacobian<T>(
    model.distortion, undistorted_xy, distorted_xy, j00, j01, j10, j11
  );
  if (dist_status != StatusCode::OK)
  {
    return invalidJacobian<T>(dist_status);
  }

  const T fx = model.intrinsics.fx;
  const T fy = model.intrinsics.fy;
  const T skew = model.intrinsics.skew;

  ProjectionJacobianT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = fx * distorted_xy.x + skew * distorted_xy.y + model.intrinsics.cx;
  result.pixel.v = fy * distorted_xy.y + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }

  const T id00 = fx * j00 + skew * j10;
  const T id01 = fx * j01 + skew * j11;
  const T id10 = fy * j10;
  const T id11 = fy * j11;

  result.J(0, 0) = id00 * p00 + id01 * p10;
  result.J(0, 1) = id00 * p01 + id01 * p11;
  result.J(0, 2) = id00 * p02 + id01 * p12;
  result.J(1, 0) = id10 * p00 + id11 * p10;
  result.J(1, 1) = id10 * p01 + id11 * p11;
  result.J(1, 2) = id10 * p02 + id11 * p12;

  return detail_impl::finalizeJacobian<T>(result);
}

}  // namespace camxiom::double_sphere::impl

namespace camxiom::eucm::impl
{

template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobian(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidJacobian;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::EUCM)
  {
    return invalidJacobian<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }

  const T X = ray_direction.x();
  const T Y = ray_direction.y();
  const T Z = ray_direction.z();
  const T alpha = model.projection.alpha;
  const T beta = model.projection.beta;

  const T d = std::sqrt(beta * (X * X + Y * Y) + Z * Z);
  if (!std::isfinite(d) || d <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::INVALID_INPUT);
  }
  const T inv_d = T(1) / d;

  const T denom = alpha * d + (T(1) - alpha) * Z;
  if (std::abs(denom) <= kEpsilon)
  {
    return invalidJacobian<T>(StatusCode::DOMAIN_ERROR);
  }
  // Unconditional w-check (at alpha = 0 it degenerates to z > 0), matching
  // the scalar forward; the old alpha gate let alpha ~ 0 mirror rear points.
  {
    const T w = (alpha <= T(0.5)) ? (alpha / (T(1) - alpha)) : ((T(1) - alpha) / alpha);
    if (Z <= -w * d)
    {
      return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
    }
  }
  // theta_max contract; d above is beta-weighted, so use the true norm.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z,
                                   std::sqrt(X * X + Y * Y + Z * Z)))
  {
    return invalidJacobian<T>(StatusCode::OUT_OF_FOV);
  }

  const T inv_denom = T(1) / denom;
  const T x_n = X * inv_denom;
  const T y_n = Y * inv_denom;

  const T inv_denom2 = inv_denom * inv_denom;
  const T dd_dX = alpha * beta * X * inv_d;
  const T dd_dY = alpha * beta * Y * inv_d;
  const T dd_dZ = alpha * Z * inv_d + (T(1) - alpha);

  const T p00 = (denom - X * dd_dX) * inv_denom2;
  const T p01 = (-X * dd_dY) * inv_denom2;
  const T p02 = (-X * dd_dZ) * inv_denom2;
  const T p10 = (-Y * dd_dX) * inv_denom2;
  const T p11 = (denom - Y * dd_dY) * inv_denom2;
  const T p12 = (-Y * dd_dZ) * inv_denom2;

  detail_impl::NormPointT<T> undistorted_xy{x_n, y_n};
  detail_impl::NormPointT<T> distorted_xy{};
  T j00 = T(1);
  T j01 = T(0);
  T j10 = T(0);
  T j11 = T(1);
  const StatusCode dist_status = detail_impl::distortPlaneModelWithJacobian<T>(
    model.distortion, undistorted_xy, distorted_xy, j00, j01, j10, j11
  );
  if (dist_status != StatusCode::OK)
  {
    return invalidJacobian<T>(dist_status);
  }

  const T fx = model.intrinsics.fx;
  const T fy = model.intrinsics.fy;
  const T skew = model.intrinsics.skew;

  ProjectionJacobianT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = fx * distorted_xy.x + skew * distorted_xy.y + model.intrinsics.cx;
  result.pixel.v = fy * distorted_xy.y + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidJacobian<T>(StatusCode::NUMERIC_ERROR);
  }

  const T id00 = fx * j00 + skew * j10;
  const T id01 = fx * j01 + skew * j11;
  const T id10 = fy * j10;
  const T id11 = fy * j11;

  result.J(0, 0) = id00 * p00 + id01 * p10;
  result.J(0, 1) = id00 * p01 + id01 * p11;
  result.J(0, 2) = id00 * p02 + id01 * p12;
  result.J(1, 0) = id10 * p00 + id11 * p10;
  result.J(1, 1) = id10 * p01 + id11 * p11;
  result.J(1, 2) = id10 * p02 + id11 * p12;

  return detail_impl::finalizeJacobian<T>(result);
}

}  // namespace camxiom::eucm::impl

namespace camxiom::detail_impl
{

// Generic dispatch: validate once (precision-specific), then route by type.
// Query-tier guard, matching the scalar / batch / SIMD projection paths.
template <typename T>
inline ProjectionJacobianT<T> rayToPixelWithJacobianDispatch(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  StatusCode validation;
  if constexpr (std::is_same_v<T, float>)
  {
    validation = detail::validateCameraModelQuery(model);
  }
  else
  {
    validation = detail64::validateCameraModelQuery64(model);
  }
  if (validation != StatusCode::OK)
  {
    return invalidJacobian<T>(validation);
  }

  switch (model.projection.type)
  {
    case ProjectionModelType::PINHOLE:
      return pinhole::impl::rayToPixelWithJacobian<T>(model, ray_direction);
    case ProjectionModelType::FISHEYE_THETA:
      return fisheye::impl::rayToPixelWithJacobian<T>(model, ray_direction);
    case ProjectionModelType::OMNIDIRECTIONAL:
      return omnidirectional::impl::rayToPixelWithJacobian<T>(model, ray_direction);
    case ProjectionModelType::DOUBLE_SPHERE:
      return double_sphere::impl::rayToPixelWithJacobian<T>(model, ray_direction);
    case ProjectionModelType::EUCM:
      return eucm::impl::rayToPixelWithJacobian<T>(model, ray_direction);
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return invalidJacobian<T>(StatusCode::INVALID_MODEL);
}

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__JACOBIAN__JACOBIAN_IMPL_HPP
