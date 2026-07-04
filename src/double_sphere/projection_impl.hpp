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

#ifndef CAMXIOM__DOUBLE_SPHERE__PROJECTION_IMPL_HPP
#define CAMXIOM__DOUBLE_SPHERE__PROJECTION_IMPL_HPP

// Scalar-templated Double Sphere projection core (#1 step 3). Single source of
// truth for the forward/inverse math previously hand-duplicated between the
// float implementation (src/double_sphere/projection.cpp) and the double
// implementation (src/double_sphere/projection64.cpp). The forward shares the
// already-templated detail::computeDsForward<T>; the inverse lifting math was
// identical across precisions and is reproduced verbatim with T literals.

#include "camxiom/types.hpp"
#include "detail/ds_forward.hpp"
#include "detail/projection_common.hpp"
#include "distortion/plane_impl.hpp"

#include <Eigen/Core>

#include <cmath>

namespace camxiom::double_sphere::impl
{

template <typename T>
inline PixelResultT<T> rayToPixel(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidPixelResult;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
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
    return invalidPixelResult<T>(fwd);
  }

  // theta_max contract (types.hpp): reject rays beyond a sub-pi FOV cap.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, Z, d1))
  {
    return invalidPixelResult<T>(StatusCode::OUT_OF_FOV);
  }

  detail_impl::NormPointT<T> undistorted_xy{X / denom, Y / denom};
  if (!detail_impl::isFinite2(undistorted_xy.x, undistorted_xy.y))
  {
    return invalidPixelResult<T>(StatusCode::NUMERIC_ERROR);
  }

  detail_impl::NormPointT<T> distorted_xy{};
  const StatusCode dist_status =
    detail_impl::distortPlaneModel<T>(model.distortion, undistorted_xy, distorted_xy);
  if (dist_status != StatusCode::OK)
  {
    return invalidPixelResult<T>(dist_status);
  }

  PixelResultT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = model.intrinsics.fx * distorted_xy.x + model.intrinsics.skew * distorted_xy.y +
                   model.intrinsics.cx;
  result.pixel.v = model.intrinsics.fy * distorted_xy.y + model.intrinsics.cy;
  if (!detail_impl::isFinite2(result.pixel.u, result.pixel.v))
  {
    return invalidPixelResult<T>(StatusCode::NUMERIC_ERROR);
  }
  return result;
}

template <typename T, typename Options>
inline RayResultT<T> pixelToRay(
  const CameraModelT<T> &model, const Pixel2T<T> &pixel, const Options &solver_options
)
{
  using detail_impl::invalidRayResult;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
  {
    return invalidRayResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite2(pixel.u, pixel.v))
  {
    return invalidRayResult<T>(StatusCode::INVALID_INPUT);
  }

  T x_d = T(0);
  T y_d = T(0);
  if (!detail_impl::removeIntrinsics<T>(model.intrinsics, pixel, x_d, y_d))
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }

  detail_impl::NormPointT<T> undistorted_xy{};
  const StatusCode undist_status = detail_impl::undistortPlaneSolve(
    model.distortion, detail_impl::NormPointT<T>{x_d, y_d}, solver_options.max_iterations,
    solver_options.residual_tolerance, solver_options.step_tolerance, solver_options.skip_verify,
    undistorted_xy
  );
  if (undist_status != StatusCode::OK)
  {
    return invalidRayResult<T>(undist_status);
  }

  const T mx = undistorted_xy.x;
  const T my = undistorted_xy.y;
  const T alpha = model.projection.alpha;
  const T xi = model.projection.xi;
  const T r2 = mx * mx + my * my;

  const T inner = T(1) - (T(2) * alpha - T(1)) * r2;
  if (inner < T(0))
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }
  const T sqrt_inner = std::sqrt(inner);
  const T mz_num = T(1) - alpha * alpha * r2;
  const T mz_den = alpha * sqrt_inner + T(1) - alpha;
  if (std::abs(mz_den) <= kEpsilon)
  {
    return invalidRayResult<T>(StatusCode::DOMAIN_ERROR);
  }
  const T mz = mz_num / mz_den;

  const T norm_sq = r2 + mz * mz;
  const T inner_xi = mz * mz + (T(1) - xi * xi) * r2;
  if (inner_xi < T(0))
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }
  const T scalar = (mz * xi + std::sqrt(inner_xi)) / norm_sq;

  Eigen::Matrix<T, 3, 1> direction(scalar * mx, scalar * my, scalar * mz - xi);
  const T norm = direction.norm();
  if (!std::isfinite(norm) || norm <= kEpsilon)
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }
  direction /= norm;

  // Round-trip consistency with the forward theta_max cap.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max, direction.z(), T(1)))
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }

  RayResultT<T> result;
  result.status = StatusCode::OK;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = direction;
  return result;
}

}  // namespace camxiom::double_sphere::impl

#endif  // CAMXIOM__DOUBLE_SPHERE__PROJECTION_IMPL_HPP
