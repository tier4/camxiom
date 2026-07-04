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

#ifndef CAMXIOM__FISHEYE__PROJECTION_IMPL_HPP
#define CAMXIOM__FISHEYE__PROJECTION_IMPL_HPP

// Scalar-templated fisheye (theta) projection core (#1 step 3). Single source
// of truth for the angle-model forward/inverse math previously hand-duplicated
// between the float implementation (src/fisheye/projection.cpp) and the double
// implementation (src/fisheye/projection64.cpp).
//
// Forward uses the inlined distortTheta<T>; the heavy inverse theta solve goes
// through the out-of-line detail_impl::undistortThetaSolve shim so it is not
// inlined into every per-model translation unit (see projection_common.hpp).
// PlaneTraits<T>::kEpsilon reproduces the old detail::kEpsilon (1e-8f) /
// detail64::kEpsilon (1e-15) thresholds bit-for-bit.

#include "camxiom/types.hpp"
#include "detail/projection_common.hpp"
#include "distortion/angle_impl.hpp"
#include "distortion/plane_impl.hpp"

#include <Eigen/Core>

#include <cmath>

namespace camxiom::fisheye::impl
{

template <typename T>
inline PixelResultT<T> rayToPixel(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidPixelResult;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }
  const T ray_norm_sq = ray_direction.squaredNorm();
  if (!std::isfinite(ray_norm_sq) || ray_norm_sq <= kEpsilon)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }

  const T xy_norm =
    std::sqrt(ray_direction.x() * ray_direction.x() + ray_direction.y() * ray_direction.y());
  const T theta = std::atan2(xy_norm, ray_direction.z());
  if (theta < T(0))
  {
    return invalidPixelResult<T>(StatusCode::DOMAIN_ERROR);
  }
  if (theta > model.projection.theta_max)
  {
    return invalidPixelResult<T>(StatusCode::OUT_OF_FOV);
  }
  if (xy_norm <= kEpsilon && ray_direction.z() < T(0))
  {
    return invalidPixelResult<T>(StatusCode::DOMAIN_ERROR);
  }

  T theta_d = T(0);
  const StatusCode theta_status = detail_impl::distortTheta<T>(model.distortion, theta, theta_d);
  if (theta_status != StatusCode::OK)
  {
    return invalidPixelResult<T>(theta_status);
  }

  T x_d = T(0);
  T y_d = T(0);
  if (xy_norm > kEpsilon)
  {
    const T scale = theta_d / xy_norm;
    x_d = ray_direction.x() * scale;
    y_d = ray_direction.y() * scale;
  }

  PixelResultT<T> result;
  result.status = StatusCode::OK;
  result.pixel.u = model.intrinsics.fx * x_d + model.intrinsics.skew * y_d + model.intrinsics.cx;
  result.pixel.v = model.intrinsics.fy * y_d + model.intrinsics.cy;
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

  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    return invalidRayResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite2(pixel.u, pixel.v))
  {
    return invalidRayResult<T>(StatusCode::INVALID_INPUT);
  }

  T x_distorted = T(0);
  T y_distorted = T(0);
  if (!detail_impl::removeIntrinsics<T>(model.intrinsics, pixel, x_distorted, y_distorted))
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }

  const T radius_d = std::sqrt(x_distorted * x_distorted + y_distorted * y_distorted);
  T theta = T(0);
  const StatusCode theta_status = detail_impl::undistortThetaSolve(
    model.distortion, model.projection, radius_d, solver_options.max_iterations,
    solver_options.residual_tolerance, solver_options.step_tolerance, theta
  );
  if (theta_status != StatusCode::OK)
  {
    return invalidRayResult<T>(theta_status);
  }
  if (theta > model.projection.theta_max)
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }

  RayResultT<T> result;
  result.status = StatusCode::OK;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();

  if (radius_d <= kEpsilon)
  {
    result.ray.direction = Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1));
    return result;
  }

  const T inv_radius = T(1) / radius_d;
  const T cos_phi = x_distorted * inv_radius;
  const T sin_phi = y_distorted * inv_radius;
  const T sin_theta = std::sin(theta);
  const T cos_theta = std::cos(theta);
  Eigen::Matrix<T, 3, 1> direction(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
  const T norm = direction.norm();
  if (!std::isfinite(norm) || norm <= kEpsilon)
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }
  direction /= norm;
  result.ray.direction = direction;
  return result;
}

}  // namespace camxiom::fisheye::impl

#endif  // CAMXIOM__FISHEYE__PROJECTION_IMPL_HPP
