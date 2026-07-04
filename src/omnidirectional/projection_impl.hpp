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

#ifndef CAMXIOM__OMNIDIRECTIONAL__PROJECTION_IMPL_HPP
#define CAMXIOM__OMNIDIRECTIONAL__PROJECTION_IMPL_HPP

// Scalar-templated omnidirectional (Mei unified sphere) projection core
// (#1 step 3). Single source of truth for the forward/inverse math previously
// hand-duplicated between the float implementation (src/omnidirectional/
// projection.cpp) and the double implementation (src/omnidirectional/
// projection64.cpp).
//
// The two precisions had drifted apart in pixelToRay (a classic #1 hazard):
//   - a negative lifting discriminant returned OUT_OF_FOV in float but
//     DOMAIN_ERROR in double;
//   - float additionally guarded the discriminant with isfinite + a max(.,0)
//     clamp and rejected a non-finite / sub-epsilon (1 + r^2) denominator,
//     while double skipped both checks.
// This core unifies on the more defensive *float* behaviour (the protected
// runtime hot path): a negative discriminant is OUT_OF_FOV, the clamp and the
// denominator/finite guards are kept for both precisions. No round-trip test
// exercised the divergent double-only DOMAIN_ERROR branch.

#include "camxiom/types.hpp"
#include "detail/projection_common.hpp"
#include "distortion/plane_impl.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace camxiom::omnidirectional::impl
{

template <typename T>
inline PixelResultT<T> rayToPixel(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidPixelResult;
  constexpr T kEpsilon = detail_impl::PlaneTraits<T>::kEpsilon;

  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }

  const T xi = model.projection.xi;
  const T ray_norm = ray_direction.norm();
  if (!std::isfinite(ray_norm) || ray_norm <= kEpsilon)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }

  const T denominator = ray_direction.z() + xi * ray_norm;
  if (!std::isfinite(denominator) || denominator <= kEpsilon)
  {
    return invalidPixelResult<T>(StatusCode::OUT_OF_FOV);
  }

  // Injectivity limit of the unified-sphere forward model: r_u(theta) is
  // monotone only while 1 + xi*cos(theta) > 0, i.e. norm + xi*z > 0. For
  // |xi| <= 1 this is implied by denominator > 0, but for xi > 1 (common in
  // fisheye Mei fits) the denominator is positive for *every* direction and
  // rays beyond theta = acos(-1/xi) alias onto valid-looking pixels.
  if (ray_norm + xi * ray_direction.z() <= kEpsilon)
  {
    return invalidPixelResult<T>(StatusCode::OUT_OF_FOV);
  }

  // theta_max contract (types.hpp): reject rays beyond a sub-pi FOV cap.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max) &&
      !detail_impl::withinThetaMax(model.projection.theta_max,
                                   ray_direction.z(), ray_norm))
  {
    return invalidPixelResult<T>(StatusCode::OUT_OF_FOV);
  }

  detail_impl::NormPointT<T> undistorted_xy{};
  undistorted_xy.x = ray_direction.x() / denominator;
  undistorted_xy.y = ray_direction.y() / denominator;
  if (!detail_impl::isFinite2(undistorted_xy.x, undistorted_xy.y))
  {
    return invalidPixelResult<T>(StatusCode::NUMERIC_ERROR);
  }

  detail_impl::NormPointT<T> distorted_xy{};
  const StatusCode distortion_status =
    detail_impl::distortPlaneModel<T>(model.distortion, undistorted_xy, distorted_xy);
  if (distortion_status != StatusCode::OK)
  {
    return invalidPixelResult<T>(distortion_status);
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

  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
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

  detail_impl::NormPointT<T> undistorted_xy{};
  const StatusCode undistort_status = detail_impl::undistortPlaneSolve(
    model.distortion, detail_impl::NormPointT<T>{x_distorted, y_distorted},
    solver_options.max_iterations, solver_options.residual_tolerance, solver_options.step_tolerance,
    solver_options.skip_verify, undistorted_xy
  );
  if (undistort_status != StatusCode::OK)
  {
    return invalidRayResult<T>(undistort_status);
  }

  const T xi = model.projection.xi;
  const T r2 = undistorted_xy.x * undistorted_xy.x + undistorted_xy.y * undistorted_xy.y;
  const T inside_sqrt = T(1) + (T(1) - xi * xi) * r2;
  if (!std::isfinite(inside_sqrt) || inside_sqrt < T(0))
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }

  const T sqrt_term = std::sqrt(std::max(inside_sqrt, T(0)));
  const T denominator = T(1) + r2;
  if (!std::isfinite(denominator) || denominator <= kEpsilon)
  {
    return invalidRayResult<T>(StatusCode::DOMAIN_ERROR);
  }

  const T lambda = (xi + sqrt_term) / denominator;
  if (!std::isfinite(lambda))
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }

  Eigen::Matrix<T, 3, 1> direction(
    lambda * undistorted_xy.x, lambda * undistorted_xy.y, lambda - xi
  );
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

}  // namespace camxiom::omnidirectional::impl

#endif  // CAMXIOM__OMNIDIRECTIONAL__PROJECTION_IMPL_HPP
