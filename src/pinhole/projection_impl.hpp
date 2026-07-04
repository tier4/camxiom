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

#ifndef CAMXIOM__PINHOLE__PROJECTION_IMPL_HPP
#define CAMXIOM__PINHOLE__PROJECTION_IMPL_HPP

// Scalar-templated pinhole projection core (#1 step 3). Single source of truth
// for the perspective forward/inverse math previously hand-duplicated between
// the float implementation (src/pinhole/projection.cpp, camxiom::detail) and
// the double implementation (src/pinhole/projection64.cpp, camxiom::detail64).
//
// The thin float / double wrappers instantiate these for <float> / <double>
// and keep the public rayToPixel / rayToPixel64 (and pixelToRay*) signatures
// unchanged. PlaneTraits<T>::kEpsilon supplies the per-precision epsilon that
// matched the old detail::kEpsilon (1e-8f) / detail64::kEpsilon (1e-15).

#include "camxiom/types.hpp"
#include "detail/projection_common.hpp"
#include "distortion/plane_impl.hpp"

#include <Eigen/Core>

#include <cmath>

namespace camxiom::pinhole::impl
{

template <typename T>
inline PixelResultT<T> rayToPixel(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidPixelResult;

  if (model.projection.type != ProjectionModelType::PINHOLE)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }
  if (ray_direction.z() <= T(0))
  {
    return invalidPixelResult<T>(StatusCode::BEHIND_CAMERA);
  }

  const T inv_z = T(1) / ray_direction.z();
  detail_impl::NormPointT<T> undistorted_xy{};
  undistorted_xy.x = ray_direction.x() * inv_z;
  undistorted_xy.y = ray_direction.y() * inv_z;

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

  if (model.projection.type != ProjectionModelType::PINHOLE)
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

  Eigen::Matrix<T, 3, 1> direction(undistorted_xy.x, undistorted_xy.y, T(1));
  const T norm = direction.norm();
  if (!std::isfinite(norm) || norm <= detail_impl::PlaneTraits<T>::kEpsilon)
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }
  direction /= norm;

  RayResultT<T> result;
  result.status = StatusCode::OK;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = direction;
  return result;
}

}  // namespace camxiom::pinhole::impl

#endif  // CAMXIOM__PINHOLE__PROJECTION_IMPL_HPP
