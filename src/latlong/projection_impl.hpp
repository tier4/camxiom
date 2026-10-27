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

#ifndef CAMXIOM__LATLONG__PROJECTION_IMPL_HPP
#define CAMXIOM__LATLONG__PROJECTION_IMPL_HPP

// Scalar-templated core of the lat-long projections, EQUIRECTANGULAR and
// CYLINDRICAL (contract in types.hpp). Both put the longitude on the
// horizontal axis; they differ only in the vertical coordinate (the latitude
// itself, or its tangent). They take no lens correction, so the forward and
// inverse maps are closed form.

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "detail/projection_common.hpp"
#include "distortion/plane_impl.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>

namespace camxiom::latlong::impl
{

/// Longitude atan2(x, z) in [-pi, pi): the seam directly behind the camera
/// (atan2 returns +pi for x = +0, z < 0) belongs to the left edge.
template <typename T>
inline T longitude(const T x, const T z)
{
  const T pi = static_cast<T>(constants::kPi);
  const T lambda = std::atan2(x, z);
  return lambda >= pi ? lambda - T(2) * pi : lambda;
}

/// Distance of a ray from the vertical axis, sqrt(x^2 + z^2). Zero means a
/// pole, where the longitude is undefined. The test is relative to the ray
/// length so a valid but very short ray is not taken for a pole.
template <typename T>
inline bool isPole(const T horizontal, const T ray_norm)
{
  return horizontal <= detail_impl::PlaneTraits<T>::kEpsilon * ray_norm;
}

/// Rounding slack, in radians, for an angle that should sit exactly on a
/// limit of the image (the seam at -pi, a pole at +-pi/2): a pixel on the
/// image edge recovers that limit only to the last few bits after removing
/// the intrinsics. A handful of float ulps of pi (about 6e-6 rad) in both
/// precisions, so that a whole-sphere float model widened to double usually
/// keeps its edges (not guaranteed: widening cannot undo the float rounding
/// of the intrinsics), but never more than a hundredth of a pixel along that
/// axis (`scale` is fx or fy):
/// with a long focal length 6e-6 rad spans several pixels, and pixels that
/// far outside the image must not collapse onto the limit. The hundredth of
/// a pixel holds for a double model; in float, removing the intrinsics is
/// itself rounded to float's resolution at that pixel coordinate, which for
/// a very long focal length is coarser than this slack.
template <typename T>
inline T limitSlack(const T scale)
{
  const T rounding = static_cast<T>(16.0 * std::numeric_limits<float>::epsilon() * constants::kPi);
  return std::min(rounding, T(0.01) / std::abs(scale));
}

template <typename T>
inline PixelResultT<T> rayToPixel(
  const CameraModelT<T> &model, const Eigen::Matrix<T, 3, 1> &ray_direction
)
{
  using detail_impl::invalidPixelResult;

  if (!isLatLongProjection(model.projection.type))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite3(ray_direction.x(), ray_direction.y(), ray_direction.z()))
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }
  const T ray_norm_sq = ray_direction.squaredNorm();
  if (!std::isfinite(ray_norm_sq) || ray_norm_sq <= detail_impl::PlaneTraits<T>::kEpsilon)
  {
    return invalidPixelResult<T>(StatusCode::INVALID_INPUT);
  }

  const T x = ray_direction.x();
  const T y = ray_direction.y();
  const T z = ray_direction.z();
  const T horizontal = std::sqrt(x * x + z * z);
  if (isPole(horizontal, std::sqrt(ray_norm_sq)))
  {
    return invalidPixelResult<T>(StatusCode::DOMAIN_ERROR);
  }

  const T x_d = longitude(x, z);
  const T y_d = model.projection.type == ProjectionModelType::EQUIRECTANGULAR
                  ? std::atan2(y, horizontal)
                  : y / horizontal;

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
  const CameraModelT<T> &model, const Pixel2T<T> &pixel, const Options & /*solver_options*/
)
{
  using detail_impl::invalidRayResult;

  if (!isLatLongProjection(model.projection.type))
  {
    return invalidRayResult<T>(StatusCode::INVALID_MODEL);
  }
  if (!detail_impl::isFinite2(pixel.u, pixel.v))
  {
    return invalidRayResult<T>(StatusCode::INVALID_INPUT);
  }

  T lambda = T(0);
  T y_d = T(0);
  if (!detail_impl::removeIntrinsics<T>(model.intrinsics, pixel, lambda, y_d))
  {
    return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
  }

  // The longitude axis is not wrapped: a pixel past the seam is outside the
  // image, not another view of the scene. The left image edge is the seam
  // itself; within rounding slack it is taken as exactly -pi.
  const T pi = static_cast<T>(constants::kPi);
  if (!(lambda >= -pi - limitSlack<T>(model.intrinsics.fx) && lambda < pi))
  {
    return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
  }
  lambda = std::max(lambda, -pi);
  // sin(lambda) must keep the sign of lambda, which places the ray on the
  // side of the seam the pixel is on. The rounded pi differs from the true
  // one (it exceeds it in float), so sin(-pi) can come out with the wrong
  // sign and the left edge would project back onto the right edge; the seam
  // itself is the direction straight behind the camera.
  T sin_lambda = std::sin(lambda);
  if ((lambda < T(0) && sin_lambda > T(0)) || (lambda > T(0) && sin_lambda < T(0)))
  {
    sin_lambda = std::copysign(T(0), lambda);
  }
  const T cos_lambda = std::cos(lambda);

  Eigen::Matrix<T, 3, 1> direction;
  if (model.projection.type == ProjectionModelType::EQUIRECTANGULAR)
  {
    const T half_pi = static_cast<T>(constants::kHalfPi);
    if (!(std::abs(y_d) <= half_pi + limitSlack<T>(model.intrinsics.fy)))
    {
      return invalidRayResult<T>(StatusCode::OUT_OF_FOV);
    }
    // At |phi| = pi/2 the rounded half-pi can sit just past the true pole,
    // where the cosine turns negative; clamp so that row stays at the pole.
    const T cos_phi = std::max(std::cos(y_d), T(0));
    direction = Eigen::Matrix<T, 3, 1>(cos_phi * sin_lambda, std::sin(y_d), cos_phi * cos_lambda);
  }
  else
  {
    direction = Eigen::Matrix<T, 3, 1>(sin_lambda, y_d, cos_lambda);
    const T norm = direction.norm();
    if (!std::isfinite(norm))
    {
      return invalidRayResult<T>(StatusCode::NUMERIC_ERROR);
    }
    direction /= norm;
  }

  // A pole has no longitude, so it is outside the projection in both
  // directions: the inverse refuses the rows that reach it exactly as the
  // forward map refuses the ray (same test, so an OK ray always projects).
  const T horizontal = std::sqrt(direction.x() * direction.x() + direction.z() * direction.z());
  if (isPole(horizontal, direction.norm()))
  {
    return invalidRayResult<T>(StatusCode::DOMAIN_ERROR);
  }

  RayResultT<T> result;
  result.status = StatusCode::OK;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = direction;
  return result;
}

}  // namespace camxiom::latlong::impl

#endif  // CAMXIOM__LATLONG__PROJECTION_IMPL_HPP
