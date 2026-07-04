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

#ifndef CAMXIOM__MODEL_COMPARE_HPP
#define CAMXIOM__MODEL_COMPARE_HPP

// Equality / approximate-equality for camera model structs.
//
// * operator==  : EXACT field-wise equality of every member, including the
//   derived auxiliary distortion state (space / count / flags / tilt
//   matrices). Intended for cache-invalidation checks ("did the model
//   change at all?"). Standard float semantics apply: any NaN member makes
//   the comparison false.
// * isApprox    : tolerance-based comparison of the SEMANTIC parameters
//   (types must match exactly; intrinsics / projection scalars / the
//   `count` leading distortion coefficients are compared with a combined
//   absolute+relative tolerance). Derived state (space, aux flags, tilt
//   matrices) is NOT compared — it is a function of type+coeffs. Intended
//   for regression comparison of calibration results across
//   solver/platform noise.
//
// Header-only templates: apply to both the float aliases (CameraModel, ...)
// and the double aliases (CameraModel64, ...).

#include "camxiom/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace camxiom
{

// --------------------------------------------------------------------------
// Exact equality
// --------------------------------------------------------------------------

template <typename T>
inline bool operator==(const IntrinsicsModelT<T> &a, const IntrinsicsModelT<T> &b)
{
  return a.fx == b.fx && a.fy == b.fy && a.cx == b.cx && a.cy == b.cy && a.skew == b.skew;
}

template <typename T>
inline bool operator!=(const IntrinsicsModelT<T> &a, const IntrinsicsModelT<T> &b)
{
  return !(a == b);
}

template <typename T>
inline bool operator==(const ProjectionModelT<T> &a, const ProjectionModelT<T> &b)
{
  return a.type == b.type && a.theta_max == b.theta_max && a.xi == b.xi && a.alpha == b.alpha &&
         a.beta == b.beta;
}

template <typename T>
inline bool operator!=(const ProjectionModelT<T> &a, const ProjectionModelT<T> &b)
{
  return !(a == b);
}

template <typename T>
inline bool operator==(const DistortionModelT<T> &a, const DistortionModelT<T> &b)
{
  return a.type == b.type && a.space == b.space && a.coeffs == b.coeffs &&
         a.tilt_matrix == b.tilt_matrix && a.inv_tilt_matrix == b.inv_tilt_matrix &&
         a.count == b.count && a.is_rational == b.is_rational &&
         a.has_thin_prism == b.has_thin_prism && a.has_tilt == b.has_tilt;
}

template <typename T>
inline bool operator!=(const DistortionModelT<T> &a, const DistortionModelT<T> &b)
{
  return !(a == b);
}

template <typename T>
inline bool operator==(const CameraModelT<T> &a, const CameraModelT<T> &b)
{
  return a.intrinsics == b.intrinsics && a.projection == b.projection &&
         a.distortion == b.distortion;
}

template <typename T>
inline bool operator!=(const CameraModelT<T> &a, const CameraModelT<T> &b)
{
  return !(a == b);
}

// --------------------------------------------------------------------------
// Approximate equality
// --------------------------------------------------------------------------

namespace detail
{
/// Combined absolute + relative closeness test:
///   |a - b| <= abs_tol + rel_tol * max(|a|, |b|).
/// NaN on either side compares false; equal infinities compare true.
template <typename T>
inline bool scalarsClose(const T a, const T b, const T rel_tol, const T abs_tol)
{
  if (a == b)  // fast path; also handles matching infinities
  {
    return true;
  }
  if (!std::isfinite(a) || !std::isfinite(b))
  {
    return false;
  }
  const T diff = std::abs(a - b);
  const T scale = std::max(std::abs(a), std::abs(b));
  return diff <= abs_tol + rel_tol * scale;
}
}  // namespace detail

/// Default tolerances: loose enough to absorb float<->double round-trips and
/// cross-platform solver noise, tight enough to catch a real parameter change.
template <typename T>
inline bool isApprox(
  const IntrinsicsModelT<T> &a, const IntrinsicsModelT<T> &b, const T rel_tol = T(1e-5),
  const T abs_tol = T(1e-6)
)
{
  return detail::scalarsClose(a.fx, b.fx, rel_tol, abs_tol) &&
         detail::scalarsClose(a.fy, b.fy, rel_tol, abs_tol) &&
         detail::scalarsClose(a.cx, b.cx, rel_tol, abs_tol) &&
         detail::scalarsClose(a.cy, b.cy, rel_tol, abs_tol) &&
         detail::scalarsClose(a.skew, b.skew, rel_tol, abs_tol);
}

template <typename T>
inline bool isApprox(
  const ProjectionModelT<T> &a, const ProjectionModelT<T> &b, const T rel_tol = T(1e-5),
  const T abs_tol = T(1e-6)
)
{
  return a.type == b.type && detail::scalarsClose(a.theta_max, b.theta_max, rel_tol, abs_tol) &&
         detail::scalarsClose(a.xi, b.xi, rel_tol, abs_tol) &&
         detail::scalarsClose(a.alpha, b.alpha, rel_tol, abs_tol) &&
         detail::scalarsClose(a.beta, b.beta, rel_tol, abs_tol);
}

template <typename T>
inline bool isApprox(
  const DistortionModelT<T> &a, const DistortionModelT<T> &b, const T rel_tol = T(1e-5),
  const T abs_tol = T(1e-6)
)
{
  if (a.type != b.type || a.count != b.count)
  {
    return false;
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(a.count); ++i)
  {
    if (!detail::scalarsClose(a.coeffs[i], b.coeffs[i], rel_tol, abs_tol))
    {
      return false;
    }
  }
  return true;
}

template <typename T>
inline bool isApprox(
  const CameraModelT<T> &a, const CameraModelT<T> &b, const T rel_tol = T(1e-5),
  const T abs_tol = T(1e-6)
)
{
  return isApprox(a.intrinsics, b.intrinsics, rel_tol, abs_tol) &&
         isApprox(a.projection, b.projection, rel_tol, abs_tol) &&
         isApprox(a.distortion, b.distortion, rel_tol, abs_tol);
}

}  // namespace camxiom

#endif  // CAMXIOM__MODEL_COMPARE_HPP
