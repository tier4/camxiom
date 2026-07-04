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

#ifndef CAMXIOM__DISTORTION__ANGLE_IMPL_HPP
#define CAMXIOM__DISTORTION__ANGLE_IMPL_HPP

// Scalar-templated angle (fisheye-family) forward distortion core
// (#1 step 2b-i). Single source of truth for the theta -> theta_d mapping and
// its derivative, previously hand-duplicated between the float implementation
// (src/distortion/angle.cpp, camxiom::detail) and the double implementation
// (src/projection64/internal.hpp, camxiom::detail64).
//
// The float and double thin wrappers instantiate these for <float> / <double>.
//
// undistortThetaHybrid (the inverse theta solver) is also unified here on the
// more robust historical float algorithm: a bracketed Newton iteration with
// bisection fallback and a step-tolerance re-verify. Git history shows the
// previous float and double versions were authored together (commit d375cfd)
// and never tuned separately, so the divergence was unintentional rather than a
// deliberate speed/precision trade-off.

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "distortion/plane_impl.hpp"  // PlaneTraits<T>::kEpsilon (shared threshold)

#include <algorithm>
#include <cmath>
#include <limits>

namespace camxiom::detail_impl
{

template <typename T>
inline StatusCode distortTheta(const DistortionModelT<T> &model, const T theta, T &theta_d)
{
  if (!std::isfinite(theta))
  {
    return StatusCode::INVALID_INPUT;
  }
  if (theta < T(0))
  {
    return StatusCode::DOMAIN_ERROR;
  }

  switch (model.type)
  {
    case DistortionModelType::NONE:
    case DistortionModelType::EQUIDISTANT:
      theta_d = theta;
      break;
    case DistortionModelType::OPENCV_FISHEYE4: {
      const T t2 = theta * theta;
      const T t4 = t2 * t2;
      const T t6 = t4 * t2;
      const T t8 = t4 * t4;
      const auto &c = model.coeffs;
      theta_d = theta * (T(1) + c[0] * t2 + c[1] * t4 + c[2] * t6 + c[3] * t8);
      break;
    }
    case DistortionModelType::KB4: {
      const T t2 = theta * theta;
      const T t3 = t2 * theta;
      const T t5 = t3 * t2;
      const T t7 = t5 * t2;
      const T t9 = t7 * t2;
      const auto &c = model.coeffs;
      theta_d = theta + c[0] * t3 + c[1] * t5 + c[2] * t7 + c[3] * t9;
      break;
    }
    case DistortionModelType::KB8: {
      const T t2 = theta * theta;
      const T t3 = t2 * theta;
      const T t5 = t3 * t2;
      const T t7 = t5 * t2;
      const T t9 = t7 * t2;
      const T t11 = t9 * t2;
      const T t13 = t11 * t2;
      const T t15 = t13 * t2;
      const T t17 = t15 * t2;
      const auto &c = model.coeffs;
      theta_d = theta + c[0] * t3 + c[1] * t5 + c[2] * t7 + c[3] * t9 + c[4] * t11 + c[5] * t13 +
                c[6] * t15 + c[7] * t17;
      break;
    }
    case DistortionModelType::EQUISOLID:
      theta_d = T(2) * std::sin(theta * T(0.5));
      break;
    case DistortionModelType::STEREOGRAPHIC:
      theta_d = T(2) * std::tan(theta * T(0.5));
      break;
    case DistortionModelType::ORTHOGRAPHIC:
      theta_d = std::sin(theta);
      break;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      return StatusCode::INVALID_MODEL;
  }

  if (!std::isfinite(theta_d))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode distortThetaDerivative(
  const DistortionModelT<T> &model, const T theta, T &derivative_out
)
{
  if (!std::isfinite(theta) || theta < T(0))
  {
    return StatusCode::INVALID_INPUT;
  }

  switch (model.type)
  {
    case DistortionModelType::NONE:
    case DistortionModelType::EQUIDISTANT:
      derivative_out = T(1);
      break;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4: {
      const T t2 = theta * theta;
      const T t4 = t2 * t2;
      const T t6 = t4 * t2;
      const T t8 = t4 * t4;
      const auto &c = model.coeffs;
      derivative_out =
        T(1) + T(3) * c[0] * t2 + T(5) * c[1] * t4 + T(7) * c[2] * t6 + T(9) * c[3] * t8;
      break;
    }
    case DistortionModelType::KB8: {
      const T t2 = theta * theta;
      const T t4 = t2 * t2;
      const T t6 = t4 * t2;
      const T t8 = t4 * t4;
      const T t10 = t8 * t2;
      const T t12 = t8 * t4;
      const T t14 = t8 * t6;
      const T t16 = t8 * t8;
      const auto &c = model.coeffs;
      derivative_out = T(1) + T(3) * c[0] * t2 + T(5) * c[1] * t4 + T(7) * c[2] * t6 +
                       T(9) * c[3] * t8 + T(11) * c[4] * t10 + T(13) * c[5] * t12 +
                       T(15) * c[6] * t14 + T(17) * c[7] * t16;
      break;
    }
    case DistortionModelType::EQUISOLID:
      derivative_out = std::cos(theta * T(0.5));
      break;
    case DistortionModelType::STEREOGRAPHIC: {
      const T c2 = std::cos(theta * T(0.5));
      derivative_out = T(1) / (c2 * c2);
      break;
    }
    case DistortionModelType::ORTHOGRAPHIC:
      derivative_out = std::cos(theta);
      break;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      return StatusCode::INVALID_MODEL;
  }

  if (!std::isfinite(derivative_out))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  return StatusCode::OK;
}

// Inverse theta solve: bracketed Newton with bisection fallback (the historical
// float algorithm), now shared by both precisions. kEpsilon is per-type via
// PlaneTraits<T> (float 1e-8, double 1e-15), matching the previous thresholds.
template <typename T>
inline StatusCode undistortThetaHybrid(
  const DistortionModelT<T> &model, const ProjectionModelT<T> &projection, const T radius_d,
  const int max_iterations, const T residual_tolerance, const T step_tolerance, T &theta_out
)
{
  if (!std::isfinite(radius_d))
  {
    return StatusCode::INVALID_INPUT;
  }
  if (radius_d < T(0))
  {
    return StatusCode::DOMAIN_ERROR;
  }
  if (radius_d <= PlaneTraits<T>::kEpsilon)
  {
    theta_out = T(0);
    return StatusCode::OK;
  }

  T theta_min = T(0);
  T theta_max = projection.theta_max;
  if (!std::isfinite(theta_max) || theta_max <= T(0))
  {
    return StatusCode::INVALID_MODEL;
  }

  T f_min = T(0);
  StatusCode status = distortTheta<T>(model, theta_min, f_min);
  if (status != StatusCode::OK)
  {
    return status;
  }
  f_min -= radius_d;

  T f_max = T(0);
  status = distortTheta<T>(model, theta_max, f_max);
  if (status != StatusCode::OK)
  {
    return status;
  }
  f_max -= radius_d;
  if (f_max < T(0))
  {
    return StatusCode::OUT_OF_FOV;
  }

  T theta = std::clamp(radius_d, theta_min, theta_max);
  const int max_iter = std::max(1, max_iterations);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    T theta_d = T(0);
    status = distortTheta<T>(model, theta, theta_d);
    if (status != StatusCode::OK)
    {
      return status;
    }
    const T f_theta = theta_d - radius_d;
    if (std::abs(f_theta) <= residual_tolerance)
    {
      theta_out = theta;
      return StatusCode::OK;
    }

    if (f_theta > T(0))
    {
      theta_max = theta;
      f_max = f_theta;
    }
    else
    {
      theta_min = theta;
      f_min = f_theta;
    }

    T derivative = T(0);
    status = distortThetaDerivative<T>(model, theta, derivative);
    if (status != StatusCode::OK)
    {
      return status;
    }

    T theta_next = std::numeric_limits<T>::quiet_NaN();
    if (std::abs(derivative) > PlaneTraits<T>::kEpsilon)
    {
      theta_next = theta - (f_theta / derivative);
    }
    if (!std::isfinite(theta_next) || theta_next <= theta_min || theta_next >= theta_max)
    {
      theta_next = T(0.5) * (theta_min + theta_max);
    }
    if (!std::isfinite(theta_next))
    {
      return StatusCode::NUMERIC_ERROR;
    }

    const T step = std::abs(theta_next - theta);
    theta = theta_next;
    if (step <= step_tolerance)
    {
      T theta_d_check = T(0);
      status = distortTheta<T>(model, theta, theta_d_check);
      if (status != StatusCode::OK)
      {
        return status;
      }
      if (std::abs(theta_d_check - radius_d) <= T(10) * residual_tolerance)
      {
        theta_out = theta;
        return StatusCode::OK;
      }
    }
  }

  theta_out = std::clamp(theta, T(0), projection.theta_max);
  return StatusCode::NON_CONVERGED;
}

// ---------------------------------------------------------------------------
// Polynomial-fisheye monotone-range scan
// ---------------------------------------------------------------------------
//
// Single source for every consumer of "how far is this KB/OpenCV-fisheye
// polynomial monotonically increasing and positive": theta_max derivation on
// fit write-back (updateThetaMax / defaultFisheyeThetaMax), the optimizer's
// mid-solve cap rescue (shrinkThetaMaxToPolynomialMonotoneRange), and the
// full-tier validator's monotone-cap certification. The scan grid constants
// are exposed so the validator can express its acceptance slack in whole
// sample steps of the SAME grid instead of a disconnected magic tolerance.

inline constexpr int kFisheyeThetaScanSamples = 512;

template <typename T>
constexpr T fisheyeThetaScanStart()
{
  return static_cast<T>(1e-3);
}

// Float pi minus a float margin, widened to T: reproduces the historical
// float implementation bit-for-bit (the scan cap is a float-origin value,
// same principle as the D47 pi bound in the validator). Computing from the
// double pi and narrowing would land 1 ULP lower at T = float.
template <typename T>
constexpr T fisheyeThetaScanCap()
{
  return static_cast<T>(constants::kPiF) - static_cast<T>(1e-4f);
}

template <typename T>
constexpr T fisheyeThetaScanStep()
{
  return (fisheyeThetaScanCap<T>() - fisheyeThetaScanStart<T>()) /
         static_cast<T>(kFisheyeThetaScanSamples);
}

/// Largest theta (on the scan grid) up to which the polynomial theta
/// distortion is finite, non-negative, and strictly increasing. Scans from
/// near-zero up to pi: starting higher (a historical pi/2 start) missed
/// non-monotonicity below pi/2 caused by large negative high-order
/// coefficients (e.g. overfitting a nearly-pinhole camera with a fisheye
/// polynomial). Resolution is one scan step (~pi/512 rad); callers comparing
/// a cap against this value must allow for that granularity.
template <typename T>
inline T estimateSafeThetaMaxForPolynomialFisheye(const DistortionModelT<T> &model)
{
  constexpr T theta_start = fisheyeThetaScanStart<T>();
  constexpr T theta_cap = fisheyeThetaScanCap<T>();

  T candidate = theta_start;
  for (int i = 1; i <= kFisheyeThetaScanSamples; ++i)
  {
    const T ratio = static_cast<T>(i) / static_cast<T>(kFisheyeThetaScanSamples);
    const T theta = theta_start + (theta_cap - theta_start) * ratio;

    T theta_d = T(0);
    if (distortTheta<T>(model, theta, theta_d) != StatusCode::OK || !std::isfinite(theta_d) || theta_d < T(0))
    {
      break;
    }

    T derivative = T(0);
    if (distortThetaDerivative<T>(model, theta, derivative) != StatusCode::OK || !std::isfinite(derivative) || derivative <= T(0))
    {
      break;
    }

    candidate = theta;
  }

  return candidate;
}

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__DISTORTION__ANGLE_IMPL_HPP
