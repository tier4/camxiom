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

#ifndef CAMXIOM__PROJECTION_TEMPLATE_HPP
#define CAMXIOM__PROJECTION_TEMPLATE_HPP

/// @file projection_template.hpp
/// @brief Templated forward-projection functions usable with any scalar type T.
///
/// These functions are pure math templates with no dependencies beyond the
/// standard library and the camxiom types header.  They work with:
///   - T = float / double          (normal evaluation)
///   - T = ceres::Jet<double, N>   (automatic differentiation)
///
/// This header intentionally does NOT depend on Ceres, Eigen, or OpenCV.
/// The only requirement is that T supports standard arithmetic operators and
/// the math functions listed in the "using" declarations inside each function.
///
/// Usage from a Ceres CostFunctor:
/// @code
///   #include "camxiom/projection_template.hpp"
///   namespace tpl = camxiom::projection_template;
///
///   template <class T>
///   bool operator()(const T* rvec, const T* tvec,
///                   const T* intrinsics, const T* dist, T* residual) const
///   {
///     // ... rotate & translate point ...
///     T u, v;
///     tpl::projectRadtan5(intrinsics, dist, x_cam, y_cam, z_cam, u, v);
///     residual[0] = T(observed_u) - u;
///     residual[1] = T(observed_v) - v;
///     return true;
///   }
/// @endcode

#include "camxiom/internal/angle_law.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace camxiom::projection_template
{

namespace detail
{
// Domain predicates depend on the scalar value, never on an AutoDiff tangent.
// This keeps the header independent of Ceres while supporting its Jet type.
template <typename T>
auto scalarValue(const T &value)
{
  if constexpr (std::is_arithmetic_v<T>)
    return value;
  else
    return scalarValue(value.a);
}

template <typename T>
constexpr double epsilon()
{
  using Scalar = decltype(scalarValue(T{}));
  return std::is_same_v<Scalar, float> ? 1e-8 : 1e-15;
}

template <typename T>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::cold]]
#endif
bool reject(T &u, T &v)
{
  u = T(0);
  v = T(0);
  return false;
}

template <typename T>
bool finish(T &u, T &v)
{
  if (!(std::isfinite(scalarValue(u)) & std::isfinite(scalarValue(v)))) return reject(u, v);
  return true;
}

// For an IEC 559 binary32/64 value, positive finite encodings are ordered
// integers. The unsigned range (floor, +inf) rejects NaNs, both infinities,
// signed zero and negatives in one comparison. floor is nonnegative and
// finite (zero for depth, epsilon for squared norms). memcpy is alias-safe;
// other scalar representations retain the standard floating-point predicate.
template <typename Scalar>
bool positiveFiniteAbove(Scalar value, Scalar floor)
{
  if constexpr (std::numeric_limits<Scalar>::is_iec559 && ((std::is_same_v<Scalar, float> && sizeof(Scalar) == 4) || (std::is_same_v<Scalar, double> && sizeof(Scalar) == 8)))
  {
    using Bits = std::conditional_t<sizeof(Scalar) == 4, std::uint32_t, std::uint64_t>;
    Bits bits, lower, upper;
    const Scalar inf = std::numeric_limits<Scalar>::infinity();
    std::memcpy(&bits, &value, sizeof(bits));
    std::memcpy(&lower, &floor, sizeof(lower));
    std::memcpy(&upper, &inf, sizeof(upper));
    return bits - lower - 1 < upper - lower - 1;
  }
  else
    return std::isfinite(value) && value > floor;
}

template <typename T>
bool withinTheta(ProjectionModelType type, const T &x, const T &y, const T &z, double cap)
{
  if (type == ProjectionModelType::PINHOLE) return true;
  if (!std::isfinite(cap) || cap <= 0 || cap > static_cast<double>(constants::kPiF)) return false;
  const auto sx = scalarValue(x), sy = scalarValue(y), sz = scalarValue(z);
  // The fisheye kernel checks the cap against the angle it computes for
  // projection, so it does not need a second atan2 here.
  if (isFisheyeProjection(type)) return true;
  if (cap >= constants::kPi) return true;
  return sz >= std::sqrt(sx * sx + sy * sy + sz * sz) * std::cos(cap);
}
}  // namespace detail

// ============================================================================
// Pinhole + plane distortion models
// ============================================================================

template <typename T>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
[[nodiscard]] inline bool
projectRadtan5(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  if (!detail::positiveFiniteAbove(
        detail::scalarValue(z_cam), decltype(detail::scalarValue(z_cam))(0)
      ))
  {
    return detail::reject(u_out, v_out);
  }

  const T &fx = intrinsics[0];
  const T &fy = intrinsics[1];
  const T &cx = intrinsics[2];
  const T &cy = intrinsics[3];

  const T &k1 = dist[0];
  const T &k2 = dist[1];
  const T &p1 = dist[2];
  const T &p2 = dist[3];
  const T &k3 = dist[4];

  const T xn = x_cam / z_cam;
  const T yn = y_cam / z_cam;
  const T r2 = xn * xn + yn * yn;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const T cdist = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;

  const T a1 = T(2.0) * xn * yn;
  const T a2 = r2 + T(2.0) * xn * xn;
  const T a3 = r2 + T(2.0) * yn * yn;

  const T xpd = xn * cdist + p1 * a1 + p2 * a2;
  const T ypd = yn * cdist + p1 * a3 + p2 * a1;

  u_out = fx * xpd + cx;
  v_out = fy * ypd + cy;
  return detail::finish(u_out, v_out);
}

template <typename T>
[[nodiscard]] bool projectRational8(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  if (!detail::positiveFiniteAbove(
        detail::scalarValue(z_cam), decltype(detail::scalarValue(z_cam))(0)
      ))
  {
    return detail::reject(u_out, v_out);
  }

  const T &fx = intrinsics[0];
  const T &fy = intrinsics[1];
  const T &cx = intrinsics[2];
  const T &cy = intrinsics[3];

  const T &k1 = dist[0];
  const T &k2 = dist[1];
  const T &p1 = dist[2];
  const T &p2 = dist[3];
  const T &k3 = dist[4];
  const T &k4 = dist[5];
  const T &k5 = dist[6];
  const T &k6 = dist[7];

  const T xn = x_cam / z_cam;
  const T yn = y_cam / z_cam;
  const T r2 = xn * xn + yn * yn;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const T num = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;
  const T den = T(1.0) + k4 * r2 + k5 * r4 + k6 * r6;
  if (!std::isfinite(detail::scalarValue(den)) || std::abs(detail::scalarValue(den)) <= detail::epsilon<T>())
    return detail::reject(u_out, v_out);
  const T cdist = num / den;

  const T a1 = T(2.0) * xn * yn;
  const T a2 = r2 + T(2.0) * xn * xn;
  const T a3 = r2 + T(2.0) * yn * yn;

  const T xpd = xn * cdist + p1 * a1 + p2 * a2;
  const T ypd = yn * cdist + p1 * a3 + p2 * a1;

  u_out = fx * xpd + cx;
  v_out = fy * ypd + cy;
  return detail::finish(u_out, v_out);
}

template <typename T>
[[nodiscard]] bool projectThinPrism12(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  if (!detail::positiveFiniteAbove(
        detail::scalarValue(z_cam), decltype(detail::scalarValue(z_cam))(0)
      ))
  {
    return detail::reject(u_out, v_out);
  }

  const T &fx = intrinsics[0];
  const T &fy = intrinsics[1];
  const T &cx = intrinsics[2];
  const T &cy = intrinsics[3];

  const T &k1 = dist[0];
  const T &k2 = dist[1];
  const T &p1 = dist[2];
  const T &p2 = dist[3];
  const T &k3 = dist[4];
  const T &k4 = dist[5];
  const T &k5 = dist[6];
  const T &k6 = dist[7];
  const T &s1 = dist[8];
  const T &s2 = dist[9];
  const T &s3 = dist[10];
  const T &s4 = dist[11];

  const T xn = x_cam / z_cam;
  const T yn = y_cam / z_cam;
  const T r2 = xn * xn + yn * yn;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const T num = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;
  const T den = T(1.0) + k4 * r2 + k5 * r4 + k6 * r6;
  if (!std::isfinite(detail::scalarValue(den)) || std::abs(detail::scalarValue(den)) <= detail::epsilon<T>())
    return detail::reject(u_out, v_out);
  const T cdist = num / den;

  const T a1 = T(2.0) * xn * yn;
  const T a2 = r2 + T(2.0) * xn * xn;
  const T a3 = r2 + T(2.0) * yn * yn;

  const T xpd = xn * cdist + p1 * a1 + p2 * a2 + s1 * r2 + s2 * r4;
  const T ypd = yn * cdist + p1 * a3 + p2 * a1 + s3 * r2 + s4 * r4;

  u_out = fx * xpd + cx;
  v_out = fy * ypd + cy;
  return detail::finish(u_out, v_out);
}

template <typename T>
[[nodiscard]] bool projectTilted14(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  if (!std::isfinite(detail::scalarValue(z_cam))) return detail::reject(u_out, v_out);
  using std::cos;
  using std::sin;

  const T eps_z = T(0);
  if (!(z_cam > eps_z))
  {
    return detail::reject(u_out, v_out);
  }

  const T &fx = intrinsics[0];
  const T &fy = intrinsics[1];
  const T &cx = intrinsics[2];
  const T &cy = intrinsics[3];

  const T &k1 = dist[0];
  const T &k2 = dist[1];
  const T &p1 = dist[2];
  const T &p2 = dist[3];
  const T &k3 = dist[4];
  const T &k4 = dist[5];
  const T &k5 = dist[6];
  const T &k6 = dist[7];
  const T &s1 = dist[8];
  const T &s2 = dist[9];
  const T &s3 = dist[10];
  const T &s4 = dist[11];
  const T &tau_x = dist[12];
  const T &tau_y = dist[13];

  const T xn = x_cam / z_cam;
  const T yn = y_cam / z_cam;
  const T r2 = xn * xn + yn * yn;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const T num = T(1.0) + k1 * r2 + k2 * r4 + k3 * r6;
  const T den = T(1.0) + k4 * r2 + k5 * r4 + k6 * r6;
  if (!std::isfinite(detail::scalarValue(den)) || std::abs(detail::scalarValue(den)) <= detail::epsilon<T>())
    return detail::reject(u_out, v_out);
  const T cdist = num / den;

  const T a1 = T(2.0) * xn * yn;
  const T a2 = r2 + T(2.0) * xn * xn;
  const T a3 = r2 + T(2.0) * yn * yn;

  const T xpd_no_tilt = xn * cdist + p1 * a1 + p2 * a2 + s1 * r2 + s2 * r4;
  const T ypd_no_tilt = yn * cdist + p1 * a3 + p2 * a1 + s3 * r2 + s4 * r4;

  const T c_tx = cos(tau_x);
  const T s_tx = sin(tau_x);
  const T c_ty = cos(tau_y);
  const T s_ty = sin(tau_y);

  const T r22 = c_ty * c_tx;
  if (std::abs(detail::scalarValue(r22)) <= (std::is_same_v<decltype(detail::scalarValue(r22)), float> ? 1e-8 : 1e-12))
  {
    u_out = fx * xpd_no_tilt + cx;
    v_out = fy * ypd_no_tilt + cy;
    return detail::finish(u_out, v_out);
  }

  const T r02 = -s_ty * c_tx;
  const T r12 = s_tx;

  const T mz = s_ty * xpd_no_tilt + (-c_ty * s_tx) * ypd_no_tilt + r22;

  const T r00 = c_ty;
  const T r01 = s_ty * s_tx;
  const T r10 = T(0.0);
  const T r11 = c_tx;
  const T r20 = s_ty;
  const T r21 = -c_ty * s_tx;

  const T t00 = r22 * r00 - r02 * r20;
  const T t01 = r22 * r01 - r02 * r21;
  const T t02 = r22 * r02 - r02 * r22;
  const T t10 = -r12 * r20 + r22 * r10;
  const T t11 = -r12 * r21 + r22 * r11;
  const T t12 = -r12 * r22 + r22 * r12;

  const T mx = t00 * xpd_no_tilt + t01 * ypd_no_tilt + t02;
  const T my = t10 * xpd_no_tilt + t11 * ypd_no_tilt + t12;
  if (!std::isfinite(detail::scalarValue(mz)) || std::abs(detail::scalarValue(mz)) <= detail::epsilon<T>())
    return detail::reject(u_out, v_out);
  const T inv_mz = T(1.0) / mz;
  const T x_tilt = mx * inv_mz;
  const T y_tilt = my * inv_mz;

  u_out = fx * x_tilt + cx;
  v_out = fy * y_tilt + cy;
  return detail::finish(u_out, v_out);
}

// ============================================================================
// Fisheye (theta-based) models
// ============================================================================

namespace detail
{
template <typename T, bool CheckTheta>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
inline bool
fisheyeCoordinates(const T &x, const T &y, const T &z, double cap, T &xy_norm, T &theta)
{
  using std::atan2;
  using std::sqrt;
  const auto sx = scalarValue(x), sy = scalarValue(y), sz = scalarValue(z);
  const auto xy_sq = sx * sx + sy * sy;
  const auto norm_sq = xy_sq + sz * sz;
  if (!positiveFiniteAbove(norm_sq, decltype(norm_sq)(epsilon<T>())) || (sz < 0 && std::sqrt(xy_sq) <= epsilon<T>()))
    return false;
  constexpr double kSafeEpsSq = 1e-24;
  xy_norm = sqrt(x * x + y * y + T(kSafeEpsSq));
  if constexpr (CheckTheta)
  {
    using Value = decltype(scalarValue(T{}));
    // An entire front hemisphere fits without an angular boundary test.
    // Use the rounded half-pi of the evaluation scalar, since float atan2
    // can return a value just above the double half-pi. Other scalar formats
    // retain the exact comparison below.
    if constexpr (std::is_same_v<Value, float> || std::is_same_v<Value, double>)
    {
      if (sz >= 0 && cap >= static_cast<double>(Value(constants::kHalfPi)))
      {
        theta = atan2(xy_norm, z);
        return true;
      }
    }
    // Decide before atan2 so the raw coordinates do not stay live across
    // its call on the normal path. Near the axis, preserve the unsmoothed
    // boundary even though projection uses a smooth norm for AutoDiff.
    using Scalar = decltype(sx * sx);
    if (xy_sq + Scalar(kSafeEpsSq) == xy_sq)
    {
      theta = atan2(xy_norm, z);
      return scalarValue(theta) <= cap;
    }
    if (!(std::atan2(std::sqrt(xy_sq), sz) <= cap)) return false;
  }
  theta = atan2(xy_norm, z);
  return true;
}

template <detail_impl::AngleLaw Law, bool CheckTheta, typename T>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
inline bool
projectFisheye(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out, double cap
)
{
  using std::sin;
  using std::tan;
  T xy_norm, theta;
  if (!fisheyeCoordinates<T, CheckTheta>(x_cam, y_cam, z_cam, cap, xy_norm, theta))
    return reject(u_out, v_out);
  T theta_d;
  using detail_impl::AngleLaw;
  if constexpr (Law == AngleLaw::KB4 || Law == AngleLaw::KB8)
  {
    const T t2 = theta * theta;
    const T t3 = t2 * theta;
    const T t5 = t3 * t2;
    const T t7 = t5 * t2;
    const T t9 = t7 * t2;
    theta_d = theta + dist[0] * t3 + dist[1] * t5 + dist[2] * t7 + dist[3] * t9;
    if constexpr (Law == AngleLaw::KB8)
    {
      const T t11 = t9 * t2;
      const T t13 = t11 * t2;
      const T t15 = t13 * t2;
      const T t17 = t15 * t2;
      theta_d = theta_d + dist[4] * t11 + dist[5] * t13 + dist[6] * t15 + dist[7] * t17;
    }
  }
  else if constexpr (Law == AngleLaw::EQUISOLID)
    theta_d = T(2.0) * sin(theta * T(0.5));
  else if constexpr (Law == AngleLaw::STEREOGRAPHIC)
    theta_d = T(2.0) * tan(theta * T(0.5));
  else if constexpr (Law == AngleLaw::ORTHOGRAPHIC)
  {
    if (z_cam < T(0)) return reject(u_out, v_out);
    theta_d = sin(theta);
  }
  else
    theta_d = theta;
  const T scale = theta_d / xy_norm;
  u_out = intrinsics[0] * (scale * x_cam) + intrinsics[2];
  v_out = intrinsics[1] * (scale * y_cam) + intrinsics[3];
  return finish(u_out, v_out);
}
}  // namespace detail

template <typename T>
[[nodiscard]] bool projectKB4(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::KB4, false>(
    intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}

template <typename T>
[[nodiscard]] bool projectKB8(
  const T *intrinsics, const T *dist, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::KB8, false>(
    intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}

template <typename T>
[[nodiscard]] bool projectEquidistant(
  const T *intrinsics, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::EQUIDISTANT, false>(
    intrinsics, static_cast<const T *>(nullptr), x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}

template <typename T>
[[nodiscard]] bool projectEquisolid(
  const T *intrinsics, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::EQUISOLID, false>(
    intrinsics, static_cast<const T *>(nullptr), x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}

template <typename T>
[[nodiscard]] bool projectStereographic(
  const T *intrinsics, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::STEREOGRAPHIC, false>(
    intrinsics, static_cast<const T *>(nullptr), x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}

template <typename T>
[[nodiscard]] bool projectOrthographic(
  const T *intrinsics, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  return detail::projectFisheye<detail_impl::AngleLaw::ORTHOGRAPHIC, false>(
    intrinsics, static_cast<const T *>(nullptr), x_cam, y_cam, z_cam, u_out, v_out, constants::kPi
  );
}
namespace detail
{
// Wide-angle projections share the same normalized-plane distortion models as
// pinhole. Dispatch by the raw helpers' coefficient-count convention; the
// checked generic entry point additionally validates the distortion enum.
template <typename T>
bool projectPlaneDistortion(
  const T *intrinsics, const T *dist, int dist_count, const T &x, const T &y, T &u, T &v
)
{
  switch (dist_count)
  {
    case 0:
      u = intrinsics[0] * x + intrinsics[2];
      v = intrinsics[1] * y + intrinsics[3];
      return finish(u, v);
    case 4: {
      const T dist5[5] = {dist[0], dist[1], dist[2], dist[3], T(0)};
      return projectRadtan5(intrinsics, dist5, x, y, T(1), u, v);
    }
    case 5:
      return projectRadtan5(intrinsics, dist, x, y, T(1), u, v);
    case 8:
      return projectRational8(intrinsics, dist, x, y, T(1), u, v);
    case 12:
      return projectThinPrism12(intrinsics, dist, x, y, T(1), u, v);
    case 14:
      return projectTilted14(intrinsics, dist, x, y, T(1), u, v);
    default:
      return reject(u, v);
  }
}
}  // namespace detail

// ============================================================================
// Omnidirectional (Mei) model
// ============================================================================

template <typename T>
[[nodiscard]] bool projectOmnidirectional(
  const T *intrinsics, const T &xi, const T *dist, int dist_count, const T &x_cam, const T &y_cam,
  const T &z_cam, T &u_out, T &v_out
)
{
  if (!std::isfinite(detail::scalarValue(z_cam))) return detail::reject(u_out, v_out);
  using std::sqrt;

  const T ray_norm = sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam);
  const T eps = T(detail::epsilon<T>());
  if (!std::isfinite(detail::scalarValue(ray_norm)) || !(ray_norm > eps))
    return detail::reject(u_out, v_out);
  const T denom = z_cam + xi * ray_norm;
  if (!std::isfinite(detail::scalarValue(denom)) || !(denom > eps))
  {
    return detail::reject(u_out, v_out);
  }

  // Injectivity limit (see the runtime omni forward): ray_norm + xi*z > 0;
  // binding for xi > 1 where denom > 0 holds for every direction.
  if (!(ray_norm + xi * z_cam > eps))
  {
    return detail::reject(u_out, v_out);
  }

  return detail::projectPlaneDistortion(
    intrinsics, dist, dist_count, x_cam / denom, y_cam / denom, u_out, v_out
  );
}

// ============================================================================
// Double Sphere model
// ============================================================================

template <typename T>
[[nodiscard]] bool projectDoubleSphere(
  const T *intrinsics, const T &xi, const T &alpha, const T *dist, int dist_count, const T &x_cam,
  const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  if (!std::isfinite(detail::scalarValue(z_cam))) return detail::reject(u_out, v_out);
  using std::sqrt;

  const T d1 = sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam);
  const T eps = T(detail::epsilon<T>());
  if (!std::isfinite(detail::scalarValue(d1)) || !(d1 > eps))
  {
    return detail::reject(u_out, v_out);
  }

  const T xi_d1_z = xi * d1 + z_cam;
  const T d2 = sqrt(x_cam * x_cam + y_cam * y_cam + xi_d1_z * xi_d1_z);
  if (!std::isfinite(detail::scalarValue(d2))) return detail::reject(u_out, v_out);
  const T denom = alpha * d2 + (T(1.0) - alpha) * xi_d1_z;
  if (!std::isfinite(detail::scalarValue(denom)) || !(denom > eps))
  {
    return detail::reject(u_out, v_out);
  }

  // DS bijectivity (Usenko 2018 eq. 43-45): z > -w2 * d1, matching the
  // runtime forward in detail::computeDsForward. denom > 0 alone holds for
  // every direction once alpha > 0.5.
  const T w1 = (alpha <= T(0.5)) ? alpha / (T(1.0) - alpha) : (T(1.0) - alpha) / alpha;
  const T w2 = (w1 + xi) / sqrt(T(2.0) * w1 * xi + xi * xi + T(1.0));
  if (z_cam <= -w2 * d1)
  {
    return detail::reject(u_out, v_out);
  }

  return detail::projectPlaneDistortion(
    intrinsics, dist, dist_count, x_cam / denom, y_cam / denom, u_out, v_out
  );
}

// ============================================================================
// EUCM (Extended Unified Camera Model)
// ============================================================================

template <typename T>
[[nodiscard]] bool projectEucm(
  const T *intrinsics, const T &alpha, const T &beta, const T *dist, int dist_count, const T &x_cam,
  const T &y_cam, const T &z_cam, T &u_out, T &v_out
)
{
  if (!std::isfinite(detail::scalarValue(z_cam))) return detail::reject(u_out, v_out);
  using std::sqrt;

  const T d = sqrt(beta * (x_cam * x_cam + y_cam * y_cam) + z_cam * z_cam);
  const T eps = T(detail::epsilon<T>());
  if (!std::isfinite(detail::scalarValue(d)) || !(d > eps)) return detail::reject(u_out, v_out);
  const T denom = alpha * d + (T(1.0) - alpha) * z_cam;
  if (!std::isfinite(detail::scalarValue(denom)) || !(denom > eps))
  {
    return detail::reject(u_out, v_out);
  }

  // EUCM FOV validity (unconditional; at alpha = 0 it degenerates to
  // z > 0), matching the runtime forward. denom > 0 alone holds for every
  // direction once alpha > 0.5.
  const T w = (alpha <= T(0.5)) ? alpha / (T(1.0) - alpha) : (T(1.0) - alpha) / alpha;
  if (z_cam <= -w * d)
  {
    return detail::reject(u_out, v_out);
  }

  return detail::projectPlaneDistortion(
    intrinsics, dist, dist_count, x_cam / denom, y_cam / denom, u_out, v_out
  );
}

// ============================================================================
// Generic dispatch by model type
// ============================================================================

namespace detail
{
template <typename T>
[[nodiscard]] bool dispatchParametric(
  ProjectionModelType proj_type, DistortionModelType dist_type, const T *intrinsics, const T *dist,
  int dist_count, const T &proj_xi, const T &proj_alpha, const T &proj_beta, const T &x_cam,
  const T &y_cam, const T &z_cam, T &u_out, T &v_out, double theta_max
)
{
  switch (proj_type)
  {
    case ProjectionModelType::PINHOLE: {
      switch (dist_type)
      {
        case DistortionModelType::NONE: {
          if (!std::isfinite(scalarValue(z_cam)) || !(z_cam > T(0)))
          {
            return detail::reject(u_out, v_out);
          }
          const T xn = x_cam / z_cam;
          const T yn = y_cam / z_cam;
          u_out = intrinsics[0] * xn + intrinsics[2];
          v_out = intrinsics[1] * yn + intrinsics[3];
          return detail::finish(u_out, v_out);
        }
        case DistortionModelType::RATIONAL8:
          return projectRational8(intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out);
        case DistortionModelType::THIN_PRISM12:
          return projectThinPrism12(intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out);
        case DistortionModelType::TILTED14:
          return projectTilted14(intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out);
        case DistortionModelType::RADTAN4: {
          // Runtime also accepts the legacy 5-entry RADTAN4 layout, where
          // the fifth entry is evaluated as k3. Only pad a true 4-wide block.
          if (dist_count == 5)
            return projectRadtan5(intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out);
          // The caller's dist parameter block really is 4 wide for RADTAN4
          // (e.g. the Ceres block in the AUTO_DIFF PnP cost): reading a
          // fifth coefficient would run off the block. Widen with k3 = 0.
          const T dist5[5] = {dist[0], dist[1], dist[2], dist[3], T(0.0)};
          return projectRadtan5(intrinsics, dist5, x_cam, y_cam, z_cam, u_out, v_out);
        }
        case DistortionModelType::RADTAN5:
          return projectRadtan5(intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out);
        case DistortionModelType::KB4:
        case DistortionModelType::KB8:
        case DistortionModelType::UNKNOWN:
          break;
      }
      // Unknown plane type: fail rather than silently pretending it is
      // RADTAN5 with whatever happens to sit in the buffer.
      return detail::reject(u_out, v_out);
    }

    case ProjectionModelType::EQUIDISTANT:
    case ProjectionModelType::EQUISOLID:
    case ProjectionModelType::STEREOGRAPHIC:
    case ProjectionModelType::ORTHOGRAPHIC: {
      using detail_impl::AngleLaw;
      switch (detail_impl::angleLawOf(proj_type, dist_type))
      {
        case AngleLaw::EQUIDISTANT:
          return detail::projectFisheye<AngleLaw::EQUIDISTANT, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::KB4:
          return detail::projectFisheye<AngleLaw::KB4, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::KB8:
          return detail::projectFisheye<AngleLaw::KB8, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::EQUISOLID:
          return detail::projectFisheye<AngleLaw::EQUISOLID, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::STEREOGRAPHIC:
          return detail::projectFisheye<AngleLaw::STEREOGRAPHIC, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::ORTHOGRAPHIC:
          return detail::projectFisheye<AngleLaw::ORTHOGRAPHIC, true>(
            intrinsics, dist, x_cam, y_cam, z_cam, u_out, v_out, theta_max
          );
        case AngleLaw::INVALID:
          break;
      }
      // See the model-based dispatch: silently projecting an unknown
      // type as equidistant would bias the optimiser.
      return detail::reject(u_out, v_out);
    }

    case ProjectionModelType::OMNIDIRECTIONAL: {
      return projectOmnidirectional(
        intrinsics, proj_xi, dist, dist_count, x_cam, y_cam, z_cam, u_out, v_out
      );
    }

    case ProjectionModelType::DOUBLE_SPHERE: {
      return projectDoubleSphere(
        intrinsics, proj_xi, proj_alpha, dist, dist_count, x_cam, y_cam, z_cam, u_out, v_out
      );
    }

    case ProjectionModelType::EUCM: {
      return projectEucm(
        intrinsics, proj_alpha, proj_beta, dist, dist_count, x_cam, y_cam, z_cam, u_out, v_out
      );
    }

    case ProjectionModelType::UNKNOWN:
      break;
  }
  return detail::reject(u_out, v_out);
}

}  // namespace detail

namespace detail
{
// Model-only checks are independent of the ray and of AutoDiff derivatives.
// Every active intrinsic/distortion coefficient participates in arithmetic:
// NaN/Inf necessarily reaches an output or a checked intermediate (norm,
// rational/tilt denominator), and finish() rejects it. Do not scan those
// inputs again.
template <typename T>
bool validParameters(
  ProjectionModelType proj_type, DistortionModelType dist_type, const T *intrinsics, int dist_count,
  const T &proj_xi, const T &proj_alpha, const T &proj_beta, double eps
)
{
  if (std::abs(detail::scalarValue(intrinsics[0])) <= eps || std::abs(detail::scalarValue(intrinsics[1])) <= eps)
    return false;
  int required = 0;
  switch (dist_type)
  {
    case DistortionModelType::RADTAN4:
    case DistortionModelType::KB4:
      required = 4;
      break;
    case DistortionModelType::RADTAN5:
      required = 5;
      break;
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::KB8:
      required = 8;
      break;
    case DistortionModelType::THIN_PRISM12:
      required = 12;
      break;
    case DistortionModelType::TILTED14:
      required = 14;
      break;
    case DistortionModelType::NONE:
      break;
    case DistortionModelType::UNKNOWN:
      return false;
  }
  if (dist_count != required)
  {
    const bool legacy_radtan4 = dist_type == DistortionModelType::RADTAN4 && dist_count == 5;
    if (!legacy_radtan4) return false;
  }
  if (proj_type == ProjectionModelType::OMNIDIRECTIONAL || proj_type == ProjectionModelType::DOUBLE_SPHERE || proj_type == ProjectionModelType::EUCM)
  {
    switch (dist_type)
    {
      case DistortionModelType::NONE:
      case DistortionModelType::RADTAN4:
      case DistortionModelType::RADTAN5:
      case DistortionModelType::RATIONAL8:
      case DistortionModelType::THIN_PRISM12:
      case DistortionModelType::TILTED14:
        break;
      default:
        return false;
    }
  }
  const auto xi = detail::scalarValue(proj_xi);
  const auto alpha = detail::scalarValue(proj_alpha);
  const auto beta = detail::scalarValue(proj_beta);
  if ((proj_type == ProjectionModelType::OMNIDIRECTIONAL && !std::isfinite(xi)) || (proj_type == ProjectionModelType::DOUBLE_SPHERE && (!std::isfinite(xi) || xi <= -1 || xi >= 1 || !std::isfinite(alpha) || alpha < 0 || alpha > 1)) || (proj_type == ProjectionModelType::EUCM && (!std::isfinite(beta) || beta <= 0 || !std::isfinite(alpha) || alpha < 0 || alpha > 1)))
    return false;
  return true;
}
}  // namespace detail

/// Checked parametric projection. theta_max is an explicit, non-optimised
/// evaluation cap; callers may choose a cap distinct from observation selection.
/// Rational denominators follow the runtime contract: either sign is allowed,
/// but a zero/near-zero denominator is rejected. Stricter physical-branch
/// selection remains the caller's policy.
template <typename T>
[[nodiscard]] bool projectGenericParametric(
  ProjectionModelType proj_type, DistortionModelType dist_type, const T *intrinsics, const T *dist,
  int dist_count, const T &proj_xi, const T &proj_alpha, const T &proj_beta, const T &x_cam,
  const T &y_cam, const T &z_cam, T &u_out, T &v_out, double theta_max = constants::kPi
)
{
  if (!intrinsics || dist_count < 0 || dist_count > 14 || (dist_count && !dist) || !detail::withinTheta(proj_type, x_cam, y_cam, z_cam, theta_max))
    return detail::reject(u_out, v_out);
  if (!detail::validParameters(
        proj_type, dist_type, intrinsics, dist_count, proj_xi, proj_alpha, proj_beta,
        detail::epsilon<T>()
      ))
    return detail::reject(u_out, v_out);
  return detail::dispatchParametric(
    proj_type, dist_type, intrinsics, dist, dist_count, proj_xi, proj_alpha, proj_beta, x_cam,
    y_cam, z_cam, u_out, v_out, theta_max
  );
}

namespace detail
{
// Fixed-width copies avoid constructing 14 scalars (or Jets) for every model.
// Keep this dispatch visible to the caller so loop-invariant model checks and
// conversions can be hoisted without introducing a mutable model cache.
template <DistortionModelType Type, typename T, typename Scalar>
bool projectModel(
  const CameraModelT<Scalar> &model, const T *intrinsics, const T &x, const T &y, const T &z, T &u,
  T &v
)
{
  using detail_impl::AngleLaw;
  constexpr bool kb = Type == DistortionModelType::KB4 || Type == DistortionModelType::KB8;
  constexpr int count =
    (Type == DistortionModelType::RADTAN4 || Type == DistortionModelType::RADTAN5) ? 5
    : Type == DistortionModelType::KB4                                             ? 4
    : (Type == DistortionModelType::KB8 || Type == DistortionModelType::RATIONAL8) ? 8
    : Type == DistortionModelType::THIN_PRISM12                                    ? 12
    : Type == DistortionModelType::TILTED14                                        ? 14
                                                                                   : 0;
  const int size = model.distortion.count;
  if constexpr (Type == DistortionModelType::RADTAN4)
  {
    if (size != 4 && size != 5) return reject(u, v);
  }
  else if (size != count)
    return reject(u, v);
  T dist[count ? count : 1];
  for (int i = 0; i < count; ++i) dist[i] = i < size ? T(model.distortion.coeffs[i]) : T(0);
  if constexpr (kb)
  {
    if (model.projection.type != ProjectionModelType::EQUIDISTANT) return reject(u, v);
    constexpr AngleLaw law = Type == DistortionModelType::KB4 ? AngleLaw::KB4 : AngleLaw::KB8;
    return projectFisheye<law, true>(intrinsics, dist, x, y, z, u, v, model.projection.theta_max);
  }
  else if constexpr (Type == DistortionModelType::NONE)
  {
    const double cap = model.projection.theta_max;
    switch (model.projection.type)
    {
      case ProjectionModelType::EQUIDISTANT:
        return projectFisheye<AngleLaw::EQUIDISTANT, true>(intrinsics, dist, x, y, z, u, v, cap);
      case ProjectionModelType::EQUISOLID:
        return projectFisheye<AngleLaw::EQUISOLID, true>(intrinsics, dist, x, y, z, u, v, cap);
      case ProjectionModelType::STEREOGRAPHIC:
        return projectFisheye<AngleLaw::STEREOGRAPHIC, true>(intrinsics, dist, x, y, z, u, v, cap);
      case ProjectionModelType::ORTHOGRAPHIC:
        return projectFisheye<AngleLaw::ORTHOGRAPHIC, true>(intrinsics, dist, x, y, z, u, v, cap);
      default:
        break;
    }
    if (!std::isfinite(scalarValue(z)) || !(z > T(0))) return reject(u, v);
    u = intrinsics[0] * (x / z) + intrinsics[2];
    v = intrinsics[1] * (y / z) + intrinsics[3];
    return finish(u, v);
  }
  else
  {
    if (model.projection.type != ProjectionModelType::PINHOLE) return reject(u, v);
    if constexpr (Type == DistortionModelType::RADTAN4 || Type == DistortionModelType::RADTAN5)
      return projectRadtan5(intrinsics, dist, x, y, z, u, v);
    else if constexpr (Type == DistortionModelType::RATIONAL8)
      return projectRational8(intrinsics, dist, x, y, z, u, v);
    else if constexpr (Type == DistortionModelType::THIN_PRISM12)
      return projectThinPrism12(intrinsics, dist, x, y, z, u, v);
    else
      return projectTilted14(intrinsics, dist, x, y, z, u, v);
  }
}
}  // namespace detail

template <typename T, typename Scalar>
#if defined(__GNUC__) || defined(__clang__)
[[gnu::always_inline]]
#endif
[[nodiscard]] inline bool
projectGeneric(
  const CameraModelT<Scalar> &model, const T &x_cam, const T &y_cam, const T &z_cam, T &u_out,
  T &v_out
)
{
  if (model.distortion.count > 14 ||
      !detail::withinTheta(model.projection.type, x_cam, y_cam, z_cam, model.projection.theta_max))
    return detail::reject(u_out, v_out);
  const T intrinsics[4] = {
    T(model.intrinsics.fx), T(model.intrinsics.fy), T(model.intrinsics.cx), T(model.intrinsics.cy)};
  if (std::abs(detail::scalarValue(intrinsics[0])) <= detail::epsilon<T>() || std::abs(detail::scalarValue(intrinsics[1])) <= detail::epsilon<T>())
    return detail::reject(u_out, v_out);
  if (model.projection.type == ProjectionModelType::PINHOLE || isFisheyeProjection(model.projection.type))
  {
    switch (model.distortion.type)
    {
      case DistortionModelType::NONE:
        return detail::projectModel<DistortionModelType::NONE>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::RADTAN4:
        return detail::projectModel<DistortionModelType::RADTAN4>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::RADTAN5:
        return detail::projectModel<DistortionModelType::RADTAN5>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::RATIONAL8:
        return detail::projectModel<DistortionModelType::RATIONAL8>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::THIN_PRISM12:
        return detail::projectModel<DistortionModelType::THIN_PRISM12>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::TILTED14:
        return detail::projectModel<DistortionModelType::TILTED14>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::KB4:
        return detail::projectModel<DistortionModelType::KB4>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      case DistortionModelType::KB8:
        return detail::projectModel<DistortionModelType::KB8>(
          model, intrinsics, x_cam, y_cam, z_cam, u_out, v_out
        );
      default:
        return detail::reject(u_out, v_out);
    }
  }
  T dist[14];
  for (int i = 0; i < model.distortion.count; ++i) dist[i] = T(model.distortion.coeffs[i]);
  return projectGenericParametric(
    model.projection.type, model.distortion.type, intrinsics, dist, model.distortion.count,
    T(model.projection.xi), T(model.projection.alpha), T(model.projection.beta), x_cam, y_cam,
    z_cam, u_out, v_out, model.projection.theta_max
  );
}

}  // namespace camxiom::projection_template

#endif  // CAMXIOM__PROJECTION_TEMPLATE_HPP
