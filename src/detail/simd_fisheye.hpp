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

#ifndef CAMXIOM__DETAIL__SIMD_FISHEYE_HPP
#define CAMXIOM__DETAIL__SIMD_FISHEYE_HPP

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"

#include <cmath>
#include <limits>

#if defined(__AVX2__)
#define CAMXIOM_HAS_AVX2 1
#include <immintrin.h>
#endif

#if defined(__SSE2__)
#define CAMXIOM_HAS_SSE2 1
#include <emmintrin.h>
#include <xmmintrin.h>
#elif defined(__aarch64__)
// The 4-wide kernels below also run natively on AArch64 through a minimal
// __m128 -> NEON mapping (see simd_neon_compat.hpp for scope and caveats).
#define CAMXIOM_HAS_SSE2 1
#include "detail/simd_neon_compat.hpp"
#endif

namespace camxiom::detail
{

#ifdef CAMXIOM_HAS_SSE2

inline __m128 selectSse(const __m128 base, const __m128 value, const __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, value), _mm_andnot_ps(mask, base));
}

inline __m128 absSseFwd(const __m128 values)
{
  const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  return _mm_and_ps(values, sign_mask);
}

inline __m128 finiteMaskSseFwd(const __m128 values)
{
  const __m128 max_value = _mm_set1_ps((std::numeric_limits<float>::max)());
  return _mm_cmple_ps(absSseFwd(values), max_value);
}

inline __m128 atanSse4HighPrecision(__m128 x)
{
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 tan_pi_over_8 = _mm_set1_ps(0.41421356237f);
  const __m128 tan_3pi_over_8 = _mm_set1_ps(2.41421356237f);
  const __m128 pi_over_4 = _mm_set1_ps(0.78539816339f);
  const __m128 pi_over_2 = _mm_set1_ps(1.57079632679f);

  const __m128 p0 = _mm_set1_ps(8.05374449538e-2f);
  const __m128 p1 = _mm_set1_ps(-1.38776856032e-1f);
  const __m128 p2 = _mm_set1_ps(1.99777106478e-1f);
  const __m128 p3 = _mm_set1_ps(-3.33329491539e-1f);

  const __m128 sign = _mm_and_ps(x, sign_mask);
  __m128 ax = _mm_andnot_ps(sign_mask, x);
  __m128 base_angle = _mm_setzero_ps();

  const __m128 mask_large = _mm_cmpgt_ps(ax, tan_3pi_over_8);
  const __m128 reduced_large = _mm_sub_ps(_mm_setzero_ps(), _mm_div_ps(one, ax));
  ax = selectSse(ax, reduced_large, mask_large);
  base_angle = selectSse(base_angle, pi_over_2, mask_large);

  const __m128 mask_medium = _mm_cmpgt_ps(ax, tan_pi_over_8);
  const __m128 reduced_medium = _mm_div_ps(_mm_sub_ps(ax, one), _mm_add_ps(ax, one));
  ax = selectSse(ax, reduced_medium, mask_medium);
  base_angle = _mm_add_ps(base_angle, _mm_and_ps(mask_medium, pi_over_4));

  const __m128 z = _mm_mul_ps(ax, ax);
  __m128 poly = _mm_add_ps(_mm_mul_ps(p0, z), p1);
  poly = _mm_add_ps(_mm_mul_ps(poly, z), p2);
  poly = _mm_add_ps(_mm_mul_ps(poly, z), p3);
  const __m128 atan_core = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(poly, z), ax), ax);
  const __m128 result = _mm_add_ps(base_angle, atan_core);
  return _mm_xor_ps(result, sign);
}

inline __m128 atan2Sse4HighPrecision(const __m128 y, const __m128 x)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 abs_x = _mm_andnot_ps(sign_mask, x);
  const __m128 abs_y = _mm_andnot_ps(sign_mask, y);
  const __m128 min_v = _mm_min_ps(abs_x, abs_y);
  const __m128 max_v = _mm_max_ps(abs_x, abs_y);
  const __m128 eps = _mm_set1_ps(1e-20f);
  const __m128 ratio = _mm_div_ps(min_v, _mm_add_ps(max_v, eps));

  __m128 angle = atanSse4HighPrecision(ratio);
  const __m128 half_pi = _mm_set1_ps(1.57079632679f);
  const __m128 pi = _mm_set1_ps(3.14159265359f);

  const __m128 swap_mask = _mm_cmpgt_ps(abs_y, abs_x);
  angle = selectSse(angle, _mm_sub_ps(half_pi, angle), swap_mask);

  const __m128 x_negative = _mm_cmplt_ps(x, _mm_setzero_ps());
  angle = selectSse(angle, _mm_sub_ps(pi, angle), x_negative);

  const __m128 y_negative = _mm_cmplt_ps(y, _mm_setzero_ps());
  angle = selectSse(angle, _mm_xor_ps(angle, sign_mask), y_negative);
  return angle;
}

/// SSE sqrt approximation using _mm_sqrt_ps (exact HW instruction)
inline __m128 sqrtSse4(__m128 x) { return _mm_sqrt_ps(x); }

/// Process 4 fisheye forward projections simultaneously using SSE.
/// Handles: polynomial angle distortion (OPENCV_FISHEYE4 / KB4 / EQUIDISTANT), intrinsics.
/// Returns bitmask of valid points (0-15).
inline int rayToPixelFisheyeSse4(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m128 xs = _mm_set_ps(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m128 ys = _mm_set_ps(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m128 zs = _mm_set_ps(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m128 zero = _mm_setzero_ps();
  const __m128 finite_xyz =
    _mm_and_ps(finiteMaskSseFwd(xs), _mm_and_ps(finiteMaskSseFwd(ys), finiteMaskSseFwd(zs)));

  // xy_norm = sqrt(x² + y²)
  const __m128 xy_sq = _mm_add_ps(_mm_mul_ps(xs, xs), _mm_mul_ps(ys, ys));
  const __m128 ray_norm_sq = _mm_add_ps(xy_sq, _mm_mul_ps(zs, zs));
  const __m128 xy_norm = sqrtSse4(xy_sq);

  // theta = atan2(xy_norm, z)
  const __m128 theta = atan2Sse4HighPrecision(xy_norm, zs);

  // theta_max check
  const __m128 t_max = _mm_set1_ps(model.projection.theta_max);
  const __m128 theta_non_negative = _mm_cmpge_ps(theta, zero);
  const __m128 fov_valid = _mm_cmple_ps(theta, t_max);
  const __m128 eps = _mm_set1_ps(1e-7f);
  const __m128 axis_mask = _mm_cmple_ps(xy_norm, eps);
  const __m128 z_negative = _mm_cmplt_ps(zs, zero);
  const __m128 singular_back_axis = _mm_and_ps(axis_mask, z_negative);
  const __m128 norm_valid = _mm_cmpgt_ps(ray_norm_sq, eps);
  __m128 valid = _mm_and_ps(
    _mm_and_ps(finite_xyz, norm_valid),
    _mm_andnot_ps(singular_back_axis, _mm_and_ps(theta_non_negative, fov_valid))
  );
  const int valid_mask = _mm_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // Distort theta: theta_d = theta * (1 + k1*t² + k2*t⁴ + k3*t⁶ + k4*t⁸)
  // This works for both OPENCV_FISHEYE4 and KB4 since derivative formula is shared.
  // For KB4: theta_d = theta + k1*t³ + k2*t⁵ + k3*t⁷ + k4*t⁹
  //        = theta * (1 + k1*t² + k2*t⁴ + k3*t⁶ + k4*t⁸)  (same form)
  const auto &c = model.distortion.coeffs;
  const __m128 k1 = _mm_set1_ps(c[0]);
  const __m128 k2 = _mm_set1_ps(c[1]);
  const __m128 k3 = _mm_set1_ps(c[2]);
  const __m128 k4 = _mm_set1_ps(c[3]);
  const __m128 t2 = _mm_mul_ps(theta, theta);
  const __m128 t4 = _mm_mul_ps(t2, t2);
  const __m128 t6 = _mm_mul_ps(t4, t2);
  const __m128 t8 = _mm_mul_ps(t4, t4);
  const __m128 one = _mm_set1_ps(1.0f);

  __m128 poly = _mm_add_ps(one, _mm_mul_ps(k1, t2));
  poly = _mm_add_ps(poly, _mm_mul_ps(k2, t4));
  poly = _mm_add_ps(poly, _mm_mul_ps(k3, t6));
  poly = _mm_add_ps(poly, _mm_mul_ps(k4, t8));
  const __m128 theta_d = _mm_mul_ps(theta, poly);

  // scale = theta_d / xy_norm (safe with epsilon guard)
  const __m128 safe_norm = _mm_max_ps(xy_norm, eps);
  const __m128 scale = _mm_div_ps(theta_d, safe_norm);

  // x_d = X * scale, y_d = Y * scale
  const __m128 x_d = _mm_mul_ps(xs, scale);
  const __m128 y_d = _mm_mul_ps(ys, scale);

  // Apply intrinsics
  const __m128 fx = _mm_set1_ps(model.intrinsics.fx);
  const __m128 fy = _mm_set1_ps(model.intrinsics.fy);
  const __m128 cx = _mm_set1_ps(model.intrinsics.cx);
  const __m128 cy = _mm_set1_ps(model.intrinsics.cy);
  const __m128 sk = _mm_set1_ps(model.intrinsics.skew);

  __m128 u_v = _mm_add_ps(_mm_add_ps(_mm_mul_ps(fx, x_d), _mm_mul_ps(sk, y_d)), cx);
  __m128 v_v = _mm_add_ps(_mm_mul_ps(fy, y_d), cy);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseFwd(u_v), finiteMaskSseFwd(v_v)));

  u_v = _mm_and_ps(u_v, valid);
  v_v = _mm_and_ps(v_v, valid);

  _mm_storeu_ps(u_out, u_v);
  _mm_storeu_ps(v_out, v_v);

  return _mm_movemask_ps(valid);
}

#ifdef CAMXIOM_HAS_AVX2

inline __m256 selectAvx(const __m256 base, const __m256 value, const __m256 mask)
{
  return _mm256_or_ps(_mm256_and_ps(mask, value), _mm256_andnot_ps(mask, base));
}

inline __m256 absAvxFwd(const __m256 values)
{
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(values, sign_mask);
}

inline __m256 finiteMaskAvxFwd(const __m256 values)
{
  const __m256 max_value = _mm256_set1_ps((std::numeric_limits<float>::max)());
  return _mm256_cmp_ps(absAvxFwd(values), max_value, _CMP_LE_OQ);
}

inline __m256 atanAvx8HighPrecision(__m256 x)
{
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 tan_pi_over_8 = _mm256_set1_ps(0.41421356237f);
  const __m256 tan_3pi_over_8 = _mm256_set1_ps(2.41421356237f);
  const __m256 pi_over_4 = _mm256_set1_ps(0.78539816339f);
  const __m256 pi_over_2 = _mm256_set1_ps(1.57079632679f);

  const __m256 p0 = _mm256_set1_ps(8.05374449538e-2f);
  const __m256 p1 = _mm256_set1_ps(-1.38776856032e-1f);
  const __m256 p2 = _mm256_set1_ps(1.99777106478e-1f);
  const __m256 p3 = _mm256_set1_ps(-3.33329491539e-1f);

  const __m256 sign = _mm256_and_ps(x, sign_mask);
  __m256 ax = _mm256_andnot_ps(sign_mask, x);
  __m256 base_angle = _mm256_setzero_ps();

  const __m256 mask_large = _mm256_cmp_ps(ax, tan_3pi_over_8, _CMP_GT_OQ);
  const __m256 reduced_large = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_div_ps(one, ax));
  ax = selectAvx(ax, reduced_large, mask_large);
  base_angle = selectAvx(base_angle, pi_over_2, mask_large);

  const __m256 mask_medium = _mm256_cmp_ps(ax, tan_pi_over_8, _CMP_GT_OQ);
  const __m256 reduced_medium = _mm256_div_ps(_mm256_sub_ps(ax, one), _mm256_add_ps(ax, one));
  ax = selectAvx(ax, reduced_medium, mask_medium);
  base_angle = _mm256_add_ps(base_angle, _mm256_and_ps(mask_medium, pi_over_4));

  const __m256 z = _mm256_mul_ps(ax, ax);
  __m256 poly = _mm256_add_ps(_mm256_mul_ps(p0, z), p1);
  poly = _mm256_add_ps(_mm256_mul_ps(poly, z), p2);
  poly = _mm256_add_ps(_mm256_mul_ps(poly, z), p3);
  const __m256 atan_core = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(poly, z), ax), ax);
  const __m256 result = _mm256_add_ps(base_angle, atan_core);
  return _mm256_xor_ps(result, sign);
}

inline __m256 atan2Avx8HighPrecision(const __m256 y, const __m256 x)
{
  const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  const __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
  const __m256 abs_y = _mm256_andnot_ps(sign_mask, y);
  const __m256 min_v = _mm256_min_ps(abs_x, abs_y);
  const __m256 max_v = _mm256_max_ps(abs_x, abs_y);
  const __m256 eps = _mm256_set1_ps(1e-20f);
  const __m256 ratio = _mm256_div_ps(min_v, _mm256_add_ps(max_v, eps));

  __m256 angle = atanAvx8HighPrecision(ratio);
  const __m256 half_pi = _mm256_set1_ps(1.57079632679f);
  const __m256 pi = _mm256_set1_ps(3.14159265359f);

  const __m256 swap_mask = _mm256_cmp_ps(abs_y, abs_x, _CMP_GT_OQ);
  angle = selectAvx(angle, _mm256_sub_ps(half_pi, angle), swap_mask);

  const __m256 x_negative = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
  angle = selectAvx(angle, _mm256_sub_ps(pi, angle), x_negative);

  const __m256 y_negative = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_LT_OQ);
  angle = selectAvx(angle, _mm256_xor_ps(angle, sign_mask), y_negative);
  return angle;
}

inline int rayToPixelFisheyeAvx8(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m256 xs = _mm256_set_ps(
    rays_xyz[21], rays_xyz[18], rays_xyz[15], rays_xyz[12], rays_xyz[9], rays_xyz[6], rays_xyz[3],
    rays_xyz[0]
  );
  const __m256 ys = _mm256_set_ps(
    rays_xyz[22], rays_xyz[19], rays_xyz[16], rays_xyz[13], rays_xyz[10], rays_xyz[7], rays_xyz[4],
    rays_xyz[1]
  );
  const __m256 zs = _mm256_set_ps(
    rays_xyz[23], rays_xyz[20], rays_xyz[17], rays_xyz[14], rays_xyz[11], rays_xyz[8], rays_xyz[5],
    rays_xyz[2]
  );

  const __m256 zero = _mm256_setzero_ps();
  const __m256 eps = _mm256_set1_ps(1e-7f);
  const __m256 finite_xyz =
    _mm256_and_ps(finiteMaskAvxFwd(xs), _mm256_and_ps(finiteMaskAvxFwd(ys), finiteMaskAvxFwd(zs)));

  const __m256 xy_sq = _mm256_add_ps(_mm256_mul_ps(xs, xs), _mm256_mul_ps(ys, ys));
  const __m256 ray_norm_sq = _mm256_add_ps(xy_sq, _mm256_mul_ps(zs, zs));
  const __m256 xy_norm = _mm256_sqrt_ps(xy_sq);
  const __m256 theta = atan2Avx8HighPrecision(xy_norm, zs);

  const __m256 t_max = _mm256_set1_ps(model.projection.theta_max);
  const __m256 theta_non_negative = _mm256_cmp_ps(theta, zero, _CMP_GE_OQ);
  const __m256 fov_valid = _mm256_cmp_ps(theta, t_max, _CMP_LE_OQ);
  const __m256 axis_mask = _mm256_cmp_ps(xy_norm, eps, _CMP_LE_OQ);
  const __m256 z_negative = _mm256_cmp_ps(zs, zero, _CMP_LT_OQ);
  const __m256 singular_back_axis = _mm256_and_ps(axis_mask, z_negative);
  const __m256 norm_valid = _mm256_cmp_ps(ray_norm_sq, eps, _CMP_GT_OQ);
  __m256 valid = _mm256_and_ps(
    _mm256_and_ps(finite_xyz, norm_valid),
    _mm256_andnot_ps(singular_back_axis, _mm256_and_ps(theta_non_negative, fov_valid))
  );
  const int valid_mask = _mm256_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const auto &c = model.distortion.coeffs;
  const __m256 k1 = _mm256_set1_ps(c[0]);
  const __m256 k2 = _mm256_set1_ps(c[1]);
  const __m256 k3 = _mm256_set1_ps(c[2]);
  const __m256 k4 = _mm256_set1_ps(c[3]);
  const __m256 t2 = _mm256_mul_ps(theta, theta);
  const __m256 t4 = _mm256_mul_ps(t2, t2);
  const __m256 t6 = _mm256_mul_ps(t4, t2);
  const __m256 t8 = _mm256_mul_ps(t4, t4);
  const __m256 one = _mm256_set1_ps(1.0f);

  __m256 poly = _mm256_add_ps(one, _mm256_mul_ps(k1, t2));
  poly = _mm256_add_ps(poly, _mm256_mul_ps(k2, t4));
  poly = _mm256_add_ps(poly, _mm256_mul_ps(k3, t6));
  poly = _mm256_add_ps(poly, _mm256_mul_ps(k4, t8));
  const __m256 theta_d = _mm256_mul_ps(theta, poly);

  const __m256 safe_norm = _mm256_max_ps(xy_norm, eps);
  const __m256 scale = _mm256_div_ps(theta_d, safe_norm);

  const __m256 x_d = _mm256_mul_ps(xs, scale);
  const __m256 y_d = _mm256_mul_ps(ys, scale);

  const __m256 fx = _mm256_set1_ps(model.intrinsics.fx);
  const __m256 fy = _mm256_set1_ps(model.intrinsics.fy);
  const __m256 cx = _mm256_set1_ps(model.intrinsics.cx);
  const __m256 cy = _mm256_set1_ps(model.intrinsics.cy);
  const __m256 sk = _mm256_set1_ps(model.intrinsics.skew);

  __m256 u_v = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(fx, x_d), _mm256_mul_ps(sk, y_d)), cx);
  __m256 v_v = _mm256_add_ps(_mm256_mul_ps(fy, y_d), cy);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxFwd(u_v), finiteMaskAvxFwd(v_v)));

  u_v = _mm256_and_ps(u_v, valid);
  v_v = _mm256_and_ps(v_v, valid);
  _mm256_storeu_ps(u_out, u_v);
  _mm256_storeu_ps(v_out, v_v);
  return _mm256_movemask_ps(valid);
}

#endif  // CAMXIOM_HAS_AVX2

/// Process 4 omni forward projections simultaneously using SSE.
/// Handles: plane distortion (RadTan5/Rational8/ThinPrism), intrinsics.
/// Returns bitmask of valid points (0-15).
inline int rayToPixelOmniSse4(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m128 xs = _mm_set_ps(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m128 ys = _mm_set_ps(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m128 zs = _mm_set_ps(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m128 zero = _mm_setzero_ps();
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 eps = _mm_set1_ps(1e-7f);
  const __m128 finite_xyz =
    _mm_and_ps(finiteMaskSseFwd(xs), _mm_and_ps(finiteMaskSseFwd(ys), finiteMaskSseFwd(zs)));

  // r = sqrt(X² + Y² + Z²)
  const __m128 r_sq =
    _mm_add_ps(_mm_add_ps(_mm_mul_ps(xs, xs), _mm_mul_ps(ys, ys)), _mm_mul_ps(zs, zs));
  const __m128 r = sqrtSse4(r_sq);
  const __m128 norm_valid = _mm_and_ps(_mm_cmpgt_ps(r, eps), finiteMaskSseFwd(r));

  // denom = Z + xi * r
  const __m128 xi = _mm_set1_ps(model.projection.xi);
  const __m128 denom = _mm_add_ps(zs, _mm_mul_ps(xi, r));
  const __m128 denom_valid = _mm_and_ps(_mm_cmpgt_ps(denom, eps), finiteMaskSseFwd(denom));
  // Injectivity limit (see the scalar omni impl): r + xi*Z > 0; binding for xi > 1.
  const __m128 mono_valid = _mm_cmpgt_ps(_mm_add_ps(r, _mm_mul_ps(xi, zs)), eps);
  __m128 valid =
    _mm_and_ps(finite_xyz, _mm_and_ps(norm_valid, _mm_and_ps(denom_valid, mono_valid)));

  // theta_max contract (see the scalar impl); skipped at the wide-angle
  // default of pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m128 cos_tm = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(zs, _mm_mul_ps(r, cos_tm)));
  }
  const int valid_mask = _mm_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // x_n = X/denom, y_n = Y/denom
  const __m128 safe_denom = selectSse(one, denom, denom_valid);
  const __m128 inv_denom = _mm_div_ps(one, safe_denom);
  const __m128 x_n = _mm_mul_ps(xs, inv_denom);
  const __m128 y_n = _mm_mul_ps(ys, inv_denom);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseFwd(x_n), finiteMaskSseFwd(y_n)));

  // Plane distortion (same as pinhole SSE kernel)
  const __m128 r2_v = _mm_add_ps(_mm_mul_ps(x_n, x_n), _mm_mul_ps(y_n, y_n));
  const __m128 r4_v = _mm_mul_ps(r2_v, r2_v);
  const __m128 r6_v = _mm_mul_ps(r4_v, r2_v);

  const auto &c = model.distortion.coeffs;
  const __m128 k1 = _mm_set1_ps(c[0]);
  const __m128 k2 = _mm_set1_ps(c[1]);
  const __m128 p1 = _mm_set1_ps(c[2]);
  const __m128 p2 = _mm_set1_ps(c[3]);
  const __m128 k3 = _mm_set1_ps(c[4]);
  const __m128 two = _mm_set1_ps(2.0f);

  __m128 rad_num = _mm_add_ps(one, _mm_mul_ps(k1, r2_v));
  rad_num = _mm_add_ps(rad_num, _mm_mul_ps(k2, r4_v));
  rad_num = _mm_add_ps(rad_num, _mm_mul_ps(k3, r6_v));
  __m128 radial = rad_num;

  if (model.distortion.is_rational)
  {
    const __m128 k4v = _mm_set1_ps(c[5]);
    const __m128 k5v = _mm_set1_ps(c[6]);
    const __m128 k6v = _mm_set1_ps(c[7]);
    __m128 rad_den = _mm_add_ps(one, _mm_mul_ps(k4v, r2_v));
    rad_den = _mm_add_ps(rad_den, _mm_mul_ps(k5v, r4_v));
    rad_den = _mm_add_ps(rad_den, _mm_mul_ps(k6v, r6_v));
    const __m128 den_valid =
      _mm_and_ps(finiteMaskSseFwd(rad_den), _mm_cmpgt_ps(absSseFwd(rad_den), eps));
    valid = _mm_and_ps(valid, den_valid);
    const __m128 safe_rad_den = selectSse(one, rad_den, den_valid);
    radial = _mm_div_ps(rad_num, safe_rad_den);
  }

  const __m128 xy = _mm_mul_ps(x_n, y_n);
  const __m128 xx = _mm_mul_ps(x_n, x_n);
  const __m128 yy = _mm_mul_ps(y_n, y_n);
  const __m128 x_tan = _mm_add_ps(
    _mm_mul_ps(_mm_mul_ps(two, p1), xy), _mm_mul_ps(p2, _mm_add_ps(r2_v, _mm_mul_ps(two, xx)))
  );
  const __m128 y_tan = _mm_add_ps(
    _mm_mul_ps(p1, _mm_add_ps(r2_v, _mm_mul_ps(two, yy))), _mm_mul_ps(_mm_mul_ps(two, p2), xy)
  );

  __m128 x_d = _mm_add_ps(_mm_mul_ps(x_n, radial), x_tan);
  __m128 y_d = _mm_add_ps(_mm_mul_ps(y_n, radial), y_tan);

  if (model.distortion.has_thin_prism)
  {
    const __m128 s1 = _mm_set1_ps(c[8]);
    const __m128 s2 = _mm_set1_ps(c[9]);
    const __m128 s3 = _mm_set1_ps(c[10]);
    const __m128 s4 = _mm_set1_ps(c[11]);
    x_d = _mm_add_ps(x_d, _mm_add_ps(_mm_mul_ps(s1, r2_v), _mm_mul_ps(s2, r4_v)));
    y_d = _mm_add_ps(y_d, _mm_add_ps(_mm_mul_ps(s3, r2_v), _mm_mul_ps(s4, r4_v)));
  }

  // Apply intrinsics
  const __m128 fx = _mm_set1_ps(model.intrinsics.fx);
  const __m128 fy = _mm_set1_ps(model.intrinsics.fy);
  const __m128 cx = _mm_set1_ps(model.intrinsics.cx);
  const __m128 cy = _mm_set1_ps(model.intrinsics.cy);
  const __m128 sk = _mm_set1_ps(model.intrinsics.skew);

  __m128 u_v = _mm_add_ps(_mm_add_ps(_mm_mul_ps(fx, x_d), _mm_mul_ps(sk, y_d)), cx);
  __m128 v_v = _mm_add_ps(_mm_mul_ps(fy, y_d), cy);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseFwd(u_v), finiteMaskSseFwd(v_v)));

  u_v = _mm_and_ps(u_v, valid);
  v_v = _mm_and_ps(v_v, valid);

  _mm_storeu_ps(u_out, u_v);
  _mm_storeu_ps(v_out, v_v);

  return _mm_movemask_ps(valid);
}

int rayToPixelBatchFisheyeSse(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#ifdef CAMXIOM_HAS_AVX2
int rayToPixelBatchFisheyeAvx2(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);
#endif

int rayToPixelBatchOmniSse(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#ifdef CAMXIOM_HAS_AVX2

/// Process 8 omnidirectional forward projections using AVX2.
inline int rayToPixelOmniAvx8(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m256 xs = _mm256_set_ps(
    rays_xyz[21], rays_xyz[18], rays_xyz[15], rays_xyz[12], rays_xyz[9], rays_xyz[6], rays_xyz[3],
    rays_xyz[0]
  );
  const __m256 ys = _mm256_set_ps(
    rays_xyz[22], rays_xyz[19], rays_xyz[16], rays_xyz[13], rays_xyz[10], rays_xyz[7], rays_xyz[4],
    rays_xyz[1]
  );
  const __m256 zs = _mm256_set_ps(
    rays_xyz[23], rays_xyz[20], rays_xyz[17], rays_xyz[14], rays_xyz[11], rays_xyz[8], rays_xyz[5],
    rays_xyz[2]
  );

  const __m256 zero = _mm256_setzero_ps();
  const __m256 eps = _mm256_set1_ps(1e-7f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 finite_xyz =
    _mm256_and_ps(finiteMaskAvxFwd(xs), _mm256_and_ps(finiteMaskAvxFwd(ys), finiteMaskAvxFwd(zs)));

  const __m256 r_sq = _mm256_add_ps(
    _mm256_add_ps(_mm256_mul_ps(xs, xs), _mm256_mul_ps(ys, ys)), _mm256_mul_ps(zs, zs)
  );
  const __m256 r = _mm256_sqrt_ps(r_sq);
  const __m256 norm_valid = _mm256_and_ps(_mm256_cmp_ps(r, eps, _CMP_GT_OQ), finiteMaskAvxFwd(r));

  const __m256 xi = _mm256_set1_ps(model.projection.xi);
  const __m256 denom = _mm256_add_ps(zs, _mm256_mul_ps(xi, r));
  const __m256 denom_valid =
    _mm256_and_ps(_mm256_cmp_ps(denom, eps, _CMP_GT_OQ), finiteMaskAvxFwd(denom));
  // Injectivity limit (see the scalar omni impl): r + xi*Z > 0; binding for xi > 1.
  const __m256 mono_valid = _mm256_cmp_ps(_mm256_add_ps(r, _mm256_mul_ps(xi, zs)), eps, _CMP_GT_OQ);

  __m256 valid =
    _mm256_and_ps(finite_xyz, _mm256_and_ps(norm_valid, _mm256_and_ps(denom_valid, mono_valid)));

  // theta_max contract (see the scalar impl); skipped at the wide-angle
  // default of pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m256 cos_tm = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(zs, _mm256_mul_ps(r, cos_tm), _CMP_GE_OQ));
  }
  const int valid_mask = _mm256_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const __m256 safe_denom = selectAvx(one, denom, denom_valid);
  const __m256 inv_denom = _mm256_div_ps(one, safe_denom);
  const __m256 x_n = _mm256_mul_ps(xs, inv_denom);
  const __m256 y_n = _mm256_mul_ps(ys, inv_denom);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxFwd(x_n), finiteMaskAvxFwd(y_n)));

  const __m256 r2_v = _mm256_add_ps(_mm256_mul_ps(x_n, x_n), _mm256_mul_ps(y_n, y_n));
  const __m256 r4_v = _mm256_mul_ps(r2_v, r2_v);
  const __m256 r6_v = _mm256_mul_ps(r4_v, r2_v);

  const auto &c = model.distortion.coeffs;
  const __m256 k1 = _mm256_set1_ps(c[0]);
  const __m256 k2 = _mm256_set1_ps(c[1]);
  const __m256 p1 = _mm256_set1_ps(c[2]);
  const __m256 p2 = _mm256_set1_ps(c[3]);
  const __m256 k3 = _mm256_set1_ps(c[4]);
  const __m256 two = _mm256_set1_ps(2.0f);

  __m256 rad_num = _mm256_add_ps(one, _mm256_mul_ps(k1, r2_v));
  rad_num = _mm256_add_ps(rad_num, _mm256_mul_ps(k2, r4_v));
  rad_num = _mm256_add_ps(rad_num, _mm256_mul_ps(k3, r6_v));
  __m256 radial = rad_num;

  if (model.distortion.is_rational)
  {
    const __m256 k4v = _mm256_set1_ps(c[5]);
    const __m256 k5v = _mm256_set1_ps(c[6]);
    const __m256 k6v = _mm256_set1_ps(c[7]);
    __m256 rad_den = _mm256_add_ps(one, _mm256_mul_ps(k4v, r2_v));
    rad_den = _mm256_add_ps(rad_den, _mm256_mul_ps(k5v, r4_v));
    rad_den = _mm256_add_ps(rad_den, _mm256_mul_ps(k6v, r6_v));
    const __m256 den_valid =
      _mm256_and_ps(finiteMaskAvxFwd(rad_den), _mm256_cmp_ps(absAvxFwd(rad_den), eps, _CMP_GT_OQ));
    valid = _mm256_and_ps(valid, den_valid);
    const __m256 safe_rad_den = selectAvx(one, rad_den, den_valid);
    radial = _mm256_div_ps(rad_num, safe_rad_den);
  }

  const __m256 xy = _mm256_mul_ps(x_n, y_n);
  const __m256 xx = _mm256_mul_ps(x_n, x_n);
  const __m256 yy = _mm256_mul_ps(y_n, y_n);
  const __m256 x_tan = _mm256_add_ps(
    _mm256_mul_ps(_mm256_mul_ps(two, p1), xy),
    _mm256_mul_ps(p2, _mm256_add_ps(r2_v, _mm256_mul_ps(two, xx)))
  );
  const __m256 y_tan = _mm256_add_ps(
    _mm256_mul_ps(p1, _mm256_add_ps(r2_v, _mm256_mul_ps(two, yy))),
    _mm256_mul_ps(_mm256_mul_ps(two, p2), xy)
  );

  __m256 x_d = _mm256_add_ps(_mm256_mul_ps(x_n, radial), x_tan);
  __m256 y_d = _mm256_add_ps(_mm256_mul_ps(y_n, radial), y_tan);

  if (model.distortion.has_thin_prism)
  {
    const __m256 s1 = _mm256_set1_ps(c[8]);
    const __m256 s2 = _mm256_set1_ps(c[9]);
    const __m256 s3 = _mm256_set1_ps(c[10]);
    const __m256 s4 = _mm256_set1_ps(c[11]);
    x_d = _mm256_add_ps(x_d, _mm256_add_ps(_mm256_mul_ps(s1, r2_v), _mm256_mul_ps(s2, r4_v)));
    y_d = _mm256_add_ps(y_d, _mm256_add_ps(_mm256_mul_ps(s3, r2_v), _mm256_mul_ps(s4, r4_v)));
  }

  const __m256 fx = _mm256_set1_ps(model.intrinsics.fx);
  const __m256 fy = _mm256_set1_ps(model.intrinsics.fy);
  const __m256 cx = _mm256_set1_ps(model.intrinsics.cx);
  const __m256 cy = _mm256_set1_ps(model.intrinsics.cy);
  const __m256 sk = _mm256_set1_ps(model.intrinsics.skew);

  __m256 u_v = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(fx, x_d), _mm256_mul_ps(sk, y_d)), cx);
  __m256 v_v = _mm256_add_ps(_mm256_mul_ps(fy, y_d), cy);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxFwd(u_v), finiteMaskAvxFwd(v_v)));

  u_v = _mm256_and_ps(u_v, valid);
  v_v = _mm256_and_ps(v_v, valid);

  _mm256_storeu_ps(u_out, u_v);
  _mm256_storeu_ps(v_out, v_v);

  return _mm256_movemask_ps(valid);
}

int rayToPixelBatchOmniAvx2(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#endif  // CAMXIOM_HAS_AVX2

#endif  // CAMXIOM_HAS_SSE2

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_FISHEYE_HPP
