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

#ifndef CAMXIOM__DETAIL__SIMD_DS_EUCM_HPP
#define CAMXIOM__DETAIL__SIMD_DS_EUCM_HPP

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

inline __m128 absSseDs(const __m128 values)
{
  return _mm_and_ps(values, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

inline __m128 finiteMaskSseDs(const __m128 values)
{
  const __m128 max_value = _mm_set1_ps((std::numeric_limits<float>::max)());
  return _mm_cmple_ps(absSseDs(values), max_value);
}

/// Process 4 DoubleSphere forward projections simultaneously using SSE.
/// Handles: plane distortion (RadTan5/Rational8/ThinPrism), intrinsics.
/// Returns bitmask of valid points (0-15).
inline int rayToPixelDsphSse4(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m128 xs = _mm_set_ps(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m128 ys = _mm_set_ps(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m128 zs = _mm_set_ps(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m128 zero = _mm_setzero_ps();
  const __m128 eps = _mm_set1_ps(1e-7f);
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 finite_xyz =
    _mm_and_ps(finiteMaskSseDs(xs), _mm_and_ps(finiteMaskSseDs(ys), finiteMaskSseDs(zs)));

  const __m128 xi_v = _mm_set1_ps(model.projection.xi);
  const __m128 alpha_v = _mm_set1_ps(model.projection.alpha);
  const __m128 one_m_alpha = _mm_sub_ps(one, alpha_v);

  // d1 = sqrt(X² + Y² + Z²)
  const __m128 xx = _mm_mul_ps(xs, xs);
  const __m128 yy = _mm_mul_ps(ys, ys);
  const __m128 zz = _mm_mul_ps(zs, zs);
  const __m128 d1_sq = _mm_add_ps(_mm_add_ps(xx, yy), zz);
  const __m128 d1 = _mm_sqrt_ps(d1_sq);

  // d1 > eps
  const __m128 d1_valid = _mm_and_ps(_mm_cmpgt_ps(d1, eps), finiteMaskSseDs(d1));
  __m128 valid = _mm_and_ps(finite_xyz, d1_valid);

  int valid_mask = _mm_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // xi_d1_z = xi * d1 + Z
  const __m128 xi_d1_z = _mm_add_ps(_mm_mul_ps(xi_v, d1), zs);

  // d2 = sqrt(X² + Y² + (xi*d1+Z)²)
  const __m128 d2 = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(xx, yy), _mm_mul_ps(xi_d1_z, xi_d1_z)));
  valid = _mm_and_ps(valid, _mm_cmpgt_ps(d2, eps));

  // denom = alpha * d2 + (1-alpha) * (xi*d1+Z); denom > eps is a numerical
  // guard only — for alpha > 0.5 it holds for every direction.
  const __m128 denom = _mm_add_ps(_mm_mul_ps(alpha_v, d2), _mm_mul_ps(one_m_alpha, xi_d1_z));
  const __m128 denom_valid = _mm_and_ps(finiteMaskSseDs(denom), _mm_cmpgt_ps(denom, eps));
  valid = _mm_and_ps(valid, denom_valid);

  // DS bijectivity (Usenko 2018 eq. 43-45): Z > -w2 * d1, matching the scalar
  // detail::computeDsForward check.
  {
    const float xi_f = model.projection.xi;
    const float alpha_f = model.projection.alpha;
    const float w1_f = (alpha_f <= 0.5f) ? alpha_f / (1.0f - alpha_f) : (1.0f - alpha_f) / alpha_f;
    const float w2_f = (w1_f + xi_f) / std::sqrt(2.0f * w1_f * xi_f + xi_f * xi_f + 1.0f);
    const __m128 neg_w2 = _mm_set1_ps(-w2_f);
    valid = _mm_and_ps(valid, _mm_cmpgt_ps(zs, _mm_mul_ps(neg_w2, d1)));
  }

  // theta_max contract (see the scalar impl); skipped at the default pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m128 cos_tm = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(zs, _mm_mul_ps(d1, cos_tm)));
  }

  valid_mask = _mm_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // x_n = X/denom, y_n = Y/denom
  const __m128 safe_denom =
    _mm_or_ps(_mm_and_ps(denom_valid, denom), _mm_andnot_ps(denom_valid, one));
  const __m128 inv_denom = _mm_div_ps(one, safe_denom);
  const __m128 x_n = _mm_mul_ps(xs, inv_denom);
  const __m128 y_n = _mm_mul_ps(ys, inv_denom);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseDs(x_n), finiteMaskSseDs(y_n)));

  // Plane distortion (shared with pinhole/omni SSE)
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
      _mm_and_ps(finiteMaskSseDs(rad_den), _mm_cmpgt_ps(absSseDs(rad_den), eps));
    valid = _mm_and_ps(valid, den_valid);
    const __m128 safe_rad_den =
      _mm_or_ps(_mm_and_ps(den_valid, rad_den), _mm_andnot_ps(den_valid, one));
    radial = _mm_div_ps(rad_num, safe_rad_den);
  }

  const __m128 xy = _mm_mul_ps(x_n, y_n);
  const __m128 xn_sq = _mm_mul_ps(x_n, x_n);
  const __m128 yn_sq = _mm_mul_ps(y_n, y_n);
  const __m128 x_tan = _mm_add_ps(
    _mm_mul_ps(_mm_mul_ps(two, p1), xy), _mm_mul_ps(p2, _mm_add_ps(r2_v, _mm_mul_ps(two, xn_sq)))
  );
  const __m128 y_tan = _mm_add_ps(
    _mm_mul_ps(p1, _mm_add_ps(r2_v, _mm_mul_ps(two, yn_sq))), _mm_mul_ps(_mm_mul_ps(two, p2), xy)
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
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseDs(u_v), finiteMaskSseDs(v_v)));

  u_v = _mm_and_ps(u_v, valid);
  v_v = _mm_and_ps(v_v, valid);

  _mm_storeu_ps(u_out, u_v);
  _mm_storeu_ps(v_out, v_v);

  return _mm_movemask_ps(valid);
}

/// Process 4 EUCM forward projections simultaneously using SSE.
/// Returns bitmask of valid points (0-15).
inline int rayToPixelEucmSse4(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  const __m128 xs = _mm_set_ps(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m128 ys = _mm_set_ps(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m128 zs = _mm_set_ps(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m128 zero = _mm_setzero_ps();
  const __m128 eps = _mm_set1_ps(1e-7f);
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 finite_xyz =
    _mm_and_ps(finiteMaskSseDs(xs), _mm_and_ps(finiteMaskSseDs(ys), finiteMaskSseDs(zs)));

  const float alpha_f = model.projection.alpha;
  const float beta_f = model.projection.beta;
  const __m128 alpha_v = _mm_set1_ps(alpha_f);
  const __m128 beta_v = _mm_set1_ps(beta_f);
  const __m128 one_m_alpha = _mm_sub_ps(one, alpha_v);

  const __m128 xx = _mm_mul_ps(xs, xs);
  const __m128 yy = _mm_mul_ps(ys, ys);
  const __m128 zz = _mm_mul_ps(zs, zs);

  // d = sqrt(beta*(X²+Y²) + Z²)
  const __m128 d = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(beta_v, _mm_add_ps(xx, yy)), zz));
  __m128 valid = _mm_and_ps(finite_xyz, _mm_and_ps(_mm_cmpgt_ps(d, eps), finiteMaskSseDs(d)));

  // denom = alpha*d + (1-alpha)*Z
  const __m128 denom = _mm_add_ps(_mm_mul_ps(alpha_v, d), _mm_mul_ps(one_m_alpha, zs));
  const __m128 denom_valid = _mm_and_ps(finiteMaskSseDs(denom), _mm_cmpgt_ps(absSseDs(denom), eps));
  valid = _mm_and_ps(valid, denom_valid);

  // FOV check (unconditional; at alpha = 0 it degenerates to Z > 0),
  // matching the scalar forward. Gating on alpha used to let alpha ~ 0
  // project rear points onto mirrored pixels.
  {
    const float w_f =
      (alpha_f <= 0.5f) ? (alpha_f / (1.0f - alpha_f)) : ((1.0f - alpha_f) / alpha_f);
    const __m128 neg_w = _mm_set1_ps(-w_f);
    valid = _mm_and_ps(valid, _mm_cmpgt_ps(zs, _mm_mul_ps(neg_w, d)));
  }

  // theta_max contract; d is beta-weighted, so use the true norm. Skipped at
  // the default pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m128 norm = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(xx, yy), zz));
    const __m128 cos_tm = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(zs, _mm_mul_ps(norm, cos_tm)));
  }

  int valid_mask = _mm_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // x_n = X/denom, y_n = Y/denom
  const __m128 safe_denom =
    _mm_or_ps(_mm_and_ps(denom_valid, denom), _mm_andnot_ps(denom_valid, one));
  const __m128 inv_denom = _mm_div_ps(one, safe_denom);
  const __m128 x_n = _mm_mul_ps(xs, inv_denom);
  const __m128 y_n = _mm_mul_ps(ys, inv_denom);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseDs(x_n), finiteMaskSseDs(y_n)));

  // Plane distortion
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
      _mm_and_ps(finiteMaskSseDs(rad_den), _mm_cmpgt_ps(absSseDs(rad_den), eps));
    valid = _mm_and_ps(valid, den_valid);
    const __m128 safe_rad_den =
      _mm_or_ps(_mm_and_ps(den_valid, rad_den), _mm_andnot_ps(den_valid, one));
    radial = _mm_div_ps(rad_num, safe_rad_den);
  }

  const __m128 xy = _mm_mul_ps(x_n, y_n);
  const __m128 xn_sq = _mm_mul_ps(x_n, x_n);
  const __m128 yn_sq = _mm_mul_ps(y_n, y_n);
  const __m128 x_tan = _mm_add_ps(
    _mm_mul_ps(_mm_mul_ps(two, p1), xy), _mm_mul_ps(p2, _mm_add_ps(r2_v, _mm_mul_ps(two, xn_sq)))
  );
  const __m128 y_tan = _mm_add_ps(
    _mm_mul_ps(p1, _mm_add_ps(r2_v, _mm_mul_ps(two, yn_sq))), _mm_mul_ps(_mm_mul_ps(two, p2), xy)
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
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseDs(u_v), finiteMaskSseDs(v_v)));

  u_v = _mm_and_ps(u_v, valid);
  v_v = _mm_and_ps(v_v, valid);

  _mm_storeu_ps(u_out, u_v);
  _mm_storeu_ps(v_out, v_v);

  return _mm_movemask_ps(valid);
}

/// SIMD-accelerated DoubleSphere batch forward projection.
/// Tilt models fall back to scalar.
int rayToPixelBatchDsphSse(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

/// SIMD-accelerated EUCM batch forward projection.
/// Tilt models fall back to scalar.
int rayToPixelBatchEucmSse(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#ifdef CAMXIOM_HAS_AVX2

inline __m256 absAvxDs(const __m256 v)
{
  return _mm256_and_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
}

inline __m256 finiteMaskAvxDs(const __m256 values)
{
  const __m256 max_value = _mm256_set1_ps((std::numeric_limits<float>::max)());
  return _mm256_cmp_ps(absAvxDs(values), max_value, _CMP_LE_OQ);
}

/// Process 8 DoubleSphere forward projections using AVX2.
inline int rayToPixelDsphAvx8(
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
    _mm256_and_ps(finiteMaskAvxDs(xs), _mm256_and_ps(finiteMaskAvxDs(ys), finiteMaskAvxDs(zs)));

  const __m256 xi_v = _mm256_set1_ps(model.projection.xi);
  const __m256 alpha_v = _mm256_set1_ps(model.projection.alpha);
  const __m256 one_m_alpha = _mm256_sub_ps(one, alpha_v);

  const __m256 xx = _mm256_mul_ps(xs, xs);
  const __m256 yy = _mm256_mul_ps(ys, ys);
  const __m256 zz = _mm256_mul_ps(zs, zs);
  const __m256 d1_sq = _mm256_add_ps(_mm256_add_ps(xx, yy), zz);
  const __m256 d1 = _mm256_sqrt_ps(d1_sq);

  const __m256 d1_valid = _mm256_and_ps(_mm256_cmp_ps(d1, eps, _CMP_GT_OQ), finiteMaskAvxDs(d1));
  __m256 valid = _mm256_and_ps(finite_xyz, d1_valid);

  int valid_mask = _mm256_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const __m256 xi_d1_z = _mm256_add_ps(_mm256_mul_ps(xi_v, d1), zs);
  const __m256 d2 =
    _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(xx, yy), _mm256_mul_ps(xi_d1_z, xi_d1_z)));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(d2, eps, _CMP_GT_OQ));

  // denom > eps is a numerical guard only — for alpha > 0.5 it holds for
  // every direction.
  const __m256 denom =
    _mm256_add_ps(_mm256_mul_ps(alpha_v, d2), _mm256_mul_ps(one_m_alpha, xi_d1_z));
  const __m256 denom_valid =
    _mm256_and_ps(finiteMaskAvxDs(denom), _mm256_cmp_ps(denom, eps, _CMP_GT_OQ));
  valid = _mm256_and_ps(valid, denom_valid);

  // DS bijectivity (Usenko 2018 eq. 43-45): Z > -w2 * d1, matching the scalar
  // detail::computeDsForward check.
  {
    const float xi_f = model.projection.xi;
    const float alpha_f = model.projection.alpha;
    const float w1_f = (alpha_f <= 0.5f) ? alpha_f / (1.0f - alpha_f) : (1.0f - alpha_f) / alpha_f;
    const float w2_f = (w1_f + xi_f) / std::sqrt(2.0f * w1_f * xi_f + xi_f * xi_f + 1.0f);
    const __m256 neg_w2 = _mm256_set1_ps(-w2_f);
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(zs, _mm256_mul_ps(neg_w2, d1), _CMP_GT_OQ));
  }

  // theta_max contract (see the scalar impl); skipped at the default pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m256 cos_tm = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(zs, _mm256_mul_ps(d1, cos_tm), _CMP_GE_OQ));
  }

  valid_mask = _mm256_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const __m256 safe_denom =
    _mm256_or_ps(_mm256_and_ps(denom_valid, denom), _mm256_andnot_ps(denom_valid, one));
  const __m256 inv_denom = _mm256_div_ps(one, safe_denom);
  const __m256 x_n = _mm256_mul_ps(xs, inv_denom);
  const __m256 y_n = _mm256_mul_ps(ys, inv_denom);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxDs(x_n), finiteMaskAvxDs(y_n)));

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
      _mm256_and_ps(finiteMaskAvxDs(rad_den), _mm256_cmp_ps(absAvxDs(rad_den), eps, _CMP_GT_OQ));
    valid = _mm256_and_ps(valid, den_valid);
    const __m256 safe_rad_den =
      _mm256_or_ps(_mm256_and_ps(den_valid, rad_den), _mm256_andnot_ps(den_valid, one));
    radial = _mm256_div_ps(rad_num, safe_rad_den);
  }

  const __m256 xy = _mm256_mul_ps(x_n, y_n);
  const __m256 xn_sq = _mm256_mul_ps(x_n, x_n);
  const __m256 yn_sq = _mm256_mul_ps(y_n, y_n);
  const __m256 x_tan = _mm256_add_ps(
    _mm256_mul_ps(_mm256_mul_ps(two, p1), xy),
    _mm256_mul_ps(p2, _mm256_add_ps(r2_v, _mm256_mul_ps(two, xn_sq)))
  );
  const __m256 y_tan = _mm256_add_ps(
    _mm256_mul_ps(p1, _mm256_add_ps(r2_v, _mm256_mul_ps(two, yn_sq))),
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
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxDs(u_v), finiteMaskAvxDs(v_v)));

  u_v = _mm256_and_ps(u_v, valid);
  v_v = _mm256_and_ps(v_v, valid);

  _mm256_storeu_ps(u_out, u_v);
  _mm256_storeu_ps(v_out, v_v);

  return _mm256_movemask_ps(valid);
}

/// Process 8 EUCM forward projections using AVX2.
inline int rayToPixelEucmAvx8(
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
    _mm256_and_ps(finiteMaskAvxDs(xs), _mm256_and_ps(finiteMaskAvxDs(ys), finiteMaskAvxDs(zs)));

  const float alpha_f = model.projection.alpha;
  const float beta_f = model.projection.beta;
  const __m256 alpha_v = _mm256_set1_ps(alpha_f);
  const __m256 beta_v = _mm256_set1_ps(beta_f);
  const __m256 one_m_alpha = _mm256_sub_ps(one, alpha_v);

  const __m256 xx = _mm256_mul_ps(xs, xs);
  const __m256 yy = _mm256_mul_ps(ys, ys);
  const __m256 zz = _mm256_mul_ps(zs, zs);

  const __m256 d = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(beta_v, _mm256_add_ps(xx, yy)), zz));
  __m256 valid =
    _mm256_and_ps(finite_xyz, _mm256_and_ps(_mm256_cmp_ps(d, eps, _CMP_GT_OQ), finiteMaskAvxDs(d)));

  const __m256 denom = _mm256_add_ps(_mm256_mul_ps(alpha_v, d), _mm256_mul_ps(one_m_alpha, zs));
  const __m256 denom_valid =
    _mm256_and_ps(finiteMaskAvxDs(denom), _mm256_cmp_ps(absAvxDs(denom), eps, _CMP_GT_OQ));
  valid = _mm256_and_ps(valid, denom_valid);

  // FOV check (unconditional; at alpha = 0 it degenerates to Z > 0),
  // matching the scalar forward.
  {
    const float w_f =
      (alpha_f <= 0.5f) ? (alpha_f / (1.0f - alpha_f)) : ((1.0f - alpha_f) / alpha_f);
    const __m256 neg_w = _mm256_set1_ps(-w_f);
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(zs, _mm256_mul_ps(neg_w, d), _CMP_GT_OQ));
  }

  // theta_max contract; d is beta-weighted, so use the true norm. Skipped at
  // the default pi.
  if (model.projection.theta_max < constants::kPiF)
  {
    const __m256 norm = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(xx, yy), zz));
    const __m256 cos_tm = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(zs, _mm256_mul_ps(norm, cos_tm), _CMP_GE_OQ));
  }

  int valid_mask = _mm256_movemask_ps(valid);
  if (valid_mask == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const __m256 safe_denom =
    _mm256_or_ps(_mm256_and_ps(denom_valid, denom), _mm256_andnot_ps(denom_valid, one));
  const __m256 inv_denom = _mm256_div_ps(one, safe_denom);
  const __m256 x_n = _mm256_mul_ps(xs, inv_denom);
  const __m256 y_n = _mm256_mul_ps(ys, inv_denom);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxDs(x_n), finiteMaskAvxDs(y_n)));

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
      _mm256_and_ps(finiteMaskAvxDs(rad_den), _mm256_cmp_ps(absAvxDs(rad_den), eps, _CMP_GT_OQ));
    valid = _mm256_and_ps(valid, den_valid);
    const __m256 safe_rad_den =
      _mm256_or_ps(_mm256_and_ps(den_valid, rad_den), _mm256_andnot_ps(den_valid, one));
    radial = _mm256_div_ps(rad_num, safe_rad_den);
  }

  const __m256 xy = _mm256_mul_ps(x_n, y_n);
  const __m256 xn_sq = _mm256_mul_ps(x_n, x_n);
  const __m256 yn_sq = _mm256_mul_ps(y_n, y_n);
  const __m256 x_tan = _mm256_add_ps(
    _mm256_mul_ps(_mm256_mul_ps(two, p1), xy),
    _mm256_mul_ps(p2, _mm256_add_ps(r2_v, _mm256_mul_ps(two, xn_sq)))
  );
  const __m256 y_tan = _mm256_add_ps(
    _mm256_mul_ps(p1, _mm256_add_ps(r2_v, _mm256_mul_ps(two, yn_sq))),
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
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxDs(u_v), finiteMaskAvxDs(v_v)));

  u_v = _mm256_and_ps(u_v, valid);
  v_v = _mm256_and_ps(v_v, valid);

  _mm256_storeu_ps(u_out, u_v);
  _mm256_storeu_ps(v_out, v_v);

  return _mm256_movemask_ps(valid);
}

int rayToPixelBatchDsphAvx2(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

int rayToPixelBatchEucmAvx2(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#endif  // CAMXIOM_HAS_AVX2

#endif  // CAMXIOM_HAS_SSE2

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_DS_EUCM_HPP
