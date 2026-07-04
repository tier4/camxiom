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

#ifndef CAMXIOM__DETAIL__SIMD_PINHOLE_HPP
#define CAMXIOM__DETAIL__SIMD_PINHOLE_HPP

#include "camxiom/types.hpp"

#include <limits>

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

#if defined(__SSE4_1__)
#define CAMXIOM_HAS_SSE41 1
#include <smmintrin.h>
#endif

namespace camxiom::detail
{

#ifdef CAMXIOM_HAS_SSE2

inline __m128 absSse(const __m128 values)
{
  const __m128 sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  return _mm_and_ps(values, sign_mask);
}

inline __m128 finiteMaskSse(const __m128 values)
{
  const __m128 max_value = _mm_set1_ps((std::numeric_limits<float>::max)());
  return _mm_cmple_ps(absSse(values), max_value);
}

/// Process 4 pinhole forward projections simultaneously using SSE.
///
/// Input: rays_xyz interleaved [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
/// Output: u_out[4], v_out[4], valid_mask (bitmask of valid points, bits 0-3)
///
/// Handles: z <= 0 masking, RadTan5 distortion (k1,k2,p1,p2,k3),
///          rational denominator (k4,k5,k6), thin prism (s1,s2,s3,s4),
///          intrinsics (fx,fy,cx,cy,skew).
///
/// Returns bitmask of valid points (0-15).
inline int rayToPixelPinholeSse4(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  // Gather 4 rays from AoS layout.
  const __m128 xs_v = _mm_set_ps(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m128 ys_v = _mm_set_ps(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m128 zs_v = _mm_set_ps(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m128 zero = _mm_setzero_ps();
  const __m128 z_positive = _mm_cmpgt_ps(zs_v, zero);
  const __m128 finite_xyz =
    _mm_and_ps(finiteMaskSse(xs_v), _mm_and_ps(finiteMaskSse(ys_v), finiteMaskSse(zs_v)));
  __m128 valid_mask = _mm_and_ps(z_positive, finite_xyz);
  if (_mm_movemask_ps(valid_mask) == 0)
  {
    _mm_storeu_ps(u_out, zero);
    _mm_storeu_ps(v_out, zero);
    return 0;
  }

  // x_n = X/Z, y_n = Y/Z
  const __m128 inv_z = _mm_div_ps(_mm_set1_ps(1.0f), zs_v);
  const __m128 x_n = _mm_mul_ps(xs_v, inv_z);
  const __m128 y_n = _mm_mul_ps(ys_v, inv_z);
  valid_mask = _mm_and_ps(valid_mask, _mm_and_ps(finiteMaskSse(x_n), finiteMaskSse(y_n)));

  // r² = x² + y²
  const __m128 r2_v = _mm_add_ps(_mm_mul_ps(x_n, x_n), _mm_mul_ps(y_n, y_n));
  const __m128 r4_v = _mm_mul_ps(r2_v, r2_v);
  const __m128 r6_v = _mm_mul_ps(r4_v, r2_v);

  // Distortion coefficients (broadcast)
  const auto &c = model.distortion.coeffs;
  const __m128 k1 = _mm_set1_ps(c[0]);
  const __m128 k2 = _mm_set1_ps(c[1]);
  const __m128 p1 = _mm_set1_ps(c[2]);
  const __m128 p2 = _mm_set1_ps(c[3]);
  const __m128 k3 = _mm_set1_ps(c[4]);
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 two = _mm_set1_ps(2.0f);

  // radial_num = 1 + k1*r² + k2*r⁴ + k3*r⁶
  __m128 rad_num = _mm_add_ps(one, _mm_mul_ps(k1, r2_v));
  rad_num = _mm_add_ps(rad_num, _mm_mul_ps(k2, r4_v));
  rad_num = _mm_add_ps(rad_num, _mm_mul_ps(k3, r6_v));

  __m128 radial = rad_num;

  // Rational denominator (if applicable)
  if (model.distortion.is_rational)
  {
    const __m128 k4 = _mm_set1_ps(c[5]);
    const __m128 k5 = _mm_set1_ps(c[6]);
    const __m128 k6 = _mm_set1_ps(c[7]);
    __m128 rad_den = _mm_add_ps(one, _mm_mul_ps(k4, r2_v));
    rad_den = _mm_add_ps(rad_den, _mm_mul_ps(k5, r4_v));
    rad_den = _mm_add_ps(rad_den, _mm_mul_ps(k6, r6_v));
    const __m128 eps = _mm_set1_ps(1e-8f);
    const __m128 den_valid = _mm_and_ps(finiteMaskSse(rad_den), _mm_cmpgt_ps(absSse(rad_den), eps));
    valid_mask = _mm_and_ps(valid_mask, den_valid);
    const __m128 safe_rad_den =
      _mm_or_ps(_mm_and_ps(den_valid, rad_den), _mm_andnot_ps(den_valid, one));
    radial = _mm_div_ps(rad_num, safe_rad_den);
  }

  // Tangential distortion
  const __m128 xy = _mm_mul_ps(x_n, y_n);
  const __m128 xx = _mm_mul_ps(x_n, x_n);
  const __m128 yy = _mm_mul_ps(y_n, y_n);
  // x_tan = 2*p1*x*y + p2*(r² + 2*x²)
  const __m128 x_tan = _mm_add_ps(
    _mm_mul_ps(_mm_mul_ps(two, p1), xy), _mm_mul_ps(p2, _mm_add_ps(r2_v, _mm_mul_ps(two, xx)))
  );
  // y_tan = p1*(r² + 2*y²) + 2*p2*x*y
  const __m128 y_tan = _mm_add_ps(
    _mm_mul_ps(p1, _mm_add_ps(r2_v, _mm_mul_ps(two, yy))), _mm_mul_ps(_mm_mul_ps(two, p2), xy)
  );

  // x_d = x*radial + x_tan, y_d = y*radial + y_tan
  __m128 x_d = _mm_add_ps(_mm_mul_ps(x_n, radial), x_tan);
  __m128 y_d = _mm_add_ps(_mm_mul_ps(y_n, radial), y_tan);

  // Thin prism (if applicable)
  if (model.distortion.has_thin_prism)
  {
    const __m128 s1 = _mm_set1_ps(c[8]);
    const __m128 s2 = _mm_set1_ps(c[9]);
    const __m128 s3 = _mm_set1_ps(c[10]);
    const __m128 s4 = _mm_set1_ps(c[11]);
    x_d = _mm_add_ps(x_d, _mm_add_ps(_mm_mul_ps(s1, r2_v), _mm_mul_ps(s2, r4_v)));
    y_d = _mm_add_ps(y_d, _mm_add_ps(_mm_mul_ps(s3, r2_v), _mm_mul_ps(s4, r4_v)));
  }

  // Apply intrinsics: u = fx*x_d + skew*y_d + cx, v = fy*y_d + cy
  const __m128 fx = _mm_set1_ps(model.intrinsics.fx);
  const __m128 fy = _mm_set1_ps(model.intrinsics.fy);
  const __m128 cx = _mm_set1_ps(model.intrinsics.cx);
  const __m128 cy = _mm_set1_ps(model.intrinsics.cy);
  const __m128 sk = _mm_set1_ps(model.intrinsics.skew);

  __m128 u_v = _mm_add_ps(_mm_add_ps(_mm_mul_ps(fx, x_d), _mm_mul_ps(sk, y_d)), cx);
  __m128 v_v = _mm_add_ps(_mm_mul_ps(fy, y_d), cy);
  valid_mask = _mm_and_ps(valid_mask, _mm_and_ps(finiteMaskSse(u_v), finiteMaskSse(v_v)));

  // Mask out invalid points with 0.
  u_v = _mm_and_ps(u_v, valid_mask);
  v_v = _mm_and_ps(v_v, valid_mask);

  _mm_storeu_ps(u_out, u_v);
  _mm_storeu_ps(v_out, v_v);

  return _mm_movemask_ps(valid_mask);
}

/// SIMD-accelerated pinhole batch forward projection.
/// Falls back to scalar for the last (count % 4) points.
/// Tilt models are not handled by the SSE path and fall back entirely to scalar.
///
/// @return Number of valid projections.
int rayToPixelBatchPinholeSse(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#endif  // CAMXIOM_HAS_SSE2

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_PINHOLE_HPP
