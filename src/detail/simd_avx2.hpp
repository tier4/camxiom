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

#ifndef CAMXIOM__DETAIL__SIMD_AVX2_HPP
#define CAMXIOM__DETAIL__SIMD_AVX2_HPP

#include "camxiom/types.hpp"

#include <limits>

#if defined(__AVX2__)
#define CAMXIOM_HAS_AVX2 1
#include <immintrin.h>
#endif

namespace camxiom::detail
{

#ifdef CAMXIOM_HAS_AVX2

inline __m256 absAvx(const __m256 values)
{
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(values, sign_mask);
}

inline __m256 finiteMaskAvx(const __m256 values)
{
  const __m256 max_value = _mm256_set1_ps((std::numeric_limits<float>::max)());
  return _mm256_cmp_ps(absAvx(values), max_value, _CMP_LE_OQ);
}

/// Process 8 pinhole forward projections simultaneously using AVX2.
/// Same logic as the SSE4 version but with 256-bit registers.
/// Returns bitmask of valid points (bits 0-7).
inline int rayToPixelPinholeAvx8(
  const CameraModel &model, const float *rays_xyz, float *u_out, float *v_out
)
{
  // Gather X, Y, Z from interleaved AoS layout (stride = 3 floats)
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
  const __m256 z_positive = _mm256_cmp_ps(zs, zero, _CMP_GT_OQ);
  const __m256 finite_xyz =
    _mm256_and_ps(finiteMaskAvx(xs), _mm256_and_ps(finiteMaskAvx(ys), finiteMaskAvx(zs)));
  __m256 valid_mask = _mm256_and_ps(z_positive, finite_xyz);
  if (_mm256_movemask_ps(valid_mask) == 0)
  {
    _mm256_storeu_ps(u_out, zero);
    _mm256_storeu_ps(v_out, zero);
    return 0;
  }

  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 inv_z = _mm256_div_ps(one, zs);
  const __m256 x_n = _mm256_mul_ps(xs, inv_z);
  const __m256 y_n = _mm256_mul_ps(ys, inv_z);
  valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(finiteMaskAvx(x_n), finiteMaskAvx(y_n)));

  const __m256 r2_v = _mm256_add_ps(_mm256_mul_ps(x_n, x_n), _mm256_mul_ps(y_n, y_n));
  const __m256 r4_v = _mm256_mul_ps(r2_v, r2_v);
  const __m256 r6_v = _mm256_mul_ps(r4_v, r2_v);

  const auto &c = model.distortion.coeffs;
  const __m256 k1 = _mm256_set1_ps(c[0]);
  const __m256 k2 = _mm256_set1_ps(c[1]);
  const __m256 p1 = _mm256_set1_ps(c[2]);
  const __m256 p2 = _mm256_set1_ps(c[3]);
  const __m256 k3 = _mm256_set1_ps(c[4]);

  __m256 rad_num = _mm256_add_ps(one, _mm256_mul_ps(k1, r2_v));
  rad_num = _mm256_add_ps(rad_num, _mm256_mul_ps(k2, r4_v));
  rad_num = _mm256_add_ps(rad_num, _mm256_mul_ps(k3, r6_v));
  __m256 radial = rad_num;

  if (model.distortion.is_rational)
  {
    const __m256 k4 = _mm256_set1_ps(c[5]);
    const __m256 k5 = _mm256_set1_ps(c[6]);
    const __m256 k6 = _mm256_set1_ps(c[7]);
    __m256 rad_den = _mm256_add_ps(one, _mm256_mul_ps(k4, r2_v));
    rad_den = _mm256_add_ps(rad_den, _mm256_mul_ps(k5, r4_v));
    rad_den = _mm256_add_ps(rad_den, _mm256_mul_ps(k6, r6_v));
    const __m256 eps = _mm256_set1_ps(1e-8f);
    const __m256 den_valid =
      _mm256_and_ps(finiteMaskAvx(rad_den), _mm256_cmp_ps(absAvx(rad_den), eps, _CMP_GT_OQ));
    valid_mask = _mm256_and_ps(valid_mask, den_valid);
    const __m256 safe_rad_den =
      _mm256_or_ps(_mm256_and_ps(den_valid, rad_den), _mm256_andnot_ps(den_valid, one));
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
  valid_mask = _mm256_and_ps(valid_mask, _mm256_and_ps(finiteMaskAvx(u_v), finiteMaskAvx(v_v)));

  u_v = _mm256_and_ps(u_v, valid_mask);
  v_v = _mm256_and_ps(v_v, valid_mask);

  _mm256_storeu_ps(u_out, u_v);
  _mm256_storeu_ps(v_out, v_v);

  return _mm256_movemask_ps(valid_mask);
}

int rayToPixelBatchPinholeAvx2(
  const CameraModel &model, const float *rays_xyz, int count, float *u_out, float *v_out,
  StatusCode *statuses_out
);

#endif  // CAMXIOM_HAS_AVX2

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_AVX2_HPP
