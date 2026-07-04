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

#ifndef CAMXIOM__DETAIL__SIMD_INVERSE_HPP
#define CAMXIOM__DETAIL__SIMD_INVERSE_HPP

#include "camxiom/types.hpp"

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

#include <limits>

namespace camxiom::detail
{

inline constexpr float kInvNormBound = 1000.0f;

#ifdef CAMXIOM_HAS_SSE2

// ---------------------------------------------------------------------------
// SSE utilities for inverse
// ---------------------------------------------------------------------------

inline __m128 absSseInv(const __m128 v)
{
  const __m128 sign = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  return _mm_and_ps(v, sign);
}

inline __m128 finiteMaskSseInv(const __m128 v)
{
  const __m128 mx = _mm_set1_ps((std::numeric_limits<float>::max)());
  return _mm_cmple_ps(absSseInv(v), mx);
}

inline __m128 selectSseInv(const __m128 base, const __m128 value, const __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, value), _mm_andnot_ps(mask, base));
}

/// Cheap "any lane set" test for the Newton early-exit checks. On AArch64 a
/// single horizontal max (vmaxvq_u32) avoids the shim's shift+add movemask
/// emulation; on x86 the plain movemask is already a single instruction.
inline bool anyLaneSetSseInv(const __m128 mask)
{
#if defined(__aarch64__)
  return vmaxvq_u32(vreinterpretq_u32_f32(mask)) != 0U;
#else
  return _mm_movemask_ps(mask) != 0;
#endif
}

// ---------------------------------------------------------------------------
// SSE removeIntrinsics: 4 pixels → 4 (x_d, y_d)
// Includes finite + bound check to reject extreme inputs early.
// ---------------------------------------------------------------------------

inline void removeIntrinsicsSse4(
  const IntrinsicsModel &intr, const float *u_in, const float *v_in, __m128 &x_d, __m128 &y_d,
  __m128 &valid
)
{
  const __m128 u = _mm_loadu_ps(u_in);
  const __m128 v = _mm_loadu_ps(v_in);

  valid = _mm_and_ps(finiteMaskSseInv(u), finiteMaskSseInv(v));

  const __m128 fy_inv = _mm_set1_ps(1.0f / intr.fy);
  const __m128 fx_inv = _mm_set1_ps(1.0f / intr.fx);
  const __m128 cx = _mm_set1_ps(intr.cx);
  const __m128 cy = _mm_set1_ps(intr.cy);
  const __m128 sk = _mm_set1_ps(intr.skew);

  y_d = _mm_mul_ps(_mm_sub_ps(v, cy), fy_inv);
  x_d = _mm_mul_ps(_mm_sub_ps(_mm_sub_ps(u, cx), _mm_mul_ps(sk, y_d)), fx_inv);

  const __m128 bound = _mm_set1_ps(kInvNormBound);
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseInv(x_d), finiteMaskSseInv(y_d)));
  valid = _mm_and_ps(
    valid, _mm_and_ps(_mm_cmple_ps(absSseInv(x_d), bound), _mm_cmple_ps(absSseInv(y_d), bound))
  );
}

// ---------------------------------------------------------------------------
// SSE plane distort forward (no tilt): for Newton solver inner loop
// Returns distorted (x_d, y_d) and Jacobian (j00, j01, j10, j11)
// ---------------------------------------------------------------------------

inline void distortPlaneSse4(
  const DistortionModel &dist, const __m128 x, const __m128 y, __m128 &xd, __m128 &yd, __m128 &j00,
  __m128 &j01, __m128 &j10, __m128 &j11
)
{
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 two = _mm_set1_ps(2.0f);
  const __m128 three = _mm_set1_ps(3.0f);

  const __m128 xx = _mm_mul_ps(x, x);
  const __m128 yy = _mm_mul_ps(y, y);
  const __m128 xy = _mm_mul_ps(x, y);
  const __m128 r2 = _mm_add_ps(xx, yy);
  const __m128 r4 = _mm_mul_ps(r2, r2);
  const __m128 r6 = _mm_mul_ps(r4, r2);

  const auto &c = dist.coeffs;
  const __m128 k1 = _mm_set1_ps(c[0]);
  const __m128 k2 = _mm_set1_ps(c[1]);
  const __m128 p1 = _mm_set1_ps(c[2]);
  const __m128 p2 = _mm_set1_ps(c[3]);
  const __m128 k3 = _mm_set1_ps(c[4]);

  __m128 num = _mm_add_ps(one, _mm_mul_ps(k1, r2));
  num = _mm_add_ps(num, _mm_mul_ps(k2, r4));
  num = _mm_add_ps(num, _mm_mul_ps(k3, r6));

  const __m128 dcoeff = _mm_add_ps(
    k1, _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(k2, r2)), _mm_mul_ps(three, _mm_mul_ps(k3, r4)))
  );
  const __m128 dnum_dx = _mm_mul_ps(_mm_mul_ps(two, x), dcoeff);
  const __m128 dnum_dy = _mm_mul_ps(_mm_mul_ps(two, y), dcoeff);

  __m128 radial = num;
  __m128 drad_dx = dnum_dx;
  __m128 drad_dy = dnum_dy;

  if (dist.is_rational)
  {
    const __m128 k4 = _mm_set1_ps(c[5]);
    const __m128 k5 = _mm_set1_ps(c[6]);
    const __m128 k6 = _mm_set1_ps(c[7]);
    __m128 den = _mm_add_ps(one, _mm_mul_ps(k4, r2));
    den = _mm_add_ps(den, _mm_mul_ps(k5, r4));
    den = _mm_add_ps(den, _mm_mul_ps(k6, r6));
    const __m128 inv_den = _mm_div_ps(one, den);
    const __m128 dcoeff_den = _mm_add_ps(
      k4, _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(k5, r2)), _mm_mul_ps(three, _mm_mul_ps(k6, r4)))
    );
    const __m128 dden_dx = _mm_mul_ps(_mm_mul_ps(two, x), dcoeff_den);
    const __m128 dden_dy = _mm_mul_ps(_mm_mul_ps(two, y), dcoeff_den);
    radial = _mm_mul_ps(num, inv_den);
    const __m128 inv_den2 = _mm_mul_ps(inv_den, inv_den);
    drad_dx = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(dnum_dx, den), _mm_mul_ps(num, dden_dx)), inv_den2);
    drad_dy = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(dnum_dy, den), _mm_mul_ps(num, dden_dy)), inv_den2);
  }

  const __m128 x_tan = _mm_add_ps(
    _mm_mul_ps(_mm_mul_ps(two, p1), xy), _mm_mul_ps(p2, _mm_add_ps(r2, _mm_mul_ps(two, xx)))
  );
  const __m128 y_tan = _mm_add_ps(
    _mm_mul_ps(p1, _mm_add_ps(r2, _mm_mul_ps(two, yy))), _mm_mul_ps(_mm_mul_ps(two, p2), xy)
  );

  xd = _mm_add_ps(_mm_mul_ps(x, radial), x_tan);
  yd = _mm_add_ps(_mm_mul_ps(y, radial), y_tan);

  __m128 dx_tan_dx = _mm_add_ps(
    _mm_mul_ps(two, _mm_mul_ps(p1, y)), _mm_mul_ps(_mm_set1_ps(6.0f), _mm_mul_ps(p2, x))
  );
  __m128 dx_tan_dy =
    _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(p1, x)), _mm_mul_ps(two, _mm_mul_ps(p2, y)));
  __m128 dy_tan_dx =
    _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(p1, x)), _mm_mul_ps(two, _mm_mul_ps(p2, y)));
  __m128 dy_tan_dy = _mm_add_ps(
    _mm_mul_ps(_mm_set1_ps(6.0f), _mm_mul_ps(p1, y)), _mm_mul_ps(two, _mm_mul_ps(p2, x))
  );

  if (dist.has_thin_prism)
  {
    const __m128 s1 = _mm_set1_ps(c[8]);
    const __m128 s2 = _mm_set1_ps(c[9]);
    const __m128 s3 = _mm_set1_ps(c[10]);
    const __m128 s4 = _mm_set1_ps(c[11]);
    xd = _mm_add_ps(xd, _mm_add_ps(_mm_mul_ps(s1, r2), _mm_mul_ps(s2, r4)));
    yd = _mm_add_ps(yd, _mm_add_ps(_mm_mul_ps(s3, r2), _mm_mul_ps(s4, r4)));

    const __m128 px =
      _mm_mul_ps(_mm_mul_ps(two, x), _mm_add_ps(s1, _mm_mul_ps(two, _mm_mul_ps(s2, r2))));
    const __m128 py =
      _mm_mul_ps(_mm_mul_ps(two, y), _mm_add_ps(s1, _mm_mul_ps(two, _mm_mul_ps(s2, r2))));
    dx_tan_dx = _mm_add_ps(dx_tan_dx, px);
    dx_tan_dy = _mm_add_ps(dx_tan_dy, py);
    const __m128 qx =
      _mm_mul_ps(_mm_mul_ps(two, x), _mm_add_ps(s3, _mm_mul_ps(two, _mm_mul_ps(s4, r2))));
    const __m128 qy =
      _mm_mul_ps(_mm_mul_ps(two, y), _mm_add_ps(s3, _mm_mul_ps(two, _mm_mul_ps(s4, r2))));
    dy_tan_dx = _mm_add_ps(dy_tan_dx, qx);
    dy_tan_dy = _mm_add_ps(dy_tan_dy, qy);
  }

  j00 = _mm_add_ps(_mm_add_ps(radial, _mm_mul_ps(x, drad_dx)), dx_tan_dx);
  j01 = _mm_add_ps(_mm_mul_ps(x, drad_dy), dx_tan_dy);
  j10 = _mm_add_ps(_mm_mul_ps(y, drad_dx), dy_tan_dx);
  j11 = _mm_add_ps(_mm_add_ps(radial, _mm_mul_ps(y, drad_dy)), dy_tan_dy);
}

// ---------------------------------------------------------------------------
// SSE Newton plane undistort: 4 pixels, fixed-iteration
// ---------------------------------------------------------------------------

inline void undistortPlaneNewtonSse4(
  const DistortionModel &dist, const __m128 obs_x, const __m128 obs_y, __m128 &ux, __m128 &uy,
  __m128 &valid, const int max_iter = 10
)
{
  ux = obs_x;
  uy = obs_y;
  const __m128 eps = _mm_set1_ps(1e-8f);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    __m128 dx, dy, j00, j01, j10, j11;
    distortPlaneSse4(dist, ux, uy, dx, dy, j00, j01, j10, j11);

    const __m128 rx = _mm_sub_ps(dx, obs_x);
    const __m128 ry = _mm_sub_ps(dy, obs_y);

    // Early exit once every still-valid lane's residual is within the same
    // 1e-6 bound the post-loop verification checks — mirrors the scalar
    // Newton's exit instead of always burning max_iter iterations (typical
    // pixels converge in 2-4). Converged results stop at ~1e-6 residual like
    // the scalar path; validity semantics are unchanged.
    {
      const __m128 resid = _mm_max_ps(absSseInv(rx), absSseInv(ry));
      const __m128 not_conv = _mm_cmpgt_ps(resid, _mm_set1_ps(1e-6f));
      if (!anyLaneSetSseInv(_mm_and_ps(valid, not_conv)))
      {
        break;
      }
    }

    const __m128 det = _mm_sub_ps(_mm_mul_ps(j00, j11), _mm_mul_ps(j01, j10));
    const __m128 abs_det = absSseInv(det);
    const __m128 det_ok = _mm_cmpgt_ps(abs_det, eps);
    const __m128 safe_det = selectSseInv(_mm_set1_ps(1.0f), det, det_ok);
    const __m128 inv_det = _mm_div_ps(_mm_set1_ps(1.0f), safe_det);

    const __m128 delta_x = _mm_mul_ps(
      _mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_setzero_ps(), j11), rx), _mm_mul_ps(j01, ry)), inv_det
    );
    const __m128 delta_y =
      _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(j10, rx), _mm_mul_ps(j00, ry)), inv_det);

    const __m128 update_mask = _mm_and_ps(valid, det_ok);
    ux = _mm_add_ps(ux, _mm_and_ps(delta_x, update_mask));
    uy = _mm_add_ps(uy, _mm_and_ps(delta_y, update_mask));
  }

  // verify convergence + finite + bound check
  valid = _mm_and_ps(valid, _mm_and_ps(finiteMaskSseInv(ux), finiteMaskSseInv(uy)));
  {
    __m128 dx, dy, j00, j01, j10, j11;
    distortPlaneSse4(dist, ux, uy, dx, dy, j00, j01, j10, j11);
    const __m128 rx = _mm_sub_ps(dx, obs_x);
    const __m128 ry = _mm_sub_ps(dy, obs_y);
    const __m128 resid = _mm_max_ps(absSseInv(rx), absSseInv(ry));
    const __m128 tol = _mm_set1_ps(1e-6f);
    valid = _mm_and_ps(valid, _mm_cmple_ps(resid, tol));
  }
}

// ---------------------------------------------------------------------------
// SSE pixelToRay kernels (4 pixels at a time)
// ---------------------------------------------------------------------------

int pixelToRayPinholeSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayDsphSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayEucmSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayOmniSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayFisheyeSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

// ---------------------------------------------------------------------------
// SSE Batch drivers
// ---------------------------------------------------------------------------

int pixelToRayBatchPinholeSse(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchDsphSse(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchEucmSse(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchOmniSse(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchFisheyeSse(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

#endif  // CAMXIOM_HAS_SSE2

#ifdef CAMXIOM_HAS_AVX2

// ---------------------------------------------------------------------------
// AVX2 utilities for inverse
// ---------------------------------------------------------------------------

inline __m256 absAvxInv(const __m256 v)
{
  const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(v, sign);
}

inline __m256 finiteMaskAvxInv(const __m256 v)
{
  const __m256 mx = _mm256_set1_ps((std::numeric_limits<float>::max)());
  return _mm256_cmp_ps(absAvxInv(v), mx, _CMP_LE_OQ);
}

inline __m256 selectAvxInv(const __m256 base, const __m256 value, const __m256 mask)
{
  return _mm256_or_ps(_mm256_and_ps(mask, value), _mm256_andnot_ps(mask, base));
}

// ---------------------------------------------------------------------------
// AVX2 removeIntrinsics: 8 pixels → 8 (x_d, y_d)
// ---------------------------------------------------------------------------

inline void removeIntrinsicsAvx8(
  const IntrinsicsModel &intr, const float *u_in, const float *v_in, __m256 &x_d, __m256 &y_d,
  __m256 &valid
)
{
  const __m256 u = _mm256_loadu_ps(u_in);
  const __m256 v = _mm256_loadu_ps(v_in);

  valid = _mm256_and_ps(finiteMaskAvxInv(u), finiteMaskAvxInv(v));

  const __m256 fy_inv = _mm256_set1_ps(1.0f / intr.fy);
  const __m256 fx_inv = _mm256_set1_ps(1.0f / intr.fx);
  const __m256 cx = _mm256_set1_ps(intr.cx);
  const __m256 cy = _mm256_set1_ps(intr.cy);
  const __m256 sk = _mm256_set1_ps(intr.skew);

  y_d = _mm256_mul_ps(_mm256_sub_ps(v, cy), fy_inv);
  x_d = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(u, cx), _mm256_mul_ps(sk, y_d)), fx_inv);

  const __m256 bound = _mm256_set1_ps(kInvNormBound);
  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxInv(x_d), finiteMaskAvxInv(y_d)));
  valid = _mm256_and_ps(
    valid, _mm256_and_ps(
             _mm256_cmp_ps(absAvxInv(x_d), bound, _CMP_LE_OQ),
             _mm256_cmp_ps(absAvxInv(y_d), bound, _CMP_LE_OQ)
           )
  );
}

// ---------------------------------------------------------------------------
// AVX2 plane distort forward (no tilt): for Newton solver inner loop
// ---------------------------------------------------------------------------

inline void distortPlaneAvx8(
  const DistortionModel &dist, const __m256 x, const __m256 y, __m256 &xd, __m256 &yd, __m256 &j00,
  __m256 &j01, __m256 &j10, __m256 &j11
)
{
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 three = _mm256_set1_ps(3.0f);

  const __m256 xx = _mm256_mul_ps(x, x);
  const __m256 yy = _mm256_mul_ps(y, y);
  const __m256 xy = _mm256_mul_ps(x, y);
  const __m256 r2 = _mm256_add_ps(xx, yy);
  const __m256 r4 = _mm256_mul_ps(r2, r2);
  const __m256 r6 = _mm256_mul_ps(r4, r2);

  const auto &c = dist.coeffs;
  const __m256 k1 = _mm256_set1_ps(c[0]);
  const __m256 k2 = _mm256_set1_ps(c[1]);
  const __m256 p1 = _mm256_set1_ps(c[2]);
  const __m256 p2 = _mm256_set1_ps(c[3]);
  const __m256 k3 = _mm256_set1_ps(c[4]);

  __m256 num = _mm256_add_ps(one, _mm256_mul_ps(k1, r2));
  num = _mm256_add_ps(num, _mm256_mul_ps(k2, r4));
  num = _mm256_add_ps(num, _mm256_mul_ps(k3, r6));

  const __m256 dcoeff = _mm256_add_ps(
    k1, _mm256_add_ps(
          _mm256_mul_ps(two, _mm256_mul_ps(k2, r2)), _mm256_mul_ps(three, _mm256_mul_ps(k3, r4))
        )
  );
  const __m256 dnum_dx = _mm256_mul_ps(_mm256_mul_ps(two, x), dcoeff);
  const __m256 dnum_dy = _mm256_mul_ps(_mm256_mul_ps(two, y), dcoeff);

  __m256 radial = num;
  __m256 drad_dx = dnum_dx;
  __m256 drad_dy = dnum_dy;

  if (dist.is_rational)
  {
    const __m256 k4 = _mm256_set1_ps(c[5]);
    const __m256 k5 = _mm256_set1_ps(c[6]);
    const __m256 k6 = _mm256_set1_ps(c[7]);
    __m256 den = _mm256_add_ps(one, _mm256_mul_ps(k4, r2));
    den = _mm256_add_ps(den, _mm256_mul_ps(k5, r4));
    den = _mm256_add_ps(den, _mm256_mul_ps(k6, r6));
    const __m256 inv_den = _mm256_div_ps(one, den);
    const __m256 dcoeff_den = _mm256_add_ps(
      k4, _mm256_add_ps(
            _mm256_mul_ps(two, _mm256_mul_ps(k5, r2)), _mm256_mul_ps(three, _mm256_mul_ps(k6, r4))
          )
    );
    const __m256 dden_dx = _mm256_mul_ps(_mm256_mul_ps(two, x), dcoeff_den);
    const __m256 dden_dy = _mm256_mul_ps(_mm256_mul_ps(two, y), dcoeff_den);
    radial = _mm256_mul_ps(num, inv_den);
    const __m256 inv_den2 = _mm256_mul_ps(inv_den, inv_den);
    drad_dx = _mm256_mul_ps(
      _mm256_sub_ps(_mm256_mul_ps(dnum_dx, den), _mm256_mul_ps(num, dden_dx)), inv_den2
    );
    drad_dy = _mm256_mul_ps(
      _mm256_sub_ps(_mm256_mul_ps(dnum_dy, den), _mm256_mul_ps(num, dden_dy)), inv_den2
    );
  }

  const __m256 x_tan = _mm256_add_ps(
    _mm256_mul_ps(_mm256_mul_ps(two, p1), xy),
    _mm256_mul_ps(p2, _mm256_add_ps(r2, _mm256_mul_ps(two, xx)))
  );
  const __m256 y_tan = _mm256_add_ps(
    _mm256_mul_ps(p1, _mm256_add_ps(r2, _mm256_mul_ps(two, yy))),
    _mm256_mul_ps(_mm256_mul_ps(two, p2), xy)
  );

  xd = _mm256_add_ps(_mm256_mul_ps(x, radial), x_tan);
  yd = _mm256_add_ps(_mm256_mul_ps(y, radial), y_tan);

  __m256 dx_tan_dx = _mm256_add_ps(
    _mm256_mul_ps(two, _mm256_mul_ps(p1, y)),
    _mm256_mul_ps(_mm256_set1_ps(6.0f), _mm256_mul_ps(p2, x))
  );
  __m256 dx_tan_dy = _mm256_add_ps(
    _mm256_mul_ps(two, _mm256_mul_ps(p1, x)), _mm256_mul_ps(two, _mm256_mul_ps(p2, y))
  );
  __m256 dy_tan_dx = _mm256_add_ps(
    _mm256_mul_ps(two, _mm256_mul_ps(p1, x)), _mm256_mul_ps(two, _mm256_mul_ps(p2, y))
  );
  __m256 dy_tan_dy = _mm256_add_ps(
    _mm256_mul_ps(_mm256_set1_ps(6.0f), _mm256_mul_ps(p1, y)),
    _mm256_mul_ps(two, _mm256_mul_ps(p2, x))
  );

  if (dist.has_thin_prism)
  {
    const __m256 s1 = _mm256_set1_ps(c[8]);
    const __m256 s2 = _mm256_set1_ps(c[9]);
    const __m256 s3 = _mm256_set1_ps(c[10]);
    const __m256 s4 = _mm256_set1_ps(c[11]);
    xd = _mm256_add_ps(xd, _mm256_add_ps(_mm256_mul_ps(s1, r2), _mm256_mul_ps(s2, r4)));
    yd = _mm256_add_ps(yd, _mm256_add_ps(_mm256_mul_ps(s3, r2), _mm256_mul_ps(s4, r4)));

    const __m256 px = _mm256_mul_ps(
      _mm256_mul_ps(two, x), _mm256_add_ps(s1, _mm256_mul_ps(two, _mm256_mul_ps(s2, r2)))
    );
    const __m256 py = _mm256_mul_ps(
      _mm256_mul_ps(two, y), _mm256_add_ps(s1, _mm256_mul_ps(two, _mm256_mul_ps(s2, r2)))
    );
    dx_tan_dx = _mm256_add_ps(dx_tan_dx, px);
    dx_tan_dy = _mm256_add_ps(dx_tan_dy, py);
    const __m256 qx = _mm256_mul_ps(
      _mm256_mul_ps(two, x), _mm256_add_ps(s3, _mm256_mul_ps(two, _mm256_mul_ps(s4, r2)))
    );
    const __m256 qy = _mm256_mul_ps(
      _mm256_mul_ps(two, y), _mm256_add_ps(s3, _mm256_mul_ps(two, _mm256_mul_ps(s4, r2)))
    );
    dy_tan_dx = _mm256_add_ps(dy_tan_dx, qx);
    dy_tan_dy = _mm256_add_ps(dy_tan_dy, qy);
  }

  j00 = _mm256_add_ps(_mm256_add_ps(radial, _mm256_mul_ps(x, drad_dx)), dx_tan_dx);
  j01 = _mm256_add_ps(_mm256_mul_ps(x, drad_dy), dx_tan_dy);
  j10 = _mm256_add_ps(_mm256_mul_ps(y, drad_dx), dy_tan_dx);
  j11 = _mm256_add_ps(_mm256_add_ps(radial, _mm256_mul_ps(y, drad_dy)), dy_tan_dy);
}

// ---------------------------------------------------------------------------
// AVX2 Newton plane undistort: 8 pixels, fixed-iteration
// ---------------------------------------------------------------------------

inline void undistortPlaneNewtonAvx8(
  const DistortionModel &dist, const __m256 obs_x, const __m256 obs_y, __m256 &ux, __m256 &uy,
  __m256 &valid, const int max_iter = 10
)
{
  ux = obs_x;
  uy = obs_y;
  const __m256 eps = _mm256_set1_ps(1e-8f);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    __m256 dx, dy, j00, j01, j10, j11;
    distortPlaneAvx8(dist, ux, uy, dx, dy, j00, j01, j10, j11);

    const __m256 rx = _mm256_sub_ps(dx, obs_x);
    const __m256 ry = _mm256_sub_ps(dy, obs_y);

    const __m256 det = _mm256_sub_ps(_mm256_mul_ps(j00, j11), _mm256_mul_ps(j01, j10));
    const __m256 abs_det = absAvxInv(det);
    const __m256 det_ok = _mm256_cmp_ps(abs_det, eps, _CMP_GT_OQ);
    const __m256 safe_det = selectAvxInv(_mm256_set1_ps(1.0f), det, det_ok);
    const __m256 inv_det = _mm256_div_ps(_mm256_set1_ps(1.0f), safe_det);

    const __m256 delta_x = _mm256_mul_ps(
      _mm256_add_ps(
        _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), j11), rx), _mm256_mul_ps(j01, ry)
      ),
      inv_det
    );
    const __m256 delta_y =
      _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(j10, rx), _mm256_mul_ps(j00, ry)), inv_det);

    const __m256 update_mask = _mm256_and_ps(valid, det_ok);
    ux = _mm256_add_ps(ux, _mm256_and_ps(delta_x, update_mask));
    uy = _mm256_add_ps(uy, _mm256_and_ps(delta_y, update_mask));
  }

  valid = _mm256_and_ps(valid, _mm256_and_ps(finiteMaskAvxInv(ux), finiteMaskAvxInv(uy)));
  {
    __m256 dx, dy, j00, j01, j10, j11;
    distortPlaneAvx8(dist, ux, uy, dx, dy, j00, j01, j10, j11);
    const __m256 rx = _mm256_sub_ps(dx, obs_x);
    const __m256 ry = _mm256_sub_ps(dy, obs_y);
    const __m256 resid = _mm256_max_ps(absAvxInv(rx), absAvxInv(ry));
    const __m256 tol = _mm256_set1_ps(1e-6f);
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(resid, tol, _CMP_LE_OQ));
  }
}

// ---------------------------------------------------------------------------
// AVX2 pixelToRay kernels (8 pixels at a time)
// ---------------------------------------------------------------------------

int pixelToRayPinholeAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayDsphAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayEucmAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayOmniAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

int pixelToRayFisheyeAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
);

// ---------------------------------------------------------------------------
// AVX2 batch drivers (AVX8 → SSE4 → scalar)
// ---------------------------------------------------------------------------

int pixelToRayBatchPinholeAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchDsphAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchEucmAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchOmniAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

int pixelToRayBatchFisheyeAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
);

#endif  // CAMXIOM_HAS_AVX2

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_INVERSE_HPP
