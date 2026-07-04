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

#ifndef CAMXIOM__DETAIL__SIMD_NEON_COMPAT_HPP
#define CAMXIOM__DETAIL__SIMD_NEON_COMPAT_HPP

// Minimal __m128 emulation on top of ARM NEON (AArch64 only).
//
// The 4-wide SIMD kernels in src/detail/simd_*.hpp are written against the
// SSE intrinsics. Rather than hand-porting every kernel to NEON (a second
// copy of ~2000 lines of carefully validated math — exactly the float/double
// duplication this library just spent #1 eliminating), this header maps the
// SMALL, CLOSED set of SSE intrinsics those kernels actually use onto their
// NEON equivalents. The kernels then compile unchanged on aarch64, and the
// batch-parity suite (tests/batch_parity_test.cpp) validates the shim against
// the scalar paths natively on ARM hardware.
//
// Scope and caveats:
//   * AArch64 only: vdivq_f32 / vsqrtq_f32 need ARMv8. 32-bit ARM stays on
//     the scalar fallback.
//   * This is NOT a general SSE-on-NEON layer (use sse2neon for that). Only
//     the intrinsics used by the camxiom kernels are provided; adding a new
//     intrinsic to a kernel means adding its mapping here (the compiler
//     errors out loudly if you forget).
//   * The `_mm_*` names shadow the x86 vendor namespace on purpose so the
//     kernels compile verbatim. This header must never be included on x86
//     (the guards in simd_*.hpp pick <emmintrin.h> first).
//   * Semantics match SSE where the kernels rely on it: NaN compares are
//     false, comparison results are all-ones/all-zero lane masks reinterpreted
//     as floats, _mm_set_ps takes lanes in (e3, e2, e1, e0) order.

#if defined(__aarch64__)

#include <arm_neon.h>

using __m128 = float32x4_t;
using __m128i = int32x4_t;

static inline __m128 _mm_set1_ps(float v) { return vdupq_n_f32(v); }
static inline __m128 _mm_setzero_ps() { return vdupq_n_f32(0.0f); }

// SSE argument order: e3 is the HIGHEST lane, e0 the lowest.
static inline __m128 _mm_set_ps(float e3, float e2, float e1, float e0)
{
  const float lanes[4] = {e0, e1, e2, e3};
  return vld1q_f32(lanes);
}

// NEON vld1q/vst1q have no alignment requirement, so the aligned and
// unaligned variants coincide.
static inline __m128 _mm_loadu_ps(const float *p) { return vld1q_f32(p); }
static inline __m128 _mm_load_ps(const float *p) { return vld1q_f32(p); }
static inline void _mm_storeu_ps(float *p, __m128 a) { vst1q_f32(p, a); }
static inline void _mm_store_ps(float *p, __m128 a) { vst1q_f32(p, a); }

static inline __m128 _mm_add_ps(__m128 a, __m128 b) { return vaddq_f32(a, b); }
static inline __m128 _mm_sub_ps(__m128 a, __m128 b) { return vsubq_f32(a, b); }
static inline __m128 _mm_mul_ps(__m128 a, __m128 b) { return vmulq_f32(a, b); }
static inline __m128 _mm_div_ps(__m128 a, __m128 b) { return vdivq_f32(a, b); }
static inline __m128 _mm_sqrt_ps(__m128 a) { return vsqrtq_f32(a); }
static inline __m128 _mm_min_ps(__m128 a, __m128 b) { return vminq_f32(a, b); }
static inline __m128 _mm_max_ps(__m128 a, __m128 b) { return vmaxq_f32(a, b); }

static inline __m128 _mm_and_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

static inline __m128 _mm_or_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

static inline __m128 _mm_xor_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

// _mm_andnot_ps(a, b) = (~a) & b; NEON vbicq_u32(x, y) = x & ~y.
static inline __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(b), vreinterpretq_u32_f32(a)));
}

// Comparisons: all-ones on true, all-zeros on false, NaN -> false — the same
// lane-mask convention the SSE kernels rely on.
static inline __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vcgtq_f32(a, b));
}
static inline __m128 _mm_cmpge_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vcgeq_f32(a, b));
}
static inline __m128 _mm_cmplt_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vcltq_f32(a, b));
}
static inline __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
  return vreinterpretq_f32_u32(vcleq_f32(a, b));
}

static inline __m128i _mm_set1_epi32(int v) { return vdupq_n_s32(v); }
static inline __m128 _mm_castsi128_ps(__m128i a) { return vreinterpretq_f32_s32(a); }

// Sign bit of each lane packed into bits 0..3.
static inline int _mm_movemask_ps(__m128 a)
{
  const uint32x4_t sign = vshrq_n_u32(vreinterpretq_u32_f32(a), 31);
  const int32x4_t shifts = {0, 1, 2, 3};
  return static_cast<int>(vaddvq_u32(vshlq_u32(sign, shifts)));
}

#endif  // __aarch64__

#endif  // CAMXIOM__DETAIL__SIMD_NEON_COMPAT_HPP
