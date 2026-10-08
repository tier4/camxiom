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

#ifndef CAMXIOM__DETAIL__SIMD_DISPATCH_HPP
#define CAMXIOM__DETAIL__SIMD_DISPATCH_HPP

// Which SIMD kernels get compiled, and how one is chosen at run time.
//
// SSE2 is part of the x86-64 baseline, so its kernels always compile there;
// aarch64 reaches the same 4-wide kernels through the __m128 -> NEON mapping in
// simd_neon_compat.hpp.
//
// AVX2 used to be gated on __AVX2__, which is only defined when the whole build
// is given -mavx2. Neither this CMakeLists nor a consumer's default build does
// that, so in practice the AVX2 kernels compiled to nothing outside the CI
// variant that sets the flag explicitly — a package built and installed the
// normal way ran SSE2 on every machine. Raising the baseline instead is not an
// option for a library that ships: the resulting binary would fault on any CPU
// without AVX2.
//
// So the AVX2 kernels are now compiled on every x86 build, each function
// carrying `__attribute__((target("avx2,fma")))` so the compiler emits AVX2 for
// that function alone while the rest of the translation unit stays baseline,
// and the batch layer picks between the AVX2 and SSE2 entry points at run time
// via cpuHasAvx2(). One binary, fast where the CPU allows it, correct where it
// does not.
//
// The attribute is a GCC/Clang extension; on any other x86 compiler the AVX2
// kernels stay out of the build and cpuHasAvx2() reports false, which is the
// pre-existing behaviour.
//
// Building with -mavx2 -mfma still works and is still what the CI `avx2`
// variant does: the attribute then matches the command line and changes
// nothing, while cpuHasAvx2() keeps the dispatch honest.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CAMXIOM_X86 1
#endif

#if defined(__SSE2__)
#define CAMXIOM_HAS_SSE2 1
#include <emmintrin.h>
#include <xmmintrin.h>
#elif defined(__aarch64__)
// The 4-wide kernels also run natively on AArch64 through a minimal
// __m128 -> NEON mapping (see simd_neon_compat.hpp for scope and caveats).
#define CAMXIOM_HAS_SSE2 1
#include "detail/simd_neon_compat.hpp"
#endif

#if defined(__SSE4_1__)
#define CAMXIOM_HAS_SSE41 1
#include <smmintrin.h>
#endif

#if defined(CAMXIOM_X86) && (defined(__GNUC__) || defined(__clang__))
#define CAMXIOM_HAS_AVX2 1
#include <immintrin.h>
#if defined(__AVX2__) && defined(__FMA__)
// The translation unit is already being built for AVX2+FMA, so the attribute
// would be a no-op. Leaving it off keeps the -mavx2 build byte-comparable with
// what it produced before runtime dispatch existed.
#define CAMXIOM_TARGET_AVX2
#else
#define CAMXIOM_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif
#endif

namespace camxiom::detail
{

/// True when the CPU executing this process supports the AVX2 + FMA the AVX2
/// kernels are compiled for. Always false where those kernels were not
/// compiled, and always false when the environment variable
/// CAMXIOM_DISABLE_AVX2 is set to a non-empty value other than "0" (see
/// simd_dispatch.cpp: without that switch the SSE2 kernels would never execute
/// on a modern machine, so nothing would test them).
///
/// The answer is a property of the process, so it is resolved once and cached;
/// callers may treat it as free. It follows that changing the environment
/// variable mid-process has no effect.
bool cpuHasAvx2();

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__SIMD_DISPATCH_HPP
