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

#include "detail/simd_dispatch.hpp"

#include <cstdlib>

namespace camxiom::detail
{

namespace
{

bool detectAvx2()
{
#if defined(CAMXIOM_HAS_AVX2)
  // Escape hatch. Compiling the AVX2 kernels into every x86 build means that on
  // any modern machine — including CI — the SSE2 kernels would otherwise never
  // execute, so the scalar-vs-SIMD parity tests would stop covering them
  // entirely. Setting CAMXIOM_DISABLE_AVX2 to a non-empty value other than "0"
  // pins the dispatch to SSE2 for the whole process, which is what the
  // sse2-baseline CI variant does. It is also a first diagnostic step when a
  // result differs between machines.
  const char *disable = std::getenv("CAMXIOM_DISABLE_AVX2");
  if (disable != nullptr && disable[0] != '\0' && !(disable[0] == '0' && disable[1] == '\0'))
  {
    return false;
  }

  // __builtin_cpu_init() is called implicitly by __builtin_cpu_supports on the
  // compilers that provide it. FMA is checked alongside AVX2 because the
  // kernels are compiled for target("avx2,fma") and use FMA intrinsics; the two
  // ship together on every CPU we target, but a mismatch would fault rather
  // than merely run slower.
  return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
#else
  return false;
#endif
}

}  // namespace

bool cpuHasAvx2()
{
  // Function-local static: initialised once, thread-safe since C++11, and the
  // guard variable is a predictable branch on every later call.
  static const bool supported = detectAvx2();
  return supported;
}

}  // namespace camxiom::detail
