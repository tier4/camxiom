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

#ifndef CAMXIOM__BATCH__BATCH_PARALLEL_HPP
#define CAMXIOM__BATCH__BATCH_PARALLEL_HPP

// OpenMP chunking for the single-threaded SIMD batch kernels.
//
// The scalar fallback loops in batch/{float,double}.cpp have always been
// `#pragma omp parallel for`, but the SIMD kernels were called directly and
// therefore ran on ONE core. On a many-core machine that inverted the
// performance order: the "optimised" SIMD path lost to the scalar path by
// roughly cores/simd_speedup (measured 2-18x slower on a 12-core Jetson
// Orin). Wrapping each kernel call in per-thread chunks composes the two
// speedups instead of trading one for the other.
//
// The kernels are stateless and write disjoint per-lane outputs, so chunking
// is race-free by construction; valid counts are summed via reduction.

#include "camxiom/types.hpp"

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

namespace camxiom::detail
{

inline StatusCode *offsetStatuses(StatusCode *statuses, const int begin)
{
  return statuses == nullptr ? nullptr : statuses + begin;
}

/// Run `kernel(begin, len)` over [0, count), in parallel chunks when OpenMP
/// is available and the batch is large enough for the fork/join overhead to
/// pay off. The kernel must return its chunk's valid-lane count.
template <typename Kernel>
inline int runBatchKernelParallel(const int count, const Kernel &kernel)
{
#ifdef CAMXIOM_HAS_OPENMP
  // Below this size the OpenMP fork/join overhead outweighs the win.
  constexpr int kParallelThreshold = 8192;
  const int num_threads = omp_get_max_threads();
  if (count >= kParallelThreshold && num_threads > 1)
  {
    // One chunk per thread, rounded up to whole 8-lane groups so the AVX2
    // main bodies stay full-width everywhere except the final chunk.
    const int per_thread = (count + num_threads - 1) / num_threads;
    const int chunk = ((per_thread + 7) / 8) * 8;
    const int num_chunks = (count + chunk - 1) / chunk;
    int valid_count = 0;
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
    for (int c = 0; c < num_chunks; ++c)
    {
      const int begin = c * chunk;
      const int len = (count - begin) < chunk ? (count - begin) : chunk;
      valid_count += kernel(begin, len);
    }
    return valid_count;
  }
#endif
  return kernel(0, count);
}

}  // namespace camxiom::detail

#endif  // CAMXIOM__BATCH__BATCH_PARALLEL_HPP
