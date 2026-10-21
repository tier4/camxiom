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

// Checked single-point templates, distinct from the runtime/batch benchmark.
// Build in Release with CAMXIOM_BUILD_BENCHMARKS=ON. No Ceres dependency.
// Compare revisions with the same compiler/flags and CPU affinity; alternate
// processes to limit clock/load drift. Times are medians of nine samples.
#include "camxiom/projection_template.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
volatile double sink = 0;

template <typename T, typename Scalar>
void run(bool fisheye)
{
  camxiom::CameraModelT<Scalar> model;
  model.intrinsics.fx = Scalar(520);
  model.intrinsics.fy = Scalar(519);
  model.intrinsics.cx = Scalar(320);
  model.intrinsics.cy = Scalar(240);
  model.projection.type =
    fisheye ? camxiom::ProjectionModelType::FISHEYE_THETA : camxiom::ProjectionModelType::PINHOLE;
  model.projection.theta_max = Scalar(2.0);
  model.distortion.type =
    fisheye ? camxiom::DistortionModelType::KB4 : camxiom::DistortionModelType::RADTAN5;
  model.distortion.space =
    fisheye ? camxiom::DistortionSpace::ANGLE : camxiom::DistortionSpace::PLANE;
  model.distortion.count = fisheye ? 4 : 5;
  model.distortion.coeffs[0] = Scalar(fisheye ? .01 : -.28);
  model.distortion.coeffs[1] = Scalar(fisheye ? -.005 : .11);
  if (!fisheye)
  {
    model.distortion.coeffs[2] = Scalar(.0006);
    model.distortion.coeffs[3] = Scalar(-.0004);
    model.distortion.coeffs[4] = Scalar(-.02);
  }
#if defined(__GNUC__) || defined(__clang__)
  // Model values normally arrive from runtime configuration. Prevent the
  // compiler from replacing this fixture's validation and coefficients with
  // compile-time constants; no barrier is placed in the measured point loop.
  asm volatile("" : : "g"(&model) : "memory");
#endif
  constexpr std::size_t count = 65536;
  std::vector<std::array<T, 3>> rays(count);
  for (std::size_t i = 0; i < count; ++i)
    rays[i] = {T(int(i % 251) - 125) / T(150), T(int(i % 127) - 63) / T(150), T(1)};

  std::uint64_t valid = 0;
  double checksum = 0;
  constexpr int sweeps = 20;
  auto sweep = [&]() {
    valid = 0;
    checksum = 0;
    for (int repeat = 0; repeat < sweeps; ++repeat)
      for (const auto &ray : rays)
      {
        T u, v;
        valid += camxiom::projection_template::projectGeneric(model, ray[0], ray[1], ray[2], u, v);
        checksum += double(u) + double(v);
      }
    sink = checksum;
  };
  sweep();
  std::array<double, 9> times;
  for (auto &ns : times)
  {
    const auto start = std::chrono::steady_clock::now();
    sweep();
    ns =
      std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
      (count * sweeps);
  }
  std::sort(times.begin(), times.end());
  std::cout << (fisheye ? "kb4" : "radtan5") << ',' << sizeof(Scalar) * 8 << ',' << sizeof(T) * 8
            << ',' << times[times.size() / 2] << ',' << valid << ',' << checksum << '\n';
}
}  // namespace

int main()
{
  std::cout << std::setprecision(17)
            << "model,model_bits,scalar_bits,ns_per_point,valid,checksum\n";
  for (bool fish : {false, true})
  {
    run<float, float>(fish);
    run<double, float>(fish);
    run<double, double>(fish);
  }
}
