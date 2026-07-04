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

// Baseline projection microbenchmark for the float/double runtime hot path.
//
// Purpose: capture a *pre-refactor* performance baseline before the #1
// float/double template unification, so the unification can be proven
// performance-neutral (especially for TILTED14, whose tilt matrix is cached
// at model-build time and must NOT regress into per-point trig).
//
// It is intentionally dependency-light: it only links camxiom::core (Eigen)
// and uses the public projection API (float `rayToPixel`/`pixelToRay` and the
// double `*64` counterparts). No GTest, no Ceres.
//
// Build (isolated, optimised) — see CLAUDE.md / docs/TEMPLATE_UNIFICATION_PLAN.md:
//   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
//         -DCAMXIOM_BUILD_BENCHMARKS=ON -DCAMXIOM_WITH_CERES=OFF \
//         -DBUILD_TESTING=OFF -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE
//   cmake --build build-bench -j
//   ./build-bench/camxiom_projection_benchmark
//
// Methodology: a fixed-seed set of N rays is projected forward, the OK pixels
// are kept, then unprojected back. Each phase is timed best-of-R over the whole
// array; we report nanoseconds per call and millions of calls per second. A
// volatile sink + a doNotOptimize() barrier prevent the optimiser from
// discarding the work.

#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "camxiom/validated_model.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace
{

// --- Optimiser barrier ------------------------------------------------------
volatile double g_sink = 0.0;

template <typename T>
inline void doNotOptimize(const T &value)
{
  // Tell the compiler `value` is read and memory may be clobbered, so the
  // producing computation cannot be eliminated. Valid on x86-64 and aarch64.
  asm volatile("" : : "r,m"(value) : "memory");
}

// --- Tilt-matrix precompute (mirrors src/model/internal.hpp) ----------------
// We replicate the cached-tilt computation here so the benchmark exercises the
// *runtime* path (cached 3x3 matvec), which is exactly the behaviour the #1
// template unification must preserve for TILTED14.
void computeTiltMatrices(
  float tau_x, float tau_y, std::array<float, 9> &tilt, std::array<float, 9> &inv_tilt
)
{
  const float c_tx = std::cos(tau_x);
  const float s_tx = std::sin(tau_x);
  const float c_ty = std::cos(tau_y);
  const float s_ty = std::sin(tau_y);

  const std::array<float, 9> rot_xy{c_ty, s_ty * s_tx, -s_ty * c_tx, 0.0f,       c_tx,
                                    s_tx, s_ty,        -c_ty * s_tx, c_ty * c_tx};

  const float r22 = rot_xy[8];
  const std::array<float, 9> proj_z{r22, 0.0f, -rot_xy[2], 0.0f, r22, -rot_xy[5], 0.0f, 0.0f, 1.0f};

  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      float value = 0.0f;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += proj_z[row * 3 + mid] * rot_xy[mid * 3 + col];
      }
      tilt[row * 3 + col] = value;
    }
  }

  const float inv_r22 = 1.0f / r22;
  const std::array<float, 9> inv_proj_z{
    inv_r22, 0.0f, rot_xy[2] * inv_r22, 0.0f, inv_r22, rot_xy[5] * inv_r22, 0.0f, 0.0f, 1.0f};
  const std::array<float, 9> rot_xy_t{rot_xy[0], rot_xy[3], rot_xy[6], rot_xy[1], rot_xy[4],
                                      rot_xy[7], rot_xy[2], rot_xy[5], rot_xy[8]};
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      float value = 0.0f;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += rot_xy_t[row * 3 + mid] * inv_proj_z[mid * 3 + col];
      }
      inv_tilt[row * 3 + col] = value;
    }
  }
}

// --- Model factories --------------------------------------------------------
camxiom::CameraModel makePinholeRadtan5()
{
  camxiom::CameraModel m;
  m.intrinsics.fx = 520.0f;
  m.intrinsics.fy = 519.0f;
  m.intrinsics.cx = 322.0f;
  m.intrinsics.cy = 241.0f;
  m.projection.type = camxiom::ProjectionModelType::PINHOLE;
  m.distortion.type = camxiom::DistortionModelType::RADTAN5;
  m.distortion.space = camxiom::DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.28f;    // k1
  m.distortion.coeffs[1] = 0.11f;     // k2
  m.distortion.coeffs[2] = 0.0006f;   // p1
  m.distortion.coeffs[3] = -0.0004f;  // p2
  m.distortion.coeffs[4] = -0.02f;    // k3
  m.distortion.count = 5U;
  return m;
}

camxiom::CameraModel makeFisheyeKb4()
{
  camxiom::CameraModel m;
  m.intrinsics.fx = 285.0f;
  m.intrinsics.fy = 285.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = camxiom::ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = 3.1415926f - 1e-3f;
  m.distortion.type = camxiom::DistortionModelType::KB4;
  m.distortion.space = camxiom::DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = -0.012f;   // k1
  m.distortion.coeffs[1] = 0.004f;    // k2
  m.distortion.coeffs[2] = -0.0007f;  // k3
  m.distortion.coeffs[3] = 0.0001f;   // k4
  m.distortion.count = 4U;
  return m;
}

camxiom::CameraModel makePinholeTilted14()
{
  camxiom::CameraModel m;
  m.intrinsics.fx = 520.0f;
  m.intrinsics.fy = 519.0f;
  m.intrinsics.cx = 322.0f;
  m.intrinsics.cy = 241.0f;
  m.projection.type = camxiom::ProjectionModelType::PINHOLE;
  m.distortion.type = camxiom::DistortionModelType::TILTED14;
  m.distortion.space = camxiom::DistortionSpace::PLANE;
  // radial (rational) + tangential + thin-prism + tilt
  m.distortion.coeffs[0] = -0.28f;    // k1
  m.distortion.coeffs[1] = 0.11f;     // k2
  m.distortion.coeffs[2] = 0.0006f;   // p1
  m.distortion.coeffs[3] = -0.0004f;  // p2
  m.distortion.coeffs[4] = -0.02f;    // k3
  m.distortion.coeffs[5] = 0.001f;    // k4
  m.distortion.coeffs[6] = 0.0005f;   // k5
  m.distortion.coeffs[7] = -0.0002f;  // k6
  m.distortion.coeffs[8] = 0.0003f;   // s1
  m.distortion.coeffs[9] = -0.0002f;  // s2
  m.distortion.coeffs[10] = 0.0001f;  // s3
  m.distortion.coeffs[11] = 0.0002f;  // s4
  m.distortion.coeffs[12] = 0.02f;    // tau_x
  m.distortion.coeffs[13] = -0.015f;  // tau_y
  m.distortion.count = 14U;
  m.distortion.is_rational = true;
  m.distortion.has_thin_prism = true;
  m.distortion.has_tilt = true;
  computeTiltMatrices(
    m.distortion.coeffs[12], m.distortion.coeffs[13], m.distortion.tilt_matrix,
    m.distortion.inv_tilt_matrix
  );
  return m;
}

// --- Input generation -------------------------------------------------------
std::vector<Eigen::Vector3f> makeRays(std::size_t n, float half_extent, std::uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-half_extent, half_extent);
  std::vector<Eigen::Vector3f> rays;
  rays.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    Eigen::Vector3f r(dist(rng), dist(rng), 1.0f);
    rays.push_back(r.normalized());
  }
  return rays;
}

// --- Timing helpers ---------------------------------------------------------
struct Stat
{
  double fwd_ns{0.0};
  double inv_ns{0.0};
  std::size_t fwd_n{0};
  std::size_t inv_n{0};
};

template <typename Fn>
double bestNsPerCall(Fn &&body, std::size_t n_calls, int reps)
{
  double best = 1e300;
  for (int r = 0; r < reps; ++r)
  {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double per = ns / static_cast<double>(n_calls);
    if (per < best)
    {
      best = per;
    }
  }
  return best;
}

Stat benchFloat(
  const std::string &name, const camxiom::CameraModel &model,
  const std::vector<Eigen::Vector3f> &rays, int fwd_reps, int inv_reps
)
{
  using namespace camxiom;
  Stat s;
  s.fwd_n = rays.size();

  // Forward.
  s.fwd_ns = bestNsPerCall(
    [&]() {
      double acc = 0.0;
      for (const auto &ray : rays)
      {
        const PixelResult px = rayToPixel(model, ray);
        acc += px.pixel.u + px.pixel.v;
      }
      g_sink += acc;
      doNotOptimize(acc);
    },
    rays.size(), fwd_reps
  );

  // Collect OK pixels for the inverse phase.
  std::vector<Pixel2> pixels;
  pixels.reserve(rays.size());
  for (const auto &ray : rays)
  {
    const PixelResult px = rayToPixel(model, ray);
    if (px.status == StatusCode::OK)
    {
      pixels.push_back(px.pixel);
    }
  }
  s.inv_n = pixels.size();

  if (!pixels.empty())
  {
    s.inv_ns = bestNsPerCall(
      [&]() {
        double acc = 0.0;
        for (const auto &p : pixels)
        {
          const RayResult r = pixelToRay(model, p);
          acc += r.ray.direction.x() + r.ray.direction.y() + r.ray.direction.z();
        }
        g_sink += acc;
        doNotOptimize(acc);
      },
      pixels.size(), inv_reps
    );
  }

  std::printf(
    "  %-22s float   fwd %8.2f ns (%6.1f M/s)   inv %9.2f ns (%6.1f M/s)  [%zu/%zu ok]\n",
    name.c_str(), s.fwd_ns, 1e3 / s.fwd_ns, s.inv_ns, s.inv_ns > 0.0 ? 1e3 / s.inv_ns : 0.0,
    s.inv_n, s.fwd_n
  );
  return s;
}

// Same forward/inverse workload as benchFloat, but driven through a
// ValidatedCameraModel (C1). tryMake() validates + resolves the per-model entry
// points ONCE up front, so the timed loops skip the per-call validateCameraModel
// + dispatch switch that the generic rayToPixel/pixelToRay pay on every point.
// Comparing this "float/val" line to the generic "float" line above quantifies
// the re-validation cost removed by the validated-once type.
Stat benchValidatedFloat(
  const std::string &name, const camxiom::CameraModel &model,
  const std::vector<Eigen::Vector3f> &rays, int fwd_reps, int inv_reps
)
{
  using namespace camxiom;
  Stat s;
  s.fwd_n = rays.size();

  const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(model);
  if (!vm.has_value())
  {
    std::printf("  %-22s float/val  ERROR: tryMake rejected a valid model\n", name.c_str());
    return s;
  }

  s.fwd_ns = bestNsPerCall(
    [&]() {
      double acc = 0.0;
      for (const auto &ray : rays)
      {
        const PixelResult px = vm->rayToPixel(ray);
        acc += px.pixel.u + px.pixel.v;
      }
      g_sink += acc;
      doNotOptimize(acc);
    },
    rays.size(), fwd_reps
  );

  std::vector<Pixel2> pixels;
  pixels.reserve(rays.size());
  for (const auto &ray : rays)
  {
    const PixelResult px = vm->rayToPixel(ray);
    if (px.status == StatusCode::OK)
    {
      pixels.push_back(px.pixel);
    }
  }
  s.inv_n = pixels.size();

  if (!pixels.empty())
  {
    s.inv_ns = bestNsPerCall(
      [&]() {
        double acc = 0.0;
        for (const auto &p : pixels)
        {
          const RayResult r = vm->pixelToRay(p);
          acc += r.ray.direction.x() + r.ray.direction.y() + r.ray.direction.z();
        }
        g_sink += acc;
        doNotOptimize(acc);
      },
      pixels.size(), inv_reps
    );
  }

  std::printf(
    "  %-22s float/val fwd %8.2f ns (%6.1f M/s)   inv %9.2f ns (%6.1f M/s)  [%zu/%zu ok]\n",
    name.c_str(), s.fwd_ns, 1e3 / s.fwd_ns, s.inv_ns, s.inv_ns > 0.0 ? 1e3 / s.inv_ns : 0.0,
    s.inv_n, s.fwd_n
  );
  return s;
}

Stat benchDouble(
  const std::string &name, const camxiom::CameraModel &model_f,
  const std::vector<Eigen::Vector3f> &rays, int fwd_reps, int inv_reps
)
{
  using namespace camxiom;
  const CameraModel64 model = toCameraModel64(model_f);

  std::vector<Eigen::Vector3d> rays_d;
  rays_d.reserve(rays.size());
  for (const auto &r : rays)
  {
    rays_d.emplace_back(r.x(), r.y(), r.z());
  }

  Stat s;
  s.fwd_n = rays_d.size();

  s.fwd_ns = bestNsPerCall(
    [&]() {
      double acc = 0.0;
      for (const auto &ray : rays_d)
      {
        const PixelResult64 px = rayToPixel64(model, ray);
        acc += px.pixel.u + px.pixel.v;
      }
      g_sink += acc;
      doNotOptimize(acc);
    },
    rays_d.size(), fwd_reps
  );

  std::vector<Pixel2d> pixels;
  pixels.reserve(rays_d.size());
  for (const auto &ray : rays_d)
  {
    const PixelResult64 px = rayToPixel64(model, ray);
    if (px.status == StatusCode::OK)
    {
      pixels.push_back(px.pixel);
    }
  }
  s.inv_n = pixels.size();

  if (!pixels.empty())
  {
    s.inv_ns = bestNsPerCall(
      [&]() {
        double acc = 0.0;
        for (const auto &p : pixels)
        {
          const RayResult64 r = pixelToRay64(model, p);
          acc += r.ray.direction.x() + r.ray.direction.y() + r.ray.direction.z();
        }
        g_sink += acc;
        doNotOptimize(acc);
      },
      pixels.size(), inv_reps
    );
  }

  std::printf(
    "  %-22s double  fwd %8.2f ns (%6.1f M/s)   inv %9.2f ns (%6.1f M/s)  [%zu/%zu ok]\n",
    name.c_str(), s.fwd_ns, 1e3 / s.fwd_ns, s.inv_ns, s.inv_ns > 0.0 ? 1e3 / s.inv_ns : 0.0,
    s.inv_n, s.fwd_n
  );
  return s;
}

}  // namespace

int main()
{
  using namespace camxiom;

  constexpr std::size_t kN = 1U << 16;  // 65536 rays
  constexpr int kFwdReps = 200;
  constexpr int kInvReps = 50;

  const std::vector<Eigen::Vector3f> rays_narrow = makeRays(kN, 0.55f, 0xC0FFEEu);
  const std::vector<Eigen::Vector3f> rays_wide = makeRays(kN, 1.20f, 0xBEEF11u);

  const CameraModel pinhole = makePinholeRadtan5();
  const CameraModel fisheye = makeFisheyeKb4();
  const CameraModel tilted = makePinholeTilted14();

  std::printf("camxiom projection baseline microbenchmark\n");
  std::printf("  N=%zu rays, forward best-of-%d, inverse best-of-%d\n", kN, kFwdReps, kInvReps);
#if defined(__aarch64__)
  std::printf("  arch=aarch64 (scalar runtime path; x86 SIMD kernels inactive)\n");
#elif defined(__x86_64__)
  std::printf("  arch=x86_64\n");
#endif
  std::printf("  (lower ns is better; ok-count shows how many forward projections succeeded)\n\n");

  for (const auto &status :
       {validateCameraModel(pinhole), validateCameraModel(fisheye), validateCameraModel(tilted)})
  {
    if (status != StatusCode::OK)
    {
      std::printf("ERROR: a benchmark model failed validation: %s\n", toString(status));
      return 1;
    }
  }

  benchFloat("pinhole+radtan5", pinhole, rays_narrow, kFwdReps, kInvReps);
  benchValidatedFloat("pinhole+radtan5", pinhole, rays_narrow, kFwdReps, kInvReps);
  benchDouble("pinhole+radtan5", pinhole, rays_narrow, kFwdReps, kInvReps);
  benchFloat("fisheye+kb4", fisheye, rays_wide, kFwdReps, kInvReps);
  benchValidatedFloat("fisheye+kb4", fisheye, rays_wide, kFwdReps, kInvReps);
  benchDouble("fisheye+kb4", fisheye, rays_wide, kFwdReps, kInvReps);
  benchFloat("pinhole+tilted14", tilted, rays_narrow, kFwdReps, kInvReps);
  benchValidatedFloat("pinhole+tilted14", tilted, rays_narrow, kFwdReps, kInvReps);
  benchDouble("pinhole+tilted14", tilted, rays_narrow, kFwdReps, kInvReps);

  std::printf("\n  (sink=%.3f)\n", static_cast<double>(g_sink));
  return 0;
}
