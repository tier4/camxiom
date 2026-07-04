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

// calibrate() timing benchmark — isolates the C3/C4 diagnostic overhead.
//
// calibrate() now optionally computes the C3 parameter uncertainty + C4
// observability diagnostic (CalibrationOptions::estimate_uncertainty, default
// on). This benchmark times a full calibrate() with the diagnostic ON vs OFF
// on identical inputs (so the Ceres solve — same iteration count either way —
// cancels out), and reports the delta as the measured diagnostic overhead.
//
// Unlike projection_benchmark (core-only, Ceres-free), this links camxiom_calib
// and therefore requires Ceres. Build (isolated, optimised):
//   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release \
//         -DCAMXIOM_BUILD_BENCHMARKS=ON -DBUILD_TESTING=OFF \
//         -DCMAKE_DISABLE_FIND_PACKAGE_ament_cmake=TRUE
//   cmake --build build-bench -j && ./build-bench/camxiom_calibrate_benchmark
// (or via colcon: colcon build --packages-select camxiom
//   --cmake-args -DCMAKE_BUILD_TYPE=Release -DCAMXIOM_BUILD_BENCHMARKS=ON,
//   then run build/camxiom/camxiom_calibrate_benchmark).
//
// Deterministic: fixed std::mt19937 seed, sigma=0.3 px corner noise.

#include "camxiom/calib/intrinsics.hpp"
#include "camxiom/default_seed.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace
{

using namespace camxiom;
using camxiom::calib::CalibrationOptions;
using camxiom::calib::CalibrationResult;
using camxiom::calib::CalibrationView;
using camxiom::optimizer::PnpFlag;

volatile double g_sink = 0.0;

constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d rotX(double a)
{
  Eigen::Matrix3d r;
  r << 1, 0, 0, 0, std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a);
  return r;
}
Eigen::Matrix3d rotY(double a)
{
  Eigen::Matrix3d r;
  r << std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a);
  return r;
}
Eigen::Matrix3d rotZ(double a)
{
  Eigen::Matrix3d r;
  r << std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1;
  return r;
}

std::vector<Eigen::Vector3d> makeBoard(int rows, int cols, double spacing)
{
  std::vector<Eigen::Vector3d> pts;
  pts.reserve(static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      pts.emplace_back(c * spacing, r * spacing, 0.0);
    }
  }
  return pts;
}

// Deterministic, perspective-diverse poses: perturbations of a proven canonical
// tilted set so every board stays liftable by the seed's DLT init.
std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> makePoses(std::size_t m)
{
  const double d30 = 30.0 * kPi / 180.0;
  const double d25 = 25.0 * kPi / 180.0;
  const double d20 = 20.0 * kPi / 180.0;
  const double d15 = 15.0 * kPi / 180.0;
  const std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> base = {
    {Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.0, 0.0, 0.30)},
    {rotX(+d30), Eigen::Vector3d(0.0, -0.15, 0.30)},
    {rotY(-d30), Eigen::Vector3d(-0.15, 0.0, 0.35)},
    {rotX(+d25) * rotY(-d25), Eigen::Vector3d(0.1, 0.1, 0.35)},
    {rotX(+d20) * rotY(+d20) * rotZ(+d15), Eigen::Vector3d(-0.1, 0.12, 0.40)},
  };
  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> poses;
  poses.reserve(m);
  for (std::size_t i = 0; i < m; ++i)
  {
    auto p = base[i % base.size()];
    // Small deterministic jitter for the repeated copies so they are not exact
    // duplicates (translation only; keeps liftability).
    const double k = static_cast<double>(i / base.size());
    p.second += Eigen::Vector3d(0.01 * k, -0.008 * k, 0.02 * k);
    poses.push_back(p);
  }
  return poses;
}

std::vector<CalibrationView> makeViews(
  const CameraModel64 &gt, const std::vector<Eigen::Vector3d> &world, std::size_t m,
  double noise_sigma, std::mt19937 &rng
)
{
  const auto poses = makePoses(m);
  std::normal_distribution<double> noise(0.0, noise_sigma);
  std::vector<CalibrationView> views;
  views.reserve(m);
  for (const auto &pose : poses)
  {
    const Eigen::Matrix3d &R = pose.first;
    const Eigen::Vector3d &t = pose.second;
    CalibrationView v;
    v.world_points = world;
    v.image_points.reserve(world.size());
    bool ok = true;
    for (const auto &Pw : world)
    {
      const PixelResult64 pr = rayToPixel64(gt, R * Pw + t);
      if (pr.status != StatusCode::OK)
      {
        ok = false;
        break;
      }
      double u = pr.pixel.u, vv = pr.pixel.v;
      if (noise_sigma > 0.0)
      {
        u += noise(rng);
        vv += noise(rng);
      }
      v.image_points.emplace_back(u, vv);
    }
    if (!ok || v.image_points.size() != world.size())
    {
      return {};
    }
    views.push_back(std::move(v));
  }
  return views;
}

CameraModel64 makeGtPinholeRadtan5()
{
  CameraModel64 m{};
  m.intrinsics.fx = 500.0;
  m.intrinsics.fy = 500.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = kPi / 2.0;
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = -0.10;
  m.distortion.coeffs[1] = 0.04;
  m.distortion.coeffs[2] = 0.0006;
  m.distortion.coeffs[3] = -0.0004;
  m.distortion.coeffs[4] = 0.008;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

CameraModel makeSeedPinholeRadtan5()
{
  CameraModel m{};
  m.intrinsics.fx = 480.0f;
  m.intrinsics.fy = 480.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.intrinsics.skew = 0.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(kPi / 2.0);
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.count = 5U;
  m.distortion.coeffs.fill(0.0f);
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

CameraModel64 makeGtKb4()
{
  CameraModel64 m{};
  m.intrinsics.fx = 200.0;
  m.intrinsics.fy = 200.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.intrinsics.skew = 0.0;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = kPi;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs.fill(0.0);
  m.distortion.coeffs[0] = 0.01;
  m.distortion.coeffs[1] = -0.005;
  m.distortion.tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  m.distortion.inv_tilt_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  return m;
}

template <typename Fn>
double bestMs(Fn &&body, int reps)
{
  double best = 1e300;
  for (int r = 0; r < reps; ++r)
  {
    const auto t0 = std::chrono::steady_clock::now();
    body();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best)
    {
      best = ms;
    }
  }
  return best;
}

void runCase(
  const char *name, const CameraModel64 &gt, const CameraModel &seed, PnpFlag lock,
  std::size_t m_views, int reps
)
{
  const auto world = makeBoard(8, 6, 0.05);
  std::mt19937 rng(42U);
  const auto views = makeViews(gt, world, m_views, 0.3, rng);
  if (views.empty())
  {
    std::printf("  %-18s views=%2zu  ERROR: view synthesis failed\n", name, m_views);
    return;
  }

  CalibrationOptions opts;
  opts.image_width = 640;
  opts.image_height = 480;
  opts.max_iterations = 100;

  // Warm up (allocations, first-touch) and capture the ON result for reporting.
  opts.estimate_uncertainty = true;
  CalibrationResult r_on = camxiom::calib::calibrate(views, seed, lock, opts);
  g_sink += r_on.rms_reprojection_error_px;

  opts.estimate_uncertainty = false;
  const double t_off = bestMs(
    [&]() {
      const CalibrationResult r = camxiom::calib::calibrate(views, seed, lock, opts);
      g_sink += r.rms_reprojection_error_px;
    },
    reps
  );

  opts.estimate_uncertainty = true;
  const double t_on = bestMs(
    [&]() {
      const CalibrationResult r = camxiom::calib::calibrate(views, seed, lock, opts);
      g_sink += r.rms_reprojection_error_px;
    },
    reps
  );

  const double overhead_ms = t_on - t_off;
  const double pct = (t_off > 0.0) ? (100.0 * overhead_ms / t_off) : 0.0;
  const std::size_t npts = m_views * world.size();

  std::printf(
    "  %-18s views=%2zu pts=%4zu p=%2zu  status=%d  calib(off)=%7.3f ms  "
    "calib(on)=%7.3f ms  diag=%+7.3f ms (%+5.1f%%)  obs_ok=%d\n",
    name, m_views, npts, r_on.uncertainty_labels.size(), static_cast<int>(r_on.status), t_off, t_on,
    overhead_ms, pct, r_on.observability_ok ? 1 : 0
  );
}

}  // namespace

int main()
{
  std::printf("camxiom calibrate() C3/C4 diagnostic-overhead benchmark\n");
#if defined(__aarch64__)
  std::printf("  arch=aarch64\n");
#elif defined(__x86_64__)
  std::printf("  arch=x86_64\n");
#endif
  std::printf(
    "  (diag = calibrate ON minus OFF on identical inputs = C3+C4 cost; "
    "best-of-reps)\n\n"
  );

  const CameraModel64 gt_pin = makeGtPinholeRadtan5();
  const CameraModel seed_pin = makeSeedPinholeRadtan5();
  const CameraModel64 gt_kb4 = makeGtKb4();
  const CameraModel seed_kb4 = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, 640, 480);

  const int reps = 15;
  for (std::size_t m : {std::size_t{5}, std::size_t{20}, std::size_t{50}})
  {
    runCase("pinhole+radtan5", gt_pin, seed_pin, PnpFlag::NONE, m, reps);
  }
  for (std::size_t m : {std::size_t{5}, std::size_t{20}, std::size_t{50}})
  {
    runCase("fisheye+kb4", gt_kb4, seed_kb4, PnpFlag::FIX_DIST_2 | PnpFlag::FIX_DIST_3, m, reps);
  }

  std::printf("\n  (sink=%.3f)\n", static_cast<double>(g_sink));
  return 0;
}
