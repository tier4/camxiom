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

#include "camxiom/calib/convert.hpp"

#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types64.hpp"
#include "optimizer/pnp/pnp_parameter_bounds.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace camxiom::calib
{

ModelConversionResult convertCameraModel(
  const CameraModel &src_model, const int image_width, const int image_height,
  const CameraModel &dst_seed, const ModelConversionOptions &options
)
{
  ModelConversionResult result;

  if (image_width <= 0 || image_height <= 0 || options.grid_cols < 2 ||
      options.grid_rows < 2 || options.max_iterations <= 0)
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  // Grid sizing in 64-bit: the caller-supplied cols x rows product overflows
  // int for pathological dimensions (same guard family as the LUT/remap
  // pixel-count checks).
  const long long grid_point_count =
    static_cast<long long>(options.grid_cols) * static_cast<long long>(options.grid_rows);
  if (grid_point_count > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }
  if (validateCameraModel(src_model) != StatusCode::OK || validateCameraModel(dst_seed) != StatusCode::OK)
  {
    result.status = StatusCode::INVALID_MODEL;
    return result;
  }

  // ------------------------------------------------------------------
  // Synthetic correspondences: source pixel grid -> unit rays. Unit rays
  // double as 3D points at distance 1 for the pose-fixed fit below.
  // ------------------------------------------------------------------
  const CameraModel64 src64 = toCameraModel64(src_model);

  optimizer::ObjectPoints rays;
  optimizer::ImagePoints pixels;
  rays.reserve(static_cast<std::size_t>(grid_point_count));
  pixels.reserve(rays.capacity());

  for (int gy = 0; gy < options.grid_rows; ++gy)
  {
    for (int gx = 0; gx < options.grid_cols; ++gx)
    {
      const double u = (static_cast<double>(image_width) - 1.0) * static_cast<double>(gx) /
                       static_cast<double>(options.grid_cols - 1);
      const double v = (static_cast<double>(image_height) - 1.0) * static_cast<double>(gy) /
                       static_cast<double>(options.grid_rows - 1);
      const RayResult64 rr = pixelToRay64(src64, Pixel2d{u, v});
      if (rr.status != StatusCode::OK)
      {
        continue;  // outside the source FOV
      }
      rays.emplace_back(rr.ray.direction.normalized());
      pixels.emplace_back(u, v);
    }
  }
  result.used_point_count = static_cast<int>(rays.size());
  if (rays.size() < 8U)
  {
    result.status = StatusCode::DEGENERATE_CONFIG;
    return result;
  }

  // ------------------------------------------------------------------
  // Fit: single synthetic view, pose pinned to identity, all intrinsic /
  // distortion / projection parameters of the destination family free.
  // Ordinary least squares (huber off): synthetic data has no outliers.
  // ------------------------------------------------------------------
  optimizer::PnpInitialGuess guess;
  guess.camera_model = dst_seed;
  guess.rvecs = {Eigen::Vector3d::Zero()};
  guess.tvecs = {Eigen::Vector3d::Zero()};

  optimizer::PnpSolverOptions solver_opts;
  solver_opts.solver_options.max_num_iterations = options.max_iterations;
  solver_opts.huber_loss_delta = 0.0;
  // The fitted focal length tracks the SOURCE geometry, so widen against the
  // longer of the seed and the source focal lengths: a telephoto source with
  // a default (short) seed must not clamp at the stock 5000 px bound.
  // Magnitudes only — a mirrored (negative-focal) source must widen by the
  // same amount as its positive twin.
  optimizer::detail::widenDefaultPnpUpperBounds(
    solver_opts, image_width, image_height,
    std::max(
      std::abs(static_cast<double>(dst_seed.intrinsics.fx)),
      std::abs(static_cast<double>(src_model.intrinsics.fx))
    ),
    std::max(
      std::abs(static_cast<double>(dst_seed.intrinsics.fy)),
      std::abs(static_cast<double>(src_model.intrinsics.fy))
    )
  );

  optimizer::PnpSolver solver;
  optimizer::PnpResult pnp;
  const bool solved =
    solver.solve(rays, pixels, guess, pnp, solver_opts, optimizer::PnpFlag::FIX_EXTRINSICS);

  // Best-effort model even when the solve reports failure (the caller can
  // still inspect the fit error).
  result.camera_model = pnp.camera_model;

  // ------------------------------------------------------------------
  // Fit quality on the same grid, through the fitted model.
  // ------------------------------------------------------------------
  const CameraModel64 fit64 = toCameraModel64(result.camera_model);
  double sum_sq = 0.0;
  double max_err = 0.0;
  int representable = 0;
  for (std::size_t j = 0; j < rays.size(); ++j)
  {
    const PixelResult64 pr = rayToPixel64(fit64, rays[j]);
    if (pr.status != StatusCode::OK)
    {
      continue;  // outside what the destination family can express
    }
    ++representable;
    const double du = pr.pixel.u - pixels[j].x();
    const double dv = pr.pixel.v - pixels[j].y();
    const double err_sq = du * du + dv * dv;
    sum_sq += err_sq;
    max_err = (std::max)(max_err, std::sqrt(err_sq));
  }
  result.representable_point_count = representable;
  if (representable > 0)
  {
    result.rms_fit_error_px = std::sqrt(sum_sq / static_cast<double>(representable));
    result.max_fit_error_px = max_err;
  }

  // OK is documented as "the fit CONVERGED". A run that stopped at
  // max_iterations can still be solution-usable (solved == true) without
  // meeting the tolerances — report that honestly as NON_CONVERGED. One
  // escape: an exact model pair drives the cost to machine precision, where
  // Ceres's RELATIVE function tolerance can never fire (|dcost|/cost stays
  // O(1) on rounding noise) and the solver "exhausts" its iterations on an
  // already-perfect fit. Treat an iteration-capped fit as converged when the
  // independently measured rms is at numerical precision: 1e-3 px is far
  // below any physical calibration residual yet safely above the
  // float-storage rounding noise of the returned CameraModel (~1e-4 px).
  // (rms is +inf when representable == 0, so the escape cannot misfire.)
  //
  // The escape additionally requires a clear margin between sample count and
  // the fit's degrees of freedom: near the 8-point floor a high-DOF
  // destination (pinhole+RATIONAL8 = 12 free parameters vs 16 residual
  // components) could over-fit the grid to machine precision while being
  // wrong off-grid. Two points per free parameter = four residuals per
  // unknown; the default 24x18 grid satisfies this for every family.
  constexpr double kNumericallyExactRmsPx = 1e-3;
  const int free_param_count = 4 +
                               optimizer::detail::projectionParamCount(dst_seed.projection.type) +
                               static_cast<int>(dst_seed.distortion.count);
  const bool numerically_exact =
    result.rms_fit_error_px <= kNumericallyExactRmsPx && representable >= 2 * free_param_count;
  const bool fit_converged = solver.lastSummary().converged || numerically_exact;
  result.status =
    (solved && fit_converged && representable >= 8) ? StatusCode::OK : StatusCode::NON_CONVERGED;
  return result;
}

}  // namespace camxiom::calib
