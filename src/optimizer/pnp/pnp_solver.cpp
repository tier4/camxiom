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

#include "camxiom/optimizer/pnp_solver.hpp"

#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/projection_template.hpp"
#include "camxiom/types64.hpp"
#include "model/distortion_aux.hpp"
#include "optimizer/pnp/pnp_cost_analytical_batch.hpp"
#include "optimizer/pnp/pnp_gauss_newton.hpp"
#include "optimizer/pnp/pnp_parameter_bounds.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace camxiom::optimizer
{
namespace
{

namespace tpl = camxiom::projection_template;

void makeSubsetManifold(
  ceres::Problem &problem, double *ptr, int dim, const std::vector<int> &fixed_indices
)
{
  if (fixed_indices.empty())
  {
    return;
  }
#if CERES_VERSION_MAJOR * 100 + CERES_VERSION_MINOR >= 201
  auto *manifold = new ceres::SubsetManifold(dim, fixed_indices);
  problem.SetManifold(ptr, manifold);
#else
  // Ceres < 2.1 predates the Manifold API (e.g. the 2.0.0 shipped in Ubuntu
  // 22.04 / JetPack apt repos). SubsetParameterization is the exact
  // equivalent: same constructor shape, and Problem takes ownership.
  auto *parameterization = new ceres::SubsetParameterization(dim, fixed_indices);
  problem.SetParameterization(ptr, parameterization);
#endif
}

/// Apply one scalar's box bound (from the shared descriptor) to a parameter
/// block index. A default (unbounded) ScalarBound is a no-op, so this can be
/// called uniformly for every free scalar. The order of the independent
/// SetParameter{Upper,Lower}Bound calls does not affect the resulting problem.
void applyScalarBound(
  ceres::Problem &problem, double *block, int index, const detail::ScalarBound &bound
)
{
  if (bound.has_upper)
  {
    problem.SetParameterUpperBound(block, index, bound.upper);
  }
  if (bound.has_lower)
  {
    problem.SetParameterLowerBound(block, index, bound.lower);
  }
}

void eigenToArray(const std::vector<Eigen::Vector3d> &src, std::vector<std::array<double, 3>> &dst)
{
  dst.resize(src.size());
  for (std::size_t i = 0; i < src.size(); ++i)
  {
    dst[i][0] = src[i][0];
    dst[i][1] = src[i][1];
    dst[i][2] = src[i][2];
  }
}

void arrayToEigen(const std::vector<std::array<double, 3>> &src, std::vector<Eigen::Vector3d> &dst)
{
  dst.resize(src.size());
  for (std::size_t i = 0; i < src.size(); ++i)
  {
    dst[i] = Eigen::Vector3d(src[i][0], src[i][1], src[i][2]);
  }
}

int effectiveDistCount(const camxiom::CameraModel &model)
{
  int count = static_cast<int>(model.distortion.count);
  if (count <= 0)
  {
    return 1;  // Ceres needs at least 1 parameter per block
  }
  return std::min(count, 14);
}

// projectionParamCount moved to optimizer/pnp/pnp_parameter_bounds.hpp so
// degrees-of-freedom reasoning outside this TU shares the same source.

std::vector<int> fixedProjectionParamIndices(camxiom::ProjectionModelType type)
{
  switch (type)
  {
    case camxiom::ProjectionModelType::OMNIDIRECTIONAL:
      return {1, 2};
    case camxiom::ProjectionModelType::DOUBLE_SPHERE:
      return {2};
    case camxiom::ProjectionModelType::EUCM:
      return {0};
    case camxiom::ProjectionModelType::PINHOLE:
    case camxiom::ProjectionModelType::FISHEYE_THETA:
    case camxiom::ProjectionModelType::UNKNOWN:
      break;
  }
  return {0, 1, 2};
}

constexpr double kInvalidPenaltyScale = 1e3;

// Distortion aux-state rebuild (is_rational / has_thin_prism / has_tilt /
// tilt matrices) is shared with the compat factory and the analytical double
// cost via camxiom::detail::rebuildDistortionAuxState in
// model/distortion_aux.hpp -- a single source so the thresholds cannot drift
// between the solver-internal model and factory-built models.

// ======================== Cost Functor ========================

struct PnpCost
{
  PnpCost(
    const Eigen::Vector3d &point3d, const Eigen::Vector2d &point2d,
    const camxiom::CameraModel &camera_model
  )
  : point3d(point3d), point2d(point2d), camera_model(camera_model)
  {
  }

  template <class T>
  bool operator()(
    const T *const rvec, const T *const tvec, const T *const focal_lengths,
    const T *const principal_points, const T *const dist_coeffs, const T *const proj_params,
    T *residual
  ) const
  {
    const T p[3] = {T(point3d.x()), T(point3d.y()), T(point3d.z())};
    T rp[3];
    ceres::AngleAxisRotatePoint(rvec, p, rp);

    const T x_cam = rp[0] + tvec[0];
    const T y_cam = rp[1] + tvec[1];
    const T z_cam = rp[2] + tvec[2];

    const T sk = T(camera_model.intrinsics.skew);
    const T intrinsics[4] = {
      focal_lengths[0], focal_lengths[1], principal_points[0], principal_points[1]};
    T u, v;
    const bool ok = tpl::projectGenericParametric(
      camera_model.projection.type, camera_model.distortion.type, intrinsics, dist_coeffs,
      static_cast<int>(camera_model.distortion.count),
      proj_params[0],  // xi
      proj_params[1],  // alpha
      proj_params[2],  // beta
      x_cam, y_cam, z_cam, u, v
    );
    if (!ok)
    {
      // Large constant penalty. For AutoDiff, a z_cam-based gradient is
      // misleading for DS/EUCM/Omni (their invalid condition depends on
      // xi/alpha/denom, not z_cam alone) and causes severe convergence
      // regression. The constant penalty works because valid-region
      // gradients from other points naturally pull the solver away.
      residual[0] = T(1e4);
      residual[1] = T(1e4);
      return true;
    }

    // Apply skew correction: u_pixel = fx*xd + sk*yd + cx, but template
    // functions output u = fx*xd + cx. We correct: u += sk * ((v - cy) / fy).
    const T yd = (v - principal_points[1]) / focal_lengths[1];
    u = u + sk * yd;

    residual[0] = T(point2d.x()) - u;
    residual[1] = T(point2d.y()) - v;
    return true;
  }

  const Eigen::Vector3d point3d;
  const Eigen::Vector2d point2d;
  const camxiom::CameraModel camera_model;
};

ceres::CostFunction *createCost(
  const Eigen::Vector3d &point3d, const Eigen::Vector2d &point2d,
  const camxiom::CameraModel &camera_model, int dist_count
)
{
  switch (dist_count)
  {
    case 0:
    case 1:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 1, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 2:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 2, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 3:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 3, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 4:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 4, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 5:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 5, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 6:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 6, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 7:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 7, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 8:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 8, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 9:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 9, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 10:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 10, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 11:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 11, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 12:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 12, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 13:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 13, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
    case 14:
    default:
      return new ceres::AutoDiffCostFunction<PnpCost, 2, 3, 3, 2, 2, 14, 3>(
        new PnpCost(point3d, point2d, camera_model)
      );
  }
}

}  // namespace

// ========================= PnpSolver::Impl =========================

// All Ceres state and the optimization logic live here, fully hidden from the
// public header. PnpSolver forwards to this PIMPL.
class PnpSolver::Impl
{
public:
  Impl();

  bool solve(
    const ObjectPointSets &object_points, const ImagePointSets &image_points,
    const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
    PnpFlag flags
  );

  bool solve(
    const ObjectPoints &object_points, const ImagePoints &image_points,
    const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
    PnpFlag flags
  );

  const PnpSummary &lastSummary() const { return last_summary_; }

private:
  void setupProblem(
    const camxiom::CameraModel &camera_model, const std::vector<Eigen::Vector3d> &rvecs,
    const std::vector<Eigen::Vector3d> &tvecs, const PnpBound &upper_bound,
    const PnpBound &lower_bound, PnpFlag flags
  );

  void writeBack(
    camxiom::CameraModel &camera_model, std::vector<Eigen::Vector3d> &rvecs,
    std::vector<Eigen::Vector3d> &tvecs
  ) const;

  double optimize(
    const ObjectPointSets &object_points, const ImagePointSets &image_points,
    const camxiom::CameraModel &camera_model, PnpCostType cost_type
  );

  double optimizeGaussNewton(
    const ObjectPointSets &object_points, const ImagePointSets &image_points,
    const camxiom::CameraModel &camera_model, std::size_t &valid_count_out, bool &all_converged_out,
    int &total_iterations_out
  );

  void resetProblem();

  ceres::Problem::Options problem_options_{};
  std::unique_ptr<ceres::Problem> problem_{};
  ceres::Solver::Options solver_options_{};
  ceres::Solver::Summary solver_summary_{};
  double huber_loss_delta_{1.0};
  bool print_summary_{false};

  std::array<double, 2> focal_lengths_{0.0, 0.0};
  std::array<double, 2> principal_points_{0.0, 0.0};
  std::array<double, 14> dist_{};
  int dist_count_{0};
  std::array<double, 3> projection_params_{};  // [xi, alpha, beta]
  int projection_param_count_{0};
  PnpCostType cost_type_{PnpCostType::AUTO_DIFF};
  PnpFlag flags_{PnpFlag::NONE};
  std::vector<std::array<double, 3>> rvecs_{};
  std::vector<std::array<double, 3>> tvecs_{};

  PnpSummary last_summary_{};
};

PnpSolver::Impl::Impl()
{
  solver_options_.linear_solver_type = ceres::DENSE_SCHUR;
  solver_options_.max_num_iterations = 1000;
  solver_options_.minimizer_progress_to_stdout = false;
  dist_.fill(0.0);
  resetProblem();
}

void PnpSolver::Impl::resetProblem()
{
  problem_ = std::make_unique<ceres::Problem>(problem_options_);
}

bool PnpSolver::Impl::solve(
  const ObjectPointSets &object_points, const ImagePointSets &image_points,
  const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
  PnpFlag flags
)
{
  result_out = PnpResult{};
  // Reset the summary snapshot up front: the validation gauntlet below
  // returns early without running an optimization, and a reused solver must
  // not report the PREVIOUS solve's converged/solution_usable through
  // lastSummary() after such a failure.
  last_summary_ = PnpSummary{};

  // --- Input validation ---
  if (object_points.empty() || object_points.size() != image_points.size())
  {
    return false;
  }

  const std::size_t view_count = object_points.size();
  std::size_t total_points = 0;
  for (std::size_t i = 0; i < view_count; ++i)
  {
    if (object_points[i].size() != image_points[i].size() || object_points[i].empty())
    {
      return false;
    }
    // PnP requires at least 3 correspondences per view for 6-DOF
    if (object_points[i].size() < 3)
    {
      return false;
    }
    // NaN / Inf check
    for (std::size_t j = 0; j < object_points[i].size(); ++j)
    {
      if (!object_points[i][j].allFinite() || !image_points[i][j].allFinite())
      {
        return false;
      }
    }
    total_points += object_points[i].size();
  }

  if (initial_guess.rvecs.size() != view_count || initial_guess.tvecs.size() != view_count)
  {
    return false;
  }
  // Validate initial guesses are finite
  for (std::size_t i = 0; i < view_count; ++i)
  {
    if (!initial_guess.rvecs[i].allFinite() || !initial_guess.tvecs[i].allFinite())
    {
      return false;
    }
  }

  // Full camera model validation using camxiom's comprehensive validator
  {
    const camxiom::CameraModel64 model64_check =
      camxiom::toCameraModel64(initial_guess.camera_model);
    if (camxiom::validateCameraModel64(model64_check) != camxiom::StatusCode::OK)
    {
      return false;
    }
  }

  camxiom::CameraModel camera_model = initial_guess.camera_model;
  std::vector<Eigen::Vector3d> rvecs = initial_guess.rvecs;
  std::vector<Eigen::Vector3d> tvecs = initial_guess.tvecs;

  // Map the backend-agnostic PnpOptimizerOptions onto Ceres' option struct.
  // Ceres types never appear in the public API; the translation lives here.
  solver_options_ = ceres::Solver::Options();
  solver_options_.linear_solver_type = ceres::DENSE_SCHUR;
  solver_options_.max_num_iterations = options.solver_options.max_num_iterations;
  solver_options_.function_tolerance = options.solver_options.function_tolerance;
  solver_options_.parameter_tolerance = options.solver_options.parameter_tolerance;
  solver_options_.gradient_tolerance = options.solver_options.gradient_tolerance;
  solver_options_.minimizer_progress_to_stdout =
    options.solver_options.minimizer_progress_to_stdout;
  huber_loss_delta_ = options.huber_loss_delta;
  print_summary_ = options.print_summary;

  cost_type_ = options.cost_type;
  flags_ = flags;
  projection_param_count_ = detail::projectionParamCount(camera_model.projection.type);
  projection_params_[0] = static_cast<double>(camera_model.projection.xi);
  projection_params_[1] = static_cast<double>(camera_model.projection.alpha);
  projection_params_[2] = static_cast<double>(camera_model.projection.beta);

  result_out.total_count = total_points;

  // Check if we can use the fast Gauss-Newton path:
  // ANALYTICAL + all intrinsics/distortion/projection fixed + extrinsics NOT fixed
  const bool proj_fixed =
    hasFlag(flags, PnpFlag::FIX_PROJECTION_PARAMS) || projection_param_count_ == 0;
  const bool use_gn = (cost_type_ == PnpCostType::ANALYTICAL) &&
                      hasAllFlags(flags, PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION) &&
                      proj_fixed && !hasFlag(flags, PnpFlag::FIX_EXTRINSICS);

  double rmse = 0.0;
  if (use_gn)
  {
    // GN path: no Ceres setup needed, operates directly on rvecs_/tvecs_
    eigenToArray(rvecs, rvecs_);
    eigenToArray(tvecs, tvecs_);
    focal_lengths_[0] = static_cast<double>(camera_model.intrinsics.fx);
    focal_lengths_[1] = static_cast<double>(camera_model.intrinsics.fy);
    principal_points_[0] = static_cast<double>(camera_model.intrinsics.cx);
    principal_points_[1] = static_cast<double>(camera_model.intrinsics.cy);
    dist_count_ = effectiveDistCount(camera_model);
    dist_.fill(0.0);
    for (int i = 0; i < std::min(dist_count_, static_cast<int>(camera_model.distortion.count)); ++i)
      dist_[static_cast<std::size_t>(i)] =
        static_cast<double>(camera_model.distortion.coeffs[static_cast<std::size_t>(i)]);

    std::size_t gn_valid_count = 0;
    bool gn_all_converged = true;
    int gn_total_iterations = 0;
    rmse = optimizeGaussNewton(
      object_points, image_points, camera_model, gn_valid_count, gn_all_converged,
      gn_total_iterations
    );
    writeBack(camera_model, rvecs, tvecs);

    result_out.valid_count = gn_valid_count;

    // Build a summary for lastSummary() compatibility with actual iteration count
    solver_summary_ = ceres::Solver::Summary();
    solver_summary_.final_cost = rmse * rmse * 0.5;
    // GN success requires convergence, finite RMSE, and sufficient valid points
    const double valid_ratio =
      (total_points > 0) ? static_cast<double>(gn_valid_count) / static_cast<double>(total_points)
                         : 0.0;
    constexpr double kMinValidRatio = 0.5;
    const bool gn_success = gn_all_converged && std::isfinite(rmse) && gn_valid_count > 0 &&
                            valid_ratio >= kMinValidRatio;
    solver_summary_.termination_type = gn_success ? ceres::CONVERGENCE : ceres::FAILURE;
    for (int it = 0; it < gn_total_iterations; ++it)
    {
      ceres::IterationSummary iter_sum;
      iter_sum.cost = solver_summary_.final_cost;
      solver_summary_.iterations.push_back(iter_sum);
    }
    if (solver_summary_.iterations.empty())
    {
      ceres::IterationSummary iter_sum;
      iter_sum.cost = solver_summary_.final_cost;
      solver_summary_.iterations.push_back(iter_sum);
    }
  }
  else
  {
    setupProblem(camera_model, rvecs, tvecs, options.upper_bound, options.lower_bound, flags);
    rmse = optimize(object_points, image_points, camera_model, cost_type_);
    writeBack(camera_model, rvecs, tvecs);
  }

  // Post-optimization model validation: the optimizer may converge to
  // parameters that are numerically valid in double but become invalid
  // after the double→float writeBack (e.g. DS xi drifting to a small
  // negative value).  Reject such solutions early. A polynomial-fisheye fit
  // whose theta_d folds below the seed's cap does NOT need rescuing here:
  // writeBack() already re-derived theta_max from the fitted coefficients
  // via updateThetaMax(), so the endpoint consistency check passes by
  // construction (the mid-solve equivalent lives in the analytical batch
  // cost, where theta_max IS still frozen at the seed value).
  const bool model_valid = camxiom::validateCameraModel(camera_model) == camxiom::StatusCode::OK;

  if (!use_gn)
  {
    // Count how many correspondences the *recovered* model and poses can
    // actually project, mirroring the GN path. The old fixed
    // valid_count = total_points made the field useless as a validity
    // indicator whenever unprojectable (penalised) points survived the
    // solve.
    std::size_t ceres_valid = 0;
    if (model_valid)
    {
      const camxiom::CameraModel64 model64 = camxiom::toCameraModel64(camera_model);
      for (std::size_t i = 0; i < object_points.size(); ++i)
      {
        Eigen::Matrix3d R;
        {
          double aa[3] = {rvecs[i].x(), rvecs[i].y(), rvecs[i].z()};
          double rm[9];
          ceres::AngleAxisToRotationMatrix(aa, rm);
          R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rm);
        }
        for (const auto &p_obj : object_points[i])
        {
          const Eigen::Vector3d p_cam = R * p_obj + tvecs[i];
          if (camxiom::rayToPixel64(model64, p_cam).status == camxiom::StatusCode::OK)
          {
            ++ceres_valid;
          }
        }
      }
    }
    result_out.valid_count = ceres_valid;
  }
  if (!model_valid && print_summary_)
  {
    // Diagnostic only, gated behind print_summary: a library must not write
    // to stderr unconditionally on real-time consumers. The failure itself
    // is reported through result_out.success below.
    std::cerr << "[PnpSolver] Post-optimization camera model is invalid "
                 "(projection params may have left valid range)."
              << std::endl;
  }

  const bool success =
    model_valid && (use_gn ? (solver_summary_.termination_type == ceres::CONVERGENCE)
                           : (solver_summary_.IsSolutionUsable() && std::isfinite(rmse)));

  // Mirror the Ceres summary into the backend-agnostic snapshot exposed by
  // lastSummary(). The GN path leaves num_*_steps at 0 (it reports progress
  // via the manually built iteration list), matching prior behavior.
  last_summary_.num_successful_steps = solver_summary_.num_successful_steps;
  last_summary_.num_unsuccessful_steps = solver_summary_.num_unsuccessful_steps;
  last_summary_.final_cost = solver_summary_.final_cost;
  last_summary_.solution_usable = solver_summary_.IsSolutionUsable();
  // Both paths set termination_type: Ceres natively, the GN fast path
  // synthesizes CONVERGENCE/FAILURE from its own convergence flag above.
  last_summary_.converged = solver_summary_.termination_type == ceres::CONVERGENCE;

  result_out.success = success;
  result_out.rmse = rmse;
  result_out.camera_model = camera_model;
  result_out.rvecs = std::move(rvecs);
  result_out.tvecs = std::move(tvecs);
  return result_out.success;
}

bool PnpSolver::Impl::solve(
  const ObjectPoints &object_points, const ImagePoints &image_points,
  const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
  PnpFlag flags
)
{
  ObjectPointSets obj_sets{object_points};
  ImagePointSets img_sets{image_points};
  return solve(obj_sets, img_sets, initial_guess, result_out, options, flags);
}

void PnpSolver::Impl::setupProblem(
  const camxiom::CameraModel &camera_model, const std::vector<Eigen::Vector3d> &rvecs,
  const std::vector<Eigen::Vector3d> &tvecs, const PnpBound &upper_bound,
  const PnpBound &lower_bound, PnpFlag flags
)
{
  resetProblem();

  focal_lengths_[0] = static_cast<double>(camera_model.intrinsics.fx);
  focal_lengths_[1] = static_cast<double>(camera_model.intrinsics.fy);
  principal_points_[0] = static_cast<double>(camera_model.intrinsics.cx);
  principal_points_[1] = static_cast<double>(camera_model.intrinsics.cy);

  dist_count_ = effectiveDistCount(camera_model);
  dist_.fill(0.0);
  for (int i = 0; i < std::min(dist_count_, static_cast<int>(camera_model.distortion.count)); ++i)
  {
    dist_[static_cast<std::size_t>(i)] =
      static_cast<double>(camera_model.distortion.coeffs[static_cast<std::size_t>(i)]);
  }

  eigenToArray(rvecs, rvecs_);
  eigenToArray(tvecs, tvecs_);

  // Single source of truth for the intrinsic / projection box bounds (C5 ⑤):
  // the same descriptor the near-bound diagnostic consumes, so solver and
  // diagnostics can never drift. rot / trans bounds stay inline below.
  const detail::CalibrationParameterBounds param_bounds =
    detail::computeCalibrationParameterBounds(camera_model, flags, lower_bound, upper_bound);

  problem_->AddParameterBlock(focal_lengths_.data(), 2);
  if (hasFlag(flags, PnpFlag::FIX_FOCAL_LENGTHS))
  {
    problem_->SetParameterBlockConstant(focal_lengths_.data());
  }
  else
  {
    applyScalarBound(*problem_, focal_lengths_.data(), 0, param_bounds.focal_lengths[0]);
    applyScalarBound(*problem_, focal_lengths_.data(), 1, param_bounds.focal_lengths[1]);
  }

  problem_->AddParameterBlock(principal_points_.data(), 2);
  if (hasFlag(flags, PnpFlag::FIX_PRINCIPAL_POINTS))
  {
    problem_->SetParameterBlockConstant(principal_points_.data());
  }
  else
  {
    applyScalarBound(*problem_, principal_points_.data(), 0, param_bounds.principal_points[0]);
    applyScalarBound(*problem_, principal_points_.data(), 1, param_bounds.principal_points[1]);
  }

  const int effective_dc = std::max(1, dist_count_);
  problem_->AddParameterBlock(dist_.data(), effective_dc);
  std::vector<int> dist_fixed;
  for (int i = 0; i < effective_dc; ++i)
  {
    const auto flag_bit = static_cast<PnpFlag>(1ull << (8 + i));
    if (hasFlag(flags, flag_bit))
    {
      dist_fixed.push_back(i);
    }
  }
  makeSubsetManifold(*problem_, dist_.data(), effective_dc, dist_fixed);

  // Projection params block: [xi, alpha, beta] — always size 3
  problem_->AddParameterBlock(projection_params_.data(), 3);
  if (hasFlag(flags, PnpFlag::FIX_PROJECTION_PARAMS) || projection_param_count_ == 0)
  {
    problem_->SetParameterBlockConstant(projection_params_.data());
  }
  else
  {
    // Fix unused slots based on model type
    makeSubsetManifold(
      *problem_, projection_params_.data(), 3,
      fixedProjectionParamIndices(camera_model.projection.type)
    );

    // Projection parameter bounds (model-validity hard bounds).
    // The denom < eps penalty in the cost functor does NOT directly enforce
    // valid xi / alpha ranges — for frontal points, xi*d1+z can stay
    // positive across a wide range of xi, so the optimizer can converge to
    // unstable parameters without triggering any penalty. After writeBack()
    // the double→float conversion preserves the sign, and
    // validateCameraModel() rejects out-of-range xi / alpha, so explicit
    // bounds are necessary for correctness. The exact ranges (Double Sphere
    // xi in (-1, 1) with an interior margin, alpha in [0, 1]; EUCM alpha in
    // [0, 1], beta > 0) live in the shared descriptor above.
    applyScalarBound(*problem_, projection_params_.data(), 0, param_bounds.projection[0]);
    applyScalarBound(*problem_, projection_params_.data(), 1, param_bounds.projection[1]);
    applyScalarBound(*problem_, projection_params_.data(), 2, param_bounds.projection[2]);
  }

  const bool fix_ext = hasFlag(flags, PnpFlag::FIX_EXTRINSICS);
  for (std::size_t i = 0; i < rvecs_.size(); ++i)
  {
    problem_->AddParameterBlock(rvecs_[i].data(), 3);
    problem_->AddParameterBlock(tvecs_[i].data(), 3);
    if (fix_ext)
    {
      problem_->SetParameterBlockConstant(rvecs_[i].data());
      problem_->SetParameterBlockConstant(tvecs_[i].data());
    }
    else
    {
      problem_->SetParameterLowerBound(rvecs_[i].data(), 0, lower_bound.rot);
      problem_->SetParameterLowerBound(rvecs_[i].data(), 1, lower_bound.rot);
      problem_->SetParameterLowerBound(rvecs_[i].data(), 2, lower_bound.rot);
      problem_->SetParameterUpperBound(rvecs_[i].data(), 0, upper_bound.rot);
      problem_->SetParameterUpperBound(rvecs_[i].data(), 1, upper_bound.rot);
      problem_->SetParameterUpperBound(rvecs_[i].data(), 2, upper_bound.rot);
      problem_->SetParameterLowerBound(tvecs_[i].data(), 0, lower_bound.trans);
      problem_->SetParameterLowerBound(tvecs_[i].data(), 1, lower_bound.trans);
      problem_->SetParameterLowerBound(tvecs_[i].data(), 2, lower_bound.trans);
      problem_->SetParameterUpperBound(tvecs_[i].data(), 0, upper_bound.trans);
      problem_->SetParameterUpperBound(tvecs_[i].data(), 1, upper_bound.trans);
      problem_->SetParameterUpperBound(tvecs_[i].data(), 2, upper_bound.trans);
    }
  }
}

void PnpSolver::Impl::writeBack(
  camxiom::CameraModel &camera_model, std::vector<Eigen::Vector3d> &rvecs,
  std::vector<Eigen::Vector3d> &tvecs
) const
{
  camera_model.intrinsics.fx = static_cast<float>(focal_lengths_[0]);
  camera_model.intrinsics.fy = static_cast<float>(focal_lengths_[1]);
  camera_model.intrinsics.cx = static_cast<float>(principal_points_[0]);
  camera_model.intrinsics.cy = static_cast<float>(principal_points_[1]);

  for (int i = 0; i < std::min(dist_count_, static_cast<int>(camera_model.distortion.count)); ++i)
  {
    camera_model.distortion.coeffs[static_cast<std::size_t>(i)] =
      static_cast<float>(dist_[static_cast<std::size_t>(i)]);
  }

  camera_model.projection.xi = static_cast<float>(projection_params_[0]);
  camera_model.projection.alpha = static_cast<float>(projection_params_[1]);
  camera_model.projection.beta = static_cast<float>(projection_params_[2]);

  // Rebuild distortion auxiliary state from updated coeffs so the returned
  // CameraModel is self-consistent (is_rational, has_thin_prism, has_tilt,
  // tilt_matrix, inv_tilt_matrix).
  camxiom::detail::rebuildDistortionAuxState(camera_model.distortion);

  // Recalculate theta_max from the updated distortion coefficients.
  // For polynomial fisheye models (KB4/KB8/OpenCV_Fisheye4) the safe monotonic
  // range of the distortion polynomial depends on the coefficients, so
  // theta_max must be refreshed whenever they change.
  camxiom::updateThetaMax(camera_model);

  arrayToEigen(rvecs_, rvecs);
  arrayToEigen(tvecs_, tvecs);
}

double PnpSolver::Impl::optimize(
  const ObjectPointSets &object_points, const ImagePointSets &image_points,
  const camxiom::CameraModel &camera_model, PnpCostType cost_type
)
{
  solver_summary_ = ceres::Solver::Summary();
  std::size_t total_points = 0;
  const int effective_dc = std::max(1, dist_count_);

  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    const auto &obj = object_points[i];
    const auto &img = image_points[i];
    const std::size_t pts = obj.size();
    total_points += pts;

    if (cost_type == PnpCostType::ANALYTICAL)
    {
      if (huber_loss_delta_ > 0.0)
      {
        // Ceres applies a LossFunction to the squared norm of an entire
        // residual block. On the batched view-level block (2*N residuals)
        // Huber would down-weight whole views by ~delta/||r|| instead of
        // suppressing individual outlier points — for a single view it
        // degenerates to plain L2. Robust solves therefore get one
        // analytical block per point, like the AUTO_DIFF path.
        for (std::size_t j = 0; j < pts; ++j)
        {
          ceres::CostFunction *cost = createAnalyticalBatchCost(
            ObjectPoints{obj[j]}, ImagePoints{img[j]}, camera_model, effective_dc
          );
          problem_->AddResidualBlock(
            cost, new ceres::HuberLoss(huber_loss_delta_), rvecs_[i].data(), tvecs_[i].data(),
            focal_lengths_.data(), principal_points_.data(), dist_.data(), projection_params_.data()
          );
        }
      }
      else
      {
        // Batch: one CostFunction per view with 2*N residuals
        ceres::CostFunction *cost = createAnalyticalBatchCost(obj, img, camera_model, effective_dc);
        problem_->AddResidualBlock(
          cost, nullptr, rvecs_[i].data(), tvecs_[i].data(), focal_lengths_.data(),
          principal_points_.data(), dist_.data(), projection_params_.data()
        );
      }
    }
    else
    {
      // Per-point: one CostFunction per point with 2 residuals
      for (std::size_t j = 0; j < pts; ++j)
      {
        ceres::CostFunction *cost = createCost(obj[j], img[j], camera_model, effective_dc);
        ceres::LossFunction *loss = nullptr;
        if (huber_loss_delta_ > 0.0)
        {
          loss = new ceres::HuberLoss(huber_loss_delta_);
        }
        problem_->AddResidualBlock(
          cost, loss, rvecs_[i].data(), tvecs_[i].data(), focal_lengths_.data(),
          principal_points_.data(), dist_.data(), projection_params_.data()
        );
      }
    }
  }

  if (total_points == 0)
  {
    return std::numeric_limits<double>::infinity();
  }

  ceres::Solver::Options opt = solver_options_;
  if (opt.max_num_iterations <= 0)
  {
    opt.max_num_iterations = 100;
  }

  ceres::Solve(opt, problem_.get(), &solver_summary_);
  if (print_summary_)
  {
    std::cout << solver_summary_.FullReport() << std::endl;
    std::cout << "Total cost: " << solver_summary_.final_cost << std::endl;
  }

  ceres::Problem::EvaluateOptions eval;
  eval.apply_loss_function = false;
  double half_l2_cost = 0.0;
  problem_->Evaluate(eval, &half_l2_cost, nullptr, nullptr, nullptr);
  const double rmse_2d = std::sqrt((2.0 * half_l2_cost) / static_cast<double>(total_points));
  return rmse_2d;
}

double PnpSolver::Impl::optimizeGaussNewton(
  const ObjectPointSets &object_points, const ImagePointSets &image_points,
  const camxiom::CameraModel &camera_model, std::size_t &valid_count_out, bool &all_converged_out,
  int &total_iterations_out
)
{
  const camxiom::CameraModel64 model64 = camxiom::toCameraModel64(camera_model);

  GaussNewtonOptions gn_opts;
  gn_opts.max_iterations = std::max(1, solver_options_.max_num_iterations);
  gn_opts.function_tolerance = solver_options_.function_tolerance;
  gn_opts.parameter_tolerance = solver_options_.parameter_tolerance;
  gn_opts.gradient_tolerance = solver_options_.gradient_tolerance;
  gn_opts.huber_delta = huber_loss_delta_;

  std::size_t total_valid = 0;
  double total_cost = 0.0;
  bool all_converged = true;
  int total_iterations = 0;

  for (std::size_t i = 0; i < object_points.size(); ++i)
  {
    const auto &obj = object_points[i];
    const auto &img = image_points[i];

    Eigen::Vector3d rv(rvecs_[i][0], rvecs_[i][1], rvecs_[i][2]);
    Eigen::Vector3d tv(tvecs_[i][0], tvecs_[i][1], tvecs_[i][2]);

    const auto gn_result = solveViewGaussNewton(model64, obj, img, rv, tv, gn_opts);
    if (!gn_result.converged)
    {
      all_converged = false;
    }
    total_iterations += gn_result.iterations;

    rvecs_[i][0] = rv.x();
    rvecs_[i][1] = rv.y();
    rvecs_[i][2] = rv.z();
    tvecs_[i][0] = tv.x();
    tvecs_[i][1] = tv.y();
    tvecs_[i][2] = tv.z();

    // Compute final residuals for RMSE (only valid projections count)
    Eigen::Matrix3d R;
    {
      double aa[3] = {rv.x(), rv.y(), rv.z()};
      double rm[9];
      ceres::AngleAxisToRotationMatrix(aa, rm);
      R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rm);
    }
    for (std::size_t j = 0; j < obj.size(); ++j)
    {
      const Eigen::Vector3d p_cam = R * obj[j] + tv;
      const auto fj = camxiom::rayToPixelWithFullJacobian64(model64, p_cam);
      if (fj.status == camxiom::StatusCode::OK)
      {
        const double dx = img[j].x() - fj.pixel.u;
        const double dy = img[j].y() - fj.pixel.v;
        total_cost += dx * dx + dy * dy;
        ++total_valid;
      }
    }
  }

  valid_count_out = total_valid;
  all_converged_out = all_converged;
  total_iterations_out = total_iterations;

  if (total_valid == 0)
  {
    return std::numeric_limits<double>::infinity();
  }

  return std::sqrt(total_cost / static_cast<double>(total_valid));
}

// ========================= PnpSolver (PIMPL facade) =========================

PnpSolver::PnpSolver() : impl_(std::make_unique<Impl>()) {}
PnpSolver::~PnpSolver() = default;
PnpSolver::PnpSolver(PnpSolver &&) noexcept = default;
PnpSolver &PnpSolver::operator=(PnpSolver &&) noexcept = default;

bool PnpSolver::solve(
  const ObjectPointSets &object_points, const ImagePointSets &image_points,
  const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
  PnpFlag flags
)
{
  return impl_->solve(object_points, image_points, initial_guess, result_out, options, flags);
}

bool PnpSolver::solve(
  const ObjectPoints &object_points, const ImagePoints &image_points,
  const PnpInitialGuess &initial_guess, PnpResult &result_out, const PnpSolverOptions &options,
  PnpFlag flags
)
{
  return impl_->solve(object_points, image_points, initial_guess, result_out, options, flags);
}

const PnpSummary &PnpSolver::lastSummary() const { return impl_->lastSummary(); }

}  // namespace camxiom::optimizer
