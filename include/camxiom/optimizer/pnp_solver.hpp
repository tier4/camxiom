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

#ifndef CAMXIOM__OPTIMIZER__PNP_SOLVER_HPP
#define CAMXIOM__OPTIMIZER__PNP_SOLVER_HPP

#include "camxiom/internal/constants.hpp"
#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace camxiom::optimizer
{

using ObjectPoints = std::vector<Eigen::Vector3d>;
using ImagePoints = std::vector<Eigen::Vector2d>;
using ObjectPointSets = std::vector<ObjectPoints>;
using ImagePointSets = std::vector<ImagePoints>;

/// NOTE: the box bounds are enforced only on the Ceres-based paths. The
/// pose-only Gauss-Newton fast path (taken when intrinsics, distortion and
/// projection parameters are all fixed) runs unconstrained and ignores
/// these fields.
struct PnpBound
{
  Eigen::Vector2d focal_lengths{5000.0, 5000.0};
  Eigen::Vector2d principal_points{5000.0, 5000.0};
  // Per-component box bound on the Rodrigues (angle-axis) vector. Must admit
  // the full canonical range [-pi, pi]: a truncated value (e.g. 3.1415) would
  // clamp legitimate ~180-degree rotations onto the bound.
  double rot{camxiom::constants::kPi};
  double trans{std::numeric_limits<double>::infinity()};

  static PnpBound createUpperBound()
  {
    return PnpBound{
      Eigen::Vector2d(5000.0, 5000.0), Eigen::Vector2d(5000.0, 5000.0), camxiom::constants::kPi,
      std::numeric_limits<double>::infinity()};
  }

  static PnpBound createLowerBound()
  {
    return PnpBound{
      Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, 1.0), -camxiom::constants::kPi,
      -std::numeric_limits<double>::infinity()};
  }
};

enum class PnpCostType : std::uint8_t { AUTO_DIFF = 0, ANALYTICAL = 1 };

/// Backend-agnostic optimizer controls.
///
/// Mirrors the small subset of non-linear least-squares knobs that the PnP
/// solver actually consumes, expressed in plain C++ so the public API does
/// not leak the underlying Ceres types. Defaults match the historical
/// `ceres::Solver::Options` defaults used by camxiom.
struct PnpOptimizerOptions
{
  int max_num_iterations{1000};
  double function_tolerance{1e-6};
  double parameter_tolerance{1e-8};
  double gradient_tolerance{1e-10};
  bool minimizer_progress_to_stdout{false};
};

struct PnpSolverOptions
{
  PnpOptimizerOptions solver_options{};
  double huber_loss_delta{1.0};
  bool print_summary{false};
  PnpBound upper_bound{PnpBound::createUpperBound()};
  PnpBound lower_bound{PnpBound::createLowerBound()};
  PnpCostType cost_type{PnpCostType::ANALYTICAL};
};

struct PnpInitialGuess
{
  camxiom::CameraModel camera_model{};
  std::vector<Eigen::Vector3d> rvecs{};
  std::vector<Eigen::Vector3d> tvecs{};
};

struct [[nodiscard]] PnpResult
{
  bool success{false};
  double rmse{std::numeric_limits<double>::infinity()};
  camxiom::CameraModel camera_model{};
  std::vector<Eigen::Vector3d> rvecs{};
  std::vector<Eigen::Vector3d> tvecs{};
  std::size_t valid_count{0};
  std::size_t total_count{0};

  constexpr bool ok() const { return success; }
  explicit operator bool() const { return ok(); }
};

/// Backend-agnostic snapshot of the last optimization.
///
/// Plain replacement for `ceres::Solver::Summary`: exposes only the fields
/// consumers need (iteration accounting, final cost, usability) without
/// pulling Ceres into the public header.
struct PnpSummary
{
  int num_successful_steps{0};
  int num_unsuccessful_steps{0};
  double final_cost{0.0};
  bool solution_usable{false};
  /// True only when the optimizer met its tolerances (function / parameter /
  /// gradient). A run that stopped at max_num_iterations can still be
  /// solution_usable without being converged — callers that promise
  /// "converged" semantics must check this, not solution_usable.
  bool converged{false};
};

/// PnP solver that works with ANY camera model supported by camxiom.
///
/// Uses projection_template.hpp for the cost functor, so all projection and
/// distortion models are handled by a single code path. OpenCV-free: the
/// public API depends only on Eigen. Ceres is an implementation detail hidden
/// behind a PIMPL, so no Ceres type appears in this header.
///
/// Thread safety: solve() MUTATES internal state (the summary that
/// lastSummary() reads), so one PnpSolver instance must not run concurrent
/// solve() calls — use one instance per thread, or synchronize externally.
/// lastSummary() refers to this instance's most recent solve(). Instances
/// share no global state, so per-thread solvers scale without contention.
/// (Details: docs/design/thread-safety.md.)
class PnpSolver
{
public:
  PnpSolver();
  ~PnpSolver();
  PnpSolver(PnpSolver &&) noexcept;
  PnpSolver &operator=(PnpSolver &&) noexcept;
  PnpSolver(const PnpSolver &) = delete;
  PnpSolver &operator=(const PnpSolver &) = delete;

  /// Solve with provided initial guess.
  bool solve(
    const ObjectPointSets &object_points, const ImagePointSets &image_points,
    const PnpInitialGuess &initial_guess, PnpResult &result_out,
    const PnpSolverOptions &options = PnpSolverOptions(), PnpFlag flags = PnpFlag::NONE
  );

  /// Solve single-view convenience overload.
  bool solve(
    const ObjectPoints &object_points, const ImagePoints &image_points,
    const PnpInitialGuess &initial_guess, PnpResult &result_out,
    const PnpSolverOptions &options = PnpSolverOptions(), PnpFlag flags = PnpFlag::NONE
  );

  PnpResult solve(
    const ObjectPointSets &object_points, const ImagePointSets &image_points,
    const PnpInitialGuess &initial_guess, const PnpSolverOptions &options = PnpSolverOptions(),
    PnpFlag flags = PnpFlag::NONE
  )
  {
    PnpResult result;
    solve(object_points, image_points, initial_guess, result, options, flags);
    return result;
  }

  /// Snapshot of the most recent solve() call on THIS instance. Reset to a
  /// default PnpSummary at the start of every solve(), so after a call that
  /// failed input validation (returned false before optimizing) every field
  /// reads as its zero/false default — never the previous solve's values.
  const PnpSummary &lastSummary() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace camxiom::optimizer

#endif  // CAMXIOM__OPTIMIZER__PNP_SOLVER_HPP
