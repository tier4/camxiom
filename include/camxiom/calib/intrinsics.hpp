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

#ifndef CAMXIOM__CALIB__INTRINSICS_HPP
#define CAMXIOM__CALIB__INTRINSICS_HPP

#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace camxiom::calib
{

/// One calibration view: a set of board-frame 3D points and the matching
/// pixel observations. world_points[j] corresponds to image_points[j].
/// Planar boards have world_points[j].z() == 0; estimatePoseDLT handles
/// that case internally so the caller does not branch on it.
struct CalibrationView
{
  std::vector<Eigen::Vector3d> world_points;
  std::vector<Eigen::Vector2d> image_points;
};

/// Knobs forwarded to the single Ceres pass. This struct intentionally
/// carries NO sampling / staging / outlier / restart strategy: those are
/// the application layer's responsibility. camxiom only does what it
/// is told here.
struct CalibrationOptions
{
  int image_width{0};
  int image_height{0};
  int max_iterations{200};
  double function_tolerance{1e-10};
  double parameter_tolerance{1e-12};
  /// Robust loss threshold for reprojection residuals. Zero selects ordinary
  /// least squares, matching cv::calibrateCamera.
  double huber_loss_delta{0.0};
  /// When true, focal-length parameters are box-constrained to initial_model
  /// +/- bound_relative_tolerance (guards against runaway focal updates on
  /// noisy data). Legacy callers that do not supply the explicit
  /// principal-point bound below retain the old combined behaviour.
  bool apply_initial_value_bounds{false};
  double bound_relative_tolerance{0.10};  // +/-10%
  /// Independently constrain the principal point around a caller-supplied
  /// reference. This is separate from apply_initial_value_bounds so an
  /// application can keep cx/cy physically plausible without also restricting
  /// focal-length recovery.
  bool apply_principal_point_bounds{false};
  double principal_point_reference_x{0.0};
  double principal_point_reference_y{0.0};
  double principal_point_bound_relative_tolerance{0.10};
  /// When true (default), calibrate() also fills the parameter uncertainty
  /// and the observability diagnostic on CalibrationResult. That costs one
  /// extra full-Jacobian pass at the solution plus small (p x p, p<=~20) linear
  /// algebra — negligible next to the Ceres solve, but a caller running a
  /// high-frequency online partial solve that only needs the model + RMS can
  /// set this false to skip it (the uncertainty_* / observability_* /
  /// near-bound fields then keep their fail-closed defaults:
  /// uncertainty_available=false, observability_available=false,
  /// observability_ok=false, uncertainty_has_parameters_near_bounds=false).
  bool estimate_uncertainty{true};
  /// When true, calibrate() also fills CalibrationResult::per_point_residuals:
  /// the signed (projected - observed) residual of every correspondence,
  /// index-aligned with the input views (unprojectable points carry NaN).
  /// This is the same residual definition behind rms_reprojection_error_px,
  /// so an application-level outlier strategy (sampling / rejection stays in
  /// the app) can classify points without re-deriving the projection. Off by
  /// default; the extra pass costs one projection per correspondence.
  bool compute_per_point_residuals{false};
};

/// Output of calibrate(). On any non-OK status from the early validation /
/// per-view DLT-PnP stages, every field except `status` is left at its
/// default (atomic failure: no partially filled result). On OK or
/// NON_CONVERGED the model, poses and diagnostics are all populated;
/// NON_CONVERGED still carries a best-effort result the app may use.
struct CalibrationResult
{
  StatusCode status{StatusCode::OK};
  CameraModel camera_model{};
  std::vector<Eigen::Matrix3d> per_view_rotations;
  std::vector<Eigen::Vector3d> per_view_translations;
  double rms_reprojection_error_px{0.0};
  double max_reprojection_error_px{0.0};
  std::vector<double> per_view_rms_px;
  /// Per-view, per-point signed reprojection residuals (projected minus
  /// observed, pixels). Filled only when
  /// CalibrationOptions::compute_per_point_residuals is set; empty otherwise.
  /// Outer index = view, inner index = correspondence, aligned with the
  /// input views; unprojectable points are (NaN, NaN) so indices never shift.
  std::vector<std::vector<Eigen::Vector2d>> per_point_residuals;
  int iterations_used{0};

  // ---- Parameter uncertainty ----------------------------------------------
  // A local Gauss-Newton / OLS covariance ESTIMATE at the final solution: the
  // 1-sigma standard deviation of each FREE (optimised) intrinsic / distortion
  // / projection parameter, recovered from the reduced normal equations JᵀJ at
  // the solution (per-view extrinsics marginalised out via the Schur
  // complement), scaled by the residual variance sigma^2 = RSS/dof. It assumes
  // independent, homoscedastic pixel residuals. For a linear model it is exact;
  // for this non-linear problem it is the asymptotically-valid Laplace / GN
  // approximation about the final estimate.
  //
  // Robust-loss weighting is NOT incorporated: the reduced normal equations are
  // built from the raw (unweighted) residual Jacobian, so when calibrate() is
  // called with huber_loss_delta > 0 these numbers describe the OLS curvature,
  // not the robustified objective the solver actually minimised, and the two
  // can disagree.
  //
  // `parameter_std[i]` is the std-dev of the parameter named
  // `uncertainty_labels[i]` (parallel arrays; only free parameters appear —
  // parameters held fixed by `lock_flags` are NOT listed). Labels are
  // "fx" / "fy" / "cx" / "cy" (pixels), "dist[k]", and "xi" / "alpha" / "beta".
  //
  // `uncertainty_available` is false (and both vectors empty) when the
  // uncertainty could not be formed: an early / non-OK return, no free
  // parameters, too few valid observations (dof <= 0), a per-view pose block
  // that could not be marginalised, or a rank-deficient reduced normal matrix
  // (which is itself the degeneracy signal surfaced by the observability
  // diagnostic below). Only read parameter_std / uncertainty_labels when
  // uncertainty_available is true.
  //
  // CAVEATS (do not over-trust the numbers):
  //   * fx/fy/cx/cy std are in pixels and directly interpretable; distortion
  //     coefficient std are in the coefficients' own (dimensionless, wildly
  //     differing) scales and are only meaningful relative to each other.
  //   * When a parameter sits on an active box bound (focal / principal-point /
  //     xi / alpha bounds in the solver), the unconstrained-Hessian covariance
  //     is optimistic for that parameter — cross-check
  //     parameters_at_or_near_bounds below.
  //   * Populated for the standard calibrate() Ceres path (intrinsics free).
  //     A pose-only solve (FIX_INTRINSICS) leaves this empty by design.
  bool uncertainty_available{false};
  std::vector<std::string> uncertainty_labels;
  std::vector<double> parameter_std;

  // ---- Observability / degeneracy diagnostic ------------------------------
  // Numerical observability of the FREE parameters, read off the SAME reduced
  // normal matrix S used for the parameter uncertainty above, after Jacobi
  // (diagonal) scaling to unit
  // diagonal — so the condition number is scale-invariant and comparable
  // across the wildly different units of focal / principal-point / distortion
  // coefficients (a raw JᵀJ condition number is meaningless here). Let s_i be
  // the singular values of the column-scaled Jacobian (s_i = sqrt(eigenvalue_i
  // of the scaled S):
  //   * min_singular_value          = min_i s_i  (0 == a fully unobservable
  //                                    parameter direction)
  //   * normalized_condition_number = max_i s_i / min_i s_i  (large == an
  //                                    ill-determined direction). When S is
  //                                    rank-deficient (min_singular_value == 0)
  //                                    this is +infinity, NOT 0 or NaN.
  //
  // Like the parameter uncertainty, this is a raw-residual (unweighted)
  // diagnostic: robust-loss
  // weighting is NOT incorporated, so under huber_loss_delta > 0 these metrics
  // describe the geometric observability of the OLS Jacobian rather than the
  // solver's effective robustified curvature.
  //
  // STATE CONTRACT (observability_available x observability_ok):
  //   * available == false, ok == false: the diagnostic value could NOT be
  //     produced — diagnostics skipped (estimate_uncertainty == false), an
  //     early / non-OK return, zero free parameters, the reduced normal matrix
  //     S could not be built (a per-view pose block failed to factor, or a
  //     non-finite entry), or the eigenvalue solver failed / returned a clearly
  //     negative eigenvalue. The numbers below are meaningless (0).
  //   * available == true, ok == false: S was built and analysed and is
  //     degenerate (rank-deficient / weak subspace present) — a SUCCESSFUL
  //     diagnosis of a degeneracy. This is the classic aliasing trap of the
  //     fisheye / omni model families: a lock_flags choice left a
  //     non-identifiable parameter free even though the RMS looks fine.
  //     `underdetermined_parameters` names the free parameters that
  //     participate in the weak subspace.
  //   * available == true, ok == true: analysed and adequately observable.
  //
  // Only read normalized_condition_number / min_singular_value /
  // underdetermined_parameters when observability_available is true; a 0 while
  // available == false means "not computed", not a healthy value.
  //
  // The observability diagnostic is independent of the parameter uncertainty:
  // it needs only S (not the residual variance), so a dof <= 0 result can
  // leave the uncertainty unavailable while observability still reports on S.
  //
  // This is a DIAGNOSTIC only: camxiom reports the numbers and never
  // changes lock_flags or drops views, and never silently regularises a
  // degenerate S.
  bool observability_available{false};
  bool observability_ok{false};
  double normalized_condition_number{0.0};
  double min_singular_value{0.0};
  std::vector<std::string> underdetermined_parameters;

  // ---- Box-constraint proximity --------------------------------------------
  // Whether any FREE, bounded parameter's final estimate sits at or near one of
  // the solver's box bounds (focal / principal-point / Double-Sphere xi & alpha
  // / EUCM alpha & beta). This is NOT the solver's KKT active set — Ceres does
  // not expose it — but a proximity test |x - bound| <= tol on the FINAL
  // estimate, with a per-side absolute/relative tolerance so a wide artificial
  // upper bound never pollutes the opposite (lower) side. One-sided bounds
  // (e.g. EUCM beta, lower only) are tested only on the side that exists.
  //
  // Fixed parameters (absent from the reduced normal) and unbounded parameters
  // (all distortion coefficients, OMNIDIRECTIONAL xi) are never listed.
  // `parameters_at_or_near_bounds` names the offenders;
  // `uncertainty_has_parameters_near_bounds` is true iff that list is non-empty.
  //
  // Computed independently of S: it is populated for any OK / NON_CONVERGED
  // result under estimate_uncertainty == true, EVEN WHEN the covariance above
  // was not available (e.g. a degenerate solve that parked alpha on its upper
  // bound). When a listed parameter also received a parameter_std entry, treat
  // that std as optimistic (the unconstrained-Hessian covariance ignores the
  // active bound).
  bool uncertainty_has_parameters_near_bounds{false};
  std::vector<std::string> parameters_at_or_near_bounds;

  /// Strict success: status == OK. Note that NON_CONVERGED is NOT ok() but
  /// still carries a best-effort model/poses (see the struct doc above);
  /// callers that want to consume best-effort results must test status
  /// explicitly.
  constexpr bool ok() const { return status == StatusCode::OK; }
  explicit operator bool() const { return ok(); }
};

/// Re-export so the application can compose per-stage lock flags without
/// reaching into the optimizer namespace directly.
using optimizer::PnpFlag;

/// Single-pass intrinsics calibration over ALL provided corners.
///
/// This is a strategy-free single function: exactly one Ceres pass,
/// no staging, no multi-restart, no outlier rejection, no model
/// dispatch. PnpSolver is model-agnostic via the CameraModel, so the
/// projection/distortion model is whatever `initial_model` specifies.
///
/// Pipeline:
///   1. Validate inputs (non-empty views, positive image size, per-view
///      size match, N >= 6 per view, valid initial_model).
///   2. Estimate every view's pose with the model-agnostic DLT-PnP
///      (init::estimatePoseDLT) using initial_model. A DLT-PnP failure on
///      any view aborts the whole call (skipping views is application
///      strategy, which deliberately stays out of camxiom).
///   3. Build PnpSolver inputs (object/image point sets, initial guess,
///      solver options + bounds derived from CalibrationOptions).
///   4. Run PnpSolver::solve ONCE.
///   5. Assemble the result and compute reprojection diagnostics.
///
/// `lock_flags` is caller-supplied and decides which intrinsic /
/// distortion / projection parameters are held fixed in the Ceres pass.
/// The caller MUST pass per-model identifiability locks: KB4
/// FIX_DIST_2|FIX_DIST_3, MEI/DS/EUCM FIX_PROJECTION_PARAMS, pinhole NONE.
/// Leaving non-identifiable distortion/projection parameters free will
/// drive the solve into the aliasing failure described in the
/// CalibrationResult observability notes (huge parameter values while the
/// reprojection RMS still looks fine).
///
/// @param views          One CalibrationView per board observation.
/// @param initial_model   Fully specified seed (e.g. from getDefaultSeed
///                         or init::estimatePinholeOpenCv, or a yaml load).
/// @param lock_flags       Ceres parameter locks (PnpFlag bitset).
/// @param options          Single-pass knobs (no strategy).
/// @return CalibrationResult; see struct doc for the status contract.
[[nodiscard]] CalibrationResult calibrate(
  const std::vector<CalibrationView> &views, const CameraModel &initial_model, PnpFlag lock_flags,
  const CalibrationOptions &options
);

/// Pure diagnostic helper: reprojection error of `model` with the given
/// per-view poses against the observed corners. Only pixels whose
/// rayToPixel64 status is OK contribute. Returns the global RMS in pixels;
/// per_view_rms_out is resized to views.size() and max_err_out receives
/// the largest single-point reprojection magnitude. With no valid points
/// every quantity is 0.0. If R/t sizes do not match views.size() the
/// function is a defensive no-op (clears per_view_rms_out, max=0, ret 0).
[[nodiscard]] double computeReprojErrors(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t,
  std::vector<double> &per_view_rms_out, double &max_err_out
);

/// Pure diagnostic helper: the signed per-point reprojection residuals
/// (projected minus observed, pixels) of `model` with the given per-view
/// poses. Outer index = view, inner index = correspondence, aligned with
/// `views`; unprojectable points are (NaN, NaN) so indices never shift.
/// Same residual definition as computeReprojErrors. If R/t sizes do not
/// match views.size() the result is empty.
[[nodiscard]] std::vector<std::vector<Eigen::Vector2d>> computePerPointResiduals(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t
);

}  // namespace camxiom::calib

#endif  // CAMXIOM__CALIB__INTRINSICS_HPP
