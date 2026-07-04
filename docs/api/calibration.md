# Calibration API

The calibration layer (`camxiom::calib`, built with `CAMXIOM_WITH_CERES`)
provides everything between "detected board corners" and "calibrated
camera model":

```
getDefaultSeed ──┐
init::estimate* ─┴─► initial CameraModel ──► calib::calibrate() ──► CalibrationResult
                                              │ per-view DLT-PnP poses      model + poses
                                              │ one PnpSolver::solve        RMS / per-view / per-point errors
                                              └ diagnostics                 covariance + observability
```

Design stance: the library is **strategy-free**. `calibrate()` is exactly
one maximum-likelihood pass — no view selection, no staged unlocking, no
outlier rejection, no multi-restart, no model dispatch. Those policies
belong to the application; camxiom gives it honest primitives and
diagnostics.

Header umbrella: `<camxiom/calibration.hpp>` (Ceres-free to include —
Ceres is an implementation detail hidden behind a PIMPL).

## Observation types

```cpp
// init layer: matrix-column form, one point per column
struct camxiom::init::PlanarObservation {
  Eigen::Matrix2Xd board_pts;   // target-plane (X, Y), Z = 0 implicit
  Eigen::Matrix2Xd image_pts;   // matching pixels
};

// calib layer: vector-list form
struct camxiom::calib::CalibrationView {
  std::vector<Eigen::Vector3d> world_points;   // planar boards: z == 0
  std::vector<Eigen::Vector2d> image_points;   // index-aligned
};
```

## Seeds: `getDefaultSeed`

```cpp
[[nodiscard]] CameraModel getDefaultSeed(ProjectionModelType type, int width, int height);
```

A data-independent seed — the canonical warm start when no data-driven
guess exists. Principal point at the image center; focal heuristic per
family (`h/2` for rectilinear/unified models, `h/π` for equidistant
fisheye); projection parameters at literature-typical values
(fisheye `θ_max = π`; Mei `ξ = 1.0`; DS `ξ = −0.2, α = 0.5`;
EUCM `α = 0.5, β = 1.0`). Guaranteed to validate for any valid input;
invalid input returns an `UNKNOWN`-type sentinel that
`validateCameraModel` rejects.

## Init estimators (`camxiom::init`)

Data-driven initial guesses. Common contract: planar estimators need
≥ 3 views with ≥ 4 correspondences each; on any non-OK status the outputs
are left untouched (atomic writes); `DEGENERATE_CONFIG` means "the data
cannot identify a solution" (vs `INVALID_INPUT` = caller bug,
`NUMERIC_ERROR` = numerical pathology).

| Function | Output | Method | Backend |
|---|---|---|---|
| `estimateHomographyDLT(src, dst, H_out)` | 3×3 `H`, `‖H‖_F = 1` | Hartley-normalized DLT (MVG Alg. 4.2) | Eigen |
| `estimatePinholeZhang(views, K_out)` | upper-triangular `K` | Zhang's IAC linear init (PAMI 2000); skew estimated freely | Eigen |
| `estimatePinholeOpenCv(views, w, h, aspect, K_out)` | `K` (principal point fixed at center) | OpenCV `initCameraMatrix2D`-compatible focal solve; `aspect = 0` → fx, fy independent | Eigen |
| `estimatePoseDLT(model64, world, image, R_out, t_out)` | pose `R, t` | model-agnostic DLT-PnP on bearings from `pixelToRay64`; handles planar (9-DOF) and non-planar (12-DOF) automatically | Eigen |
| `estimatePoseRefined(model, world, image, R_out, t_out)` | pose `R, t` | `estimatePoseDLT` + pose-only nonlinear refinement — the OpenCV-free analogue of `cv::solvePnP(ITERATIVE)` | Eigen + Ceres |
| `estimateKB4Init(views, w, h, result)` | `K`, `D(k1..k4)`, per-view poses | linear focal/k1/k2 bootstrap, then best-effort Ceres polish; **k3, k4 stay 0** (unobservable from board geometry) | Eigen + Ceres |
| `estimateMEIInit(views, w, h, result)` | `K`, `ξ`, poses | heuristic seed + Ceres K/pose polish, `ξ` locked at 1.0 | Eigen + Ceres |
| `estimateDSInit(views, w, h, result)` | `K`, `ξ`, `α`, poses | same pattern, `ξ = −0.2`, `α = 0.5` locked | Eigen + Ceres |
| `estimateEUCMInit(views, w, h, result)` | `K`, `α`, `β`, poses | same pattern, `α = 0.5`, `β = 1.0` locked | Eigen + Ceres |

Why the unified models (Mei / DS / EUCM) have no Zhang-style linear step:
their forward equations contain `‖P‖` square-root terms, so the
board-to-image map is **not a homography** for non-zero sphere offset —
an IAC init would be mathematically invalid. The projection parameters are
locked at literature seeds during init and refined later in the full
calibration. Init estimators target "inside the Ceres convergence basin",
not final accuracy.

## `PnpSolver` (`camxiom::optimizer`)

A PnP / intrinsic-refinement solver that works with **any** camxiom camera
model through a single templated cost — no per-model solver code. Ceres is
fully hidden (PIMPL): the public header exposes only plain structs.

```cpp
PnpSolver solver;                          // move-only
PnpResult result;
bool ok = solver.solve(object_point_sets,  // std::vector<std::vector<Eigen::Vector3d>>
                       image_point_sets,   // std::vector<std::vector<Eigen::Vector2d>>
                       initial_guess,      // PnpInitialGuess{ camera_model, rvecs, tvecs }
                       result,
                       options,            // PnpSolverOptions
                       flags);             // PnpFlag — what stays FIXED
const PnpSummary &s = solver.lastSummary();
```

### `PnpFlag` — per-parameter freeze flags

`enum class PnpFlag : uint64_t`, combinable with `|`:

| Flag | Fixes |
|---|---|
| `NONE` | nothing (all free) |
| `FIX_EXTRINSICS` | all per-view poses |
| `FIX_FOCAL_LENGTHS` / `FIX_PRINCIPAL_POINTS` / `FIX_INTRINSICS` | fx, fy / cx, cy / all four |
| `FIX_PROJECTION_PARAMS` | ξ / α / β |
| `FIX_DIST_0` ... `FIX_DIST_13` | individual distortion coefficients |
| `FIX_DISTORTION` | all 14 |

Helpers: `hasFlag`, `hasAnyFlag`, `hasAllFlags`, `unsetFlags`, `toUnderlying`.

Staged optimization is the intended pattern for the wide-angle families —
run the same solver instance several times, progressively unfreezing:

```cpp
// stage 1: poses only (this combination takes a Ceres-free Gauss-Newton fast path)
solver.solve(obj, img, guess, r1, opts,
             PnpFlag::FIX_INTRINSICS | PnpFlag::FIX_DISTORTION | PnpFlag::FIX_PROJECTION_PARAMS);
// stage 2: + intrinsics & distortion
solver.solve(obj, img, {r1.camera_model, r1.rvecs, r1.tvecs}, r2, opts,
             PnpFlag::FIX_PROJECTION_PARAMS);
// stage 3: everything free
solver.solve(obj, img, {r2.camera_model, r2.rvecs, r2.tvecs}, r3, opts, PnpFlag::NONE);
```

### Options and results

```cpp
struct PnpOptimizerOptions {   // backend-agnostic; defaults mirror Ceres
  int max_num_iterations{1000};
  double function_tolerance{1e-6}, parameter_tolerance{1e-8}, gradient_tolerance{1e-10};
  bool minimizer_progress_to_stdout{false};
};
struct PnpSolverOptions {
  PnpOptimizerOptions solver_options{};
  double huber_loss_delta{1.0};
  bool print_summary{false};
  PnpBound upper_bound, lower_bound;     // box bounds (focal / principal point / rotation / translation)
  PnpCostType cost_type{PnpCostType::ANALYTICAL};   // or AUTO_DIFF
};
struct PnpResult {             // [[nodiscard]]
  bool success; double rmse;
  CameraModel camera_model;
  std::vector<Eigen::Vector3d> rvecs, tvecs;        // Rodrigues + translation per view
  std::size_t valid_count, total_count;
};
struct PnpSummary {
  int num_successful_steps, num_unsuccessful_steps;
  double final_cost;
  bool solution_usable;
  bool converged;   // tolerances met — an iteration-capped run can be usable without being converged
};
```

Three cost paths: `ANALYTICAL` (default — manual Jacobians via
`rayToPixelWithFullJacobian64`), `AUTO_DIFF` (Ceres `Jet` through the
[projection templates](projection.md)), and a Ceres-free **Gauss–Newton fast
path** taken automatically for pose-only solves (everything except
extrinsics fixed). Note the box bounds apply only to the Ceres paths; the
GN fast path runs unconstrained.

## `calib::calibrate` — single-pass intrinsics calibration

```cpp
[[nodiscard]] CalibrationResult calibrate(const std::vector<CalibrationView> &views,
                                          const CameraModel &initial_model,
                                          PnpFlag lock_flags,
                                          const CalibrationOptions &options);
```

Pipeline: validate inputs → per-view pose via `init::estimatePoseDLT`
against `initial_model` (any view failing aborts the call — skipping views
is application strategy) → one `PnpSolver::solve` over all corners →
reprojection + uncertainty diagnostics.

**`lock_flags` is the caller's identifiability policy** and is mandatory
knowledge for wide-angle models — leaving non-identifiable parameters free
drives the solve into aliasing minima (huge parameter values while RMS
still looks fine):

| Model | Required locks |
|---|---|
| pinhole (radtan) | `NONE` |
| KB4 fisheye | `FIX_DIST_2 \| FIX_DIST_3` (k3, k4) |
| Mei / DS / EUCM | `FIX_PROJECTION_PARAMS` |

### `CalibrationOptions` (excerpt)

| Field | Default | Meaning |
|---|---|---|
| `image_width` / `image_height` | 0 | required |
| `max_iterations` | 200 | Ceres iteration cap |
| `huber_loss_delta` | 0.0 | 0 = ordinary least squares (matches `cv::calibrateCamera`) |
| `apply_initial_value_bounds` (+`bound_relative_tolerance`) | false (±10 %) | box-constrain focals around the seed — for "fine-tune a stored calibration" |
| `apply_principal_point_bounds` (+reference, tolerance) | false | same for the principal point |
| `estimate_uncertainty` | **true** | fill the covariance + observability diagnostics (one extra Jacobian pass) |
| `compute_per_point_residuals` | false | fill signed per-correspondence residuals (NaN for unprojectable points) |

### `CalibrationResult`

Core: `status`, `camera_model`, `per_view_rotations` / `_translations`,
`rms_reprojection_error_px`, `max_reprojection_error_px`, `per_view_rms_px`,
`per_point_residuals`, `iterations_used`.

Status contract: `ok()` is strict (`status == OK`). `NON_CONVERGED` still
carries a best-effort model, poses, and diagnostics (useful as a warm
restart). Early failures (validation, degenerate DLT) return atomically —
only `status` is set.

**Parameter uncertainty** (`uncertainty_available`, `uncertainty_labels`,
`parameter_std`): 1-σ standard deviations of every *free* parameter, from
the reduced normal equations JᵀJ with per-view extrinsics marginalized out
(Schur complement), scaled by residual variance. A local Gauss–Newton / OLS
estimate: optimistic near active box bounds and under robust loss.

**Observability diagnostic** (`observability_available`, `observability_ok`,
`normalized_condition_number`, `min_singular_value`,
`underdetermined_parameters`): the same reduced normal matrix, Jacobi-scaled
(scale-invariant), eigen-analyzed. `available && !ok` is a *successful
diagnosis of degeneracy* — the classic silent failure where RMS is low but,
say, focal length is not observable from all-frontoparallel boards; the
offending parameters are named. Diagnostic only: camxiom never changes
`lock_flags`, drops views, or regularizes a degenerate system silently.

**Near-bounds report** (`uncertainty_has_parameters_near_bounds`,
`parameters_at_or_near_bounds`): free bounded parameters whose final value
sits at/near a solver box bound (a proximity test on the final estimate).

### Diagnostics helpers (pure functions)

```cpp
double computeReprojErrors(views, model, Rs, ts, per_view_rms_out, max_err_out);
std::vector<std::vector<Eigen::Vector2d>> computePerPointResiduals(views, model, Rs, ts);
```

## See also

- [Model conversion](conversion.md) — refit a calibrated model into another
  family.
- [Design / numerical strategy](../design/numerical-strategy.md) — why
  identifiability locks exist, solver internals.
