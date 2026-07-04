# Numerical strategy

The design decisions behind camxiom's numerics — what a user should know to
trust (and correctly interpret) the results.

## Forward-only authoring, shared inverse solvers

Each camera model implements only its **forward** projection (plus
analytical Jacobians). The inverse (`pixelToRay`) is numerical, through two
shared solvers:

- **Plane solver** — 2D Newton iteration with a Levenberg–Marquardt
  fallback (~3–8 iterations typical) for the pinhole-family distortions
  (radtan / rational / thin-prism / tilted), and for Mei / DS / EUCM after
  their model-specific back-projection.
- **Angle solver** — 1D hybrid of bisection and Newton (~5–15 iterations)
  solving `θ_d(θ) = r/f` over the monotonic range `[0, theta_max]` for the
  fisheye family.

This is why adding a model is cheap and why inverse behaviour is uniform:
convergence policy, tolerances, and failure reporting (`NON_CONVERGED`)
live in one place, controlled by `SolverOptions`.

### Per-precision solver defaults are intentional

float: 10 iterations / 1e-6 residual; double: 15 / 1e-12. The double path
is meant for calibration-grade accuracy, the float path for real-time use.
Perceived float-vs-double inverse speed differences come from these
*settings*, not from the scalar type.

## `theta_max` and FOV handling

For fisheye models `theta_max` is both the FOV limit and the upper end of
the theta solver's search bracket, so it must stay consistent with the
distortion polynomial (which is only invertible on its monotonic range).
Everything that mutates coefficients refreshes it (`updateThetaMax` after
manual edits; factories and `calibrate`/`convertCameraModel` handle it
internally). Domain checks reject rather than clamp: the Mei injectivity
limit, the double-sphere bijectivity condition (Usenko 2018 eq. 43–45), the
EUCM validity condition, and the orthographic model's non-injective rear
hemisphere all return non-OK statuses instead of folding rays onto wrong
pixels.

The consistency is *enforced*, in two tiers matched to where the cost can
be paid. The public `validateCameraModel` / `validateCameraModel64` (and
everything built on them: `ValidatedCameraModel::tryMake`, the LUT/remap
builders, factories, calibration) certify via a sampled scan (~π/512
resolution) that a polynomial-fisheye cap sits inside the polynomial's
positive monotone range — so `OK` from the oracle means the forward map is
injective on `[0, theta_max]` and `pixelToRay` can invert every pixel
`rayToPixel` emits. The per-point query paths (`rayToPixel`, `pixelToRay`,
Jacobians, batch) run a cheap structural guard instead — all structural
checks plus a single derivative-sign test at the cap, which already rejects
the common single-fold case — identical across the scalar, batch, SIMD and
Jacobian paths, so a given model is accepted or rejected the same way on
every query route. A hand-set cap past the fold is `INVALID_MODEL`, with
`updateThetaMax` as the documented repair.

## Float-primary with a structurally shared validator

The runtime API is float32; calibration consumes a widened `CameraModel64`
copy. That creates one subtle class of bug: a boundary constant expressed
in float and the "same" constant in double differ by sub-ULP amounts
(float π widened to double lands ~8.7e-8 *above* true double π). At one
point the double validator used true double π as the `theta_max` bound, so
a full-FOV fisheye seed accepted by the float validator was rejected after
widening — every `pixelToRay64` returned `INVALID_MODEL`, and calibration
failed with a misleading `DEGENERATE_CONFIG`.

The fix is structural, not a tolerance: validation (like the rest of the
type system and per-model math) is single-sourced from one `T`-templated
implementation, and boundary constants are expressed so both precisions
share the *same accept/reject edge* by definition (the float-π edge, cast
per precision). Float and double behaviour can no longer drift apart —
the property is enforced by code shape, not by a comment asking two
implementations to stay in sync.

The same review pass fixed the double-sphere `ξ` range: real DS cameras
have `ξ ∈ (−1, 0)` (typically around −0.2), so validators and solver
bounds accept the open interval `(−1, 1)` rather than `ξ ≥ 0`.

## Identifiability and the lock-flag philosophy

Wide-angle model families contain parameter directions that planar-board
data cannot observe:

- KB4/KB8's high-order terms (θ⁷, θ⁹) sit below the corner-noise floor for
  typical board geometry — freeing k3/k4 yields aliasing minima with huge
  coefficients and innocent-looking RMS.
- Mei / DS / EUCM projection parameters (ξ, α, β) trade off against focal
  length; joint optimization from a cold start routinely diverges.

camxiom's stance: the library never silently fixes, drops, or regularizes
anything. Instead it gives the caller `PnpFlag` locks (and documents the
per-model required minimum), staged optimization as the recommended
pattern, and — after the solve — the observability diagnostic that *names*
underdetermined parameters (Jacobi-scaled reduced normal matrix, condition
number, weak-subspace projection). Low RMS with an unobservable focal
length is exactly the silent failure this machinery exists to expose.

## Uncertainty numbers: what they are and are not

`CalibrationResult::parameter_std` is a local Gauss–Newton / OLS covariance
estimate: `JᵀJ` from the full analytical parameter Jacobian, per-view poses
marginalized by Schur complement, scaled by residual variance
(`σ² = RSS/dof`). It is a *lower-bound-flavored* estimate — optimistic when
box bounds are active (reported separately via the near-bounds list) or
when robust loss is used (the raw, unweighted Jacobian is analyzed).
Distortion-coefficient σ values are in their own dimensionless scales and
only comparable relatively.

## SIMD strategy (why float and double differ)

Float is the throughput API: all five models have vectorized batch kernels
(8-wide AVX2 / 4-wide SSE2, run on aarch64 through a minimal NEON
compatibility layer). Double is the accuracy API: only the undistorted
pinhole fast path is vectorized; the rest is scalar under OpenMP. The
asymmetry is a deliberate optimization-target split, not drift — the two
precisions share their scalar math sources.

Two measured lessons are baked into the dispatch:

- **SIMD and OpenMP must compose.** Naively calling a single-threaded SIMD
  kernel lost to the scalar-OpenMP path by 2–18× on a 12-core Jetson Orin;
  batches ≥ 8192 are therefore chunked per thread in whole SIMD groups.
- **Vectorization is not always a win.** The KB-family fisheye inverse on
  aarch64 stays scalar — its mandatory round-trip verification plus
  per-lane trig measured slower than the scalar solver. Profitability is
  decided per model × precision × architecture, from measurements.

Lanes a SIMD kernel flags invalid are recomputed through the scalar
reference path, keeping batch ≡ single-point results exactly.

## Verification approach

- Scalar ↔ SIMD ↔ batch parity tests across models, precisions, and lane
  configurations; property-based invariants (round-trip
  `pixelToRay(rayToPixel(r)) ≈ r` inside the valid domain) with fixed
  seeds.
- Analytic Jacobians cross-checked against finite differences *and* Ceres
  `Jet` AD of the same projection templates.
- Error paths are tested deliberately (DOMAIN_ERROR / NUMERIC_ERROR
  generators), and real-correspondence regression fixtures pin calibration
  behaviour on recorded data.
- Performance baselines (`benchmarks/`) protect the properties that matter:
  e.g. TILTED14's cached tilt matrices keep forward projection trig-free
  per point — a regression there is a benchmark diff, not a code-review
  guess.
