# Error handling

## `StatusCode`, not `bool`, not exceptions

Every fallible operation reports a `StatusCode` — either directly or inside
a `[[nodiscard]]` result struct (`PixelResult`, `RayResult`,
`CalibrationResult`, ...). The enum distinguishes *why* something failed,
because callers legitimately branch on the reason:

| Code | Meaning | Typical reaction |
|---|---|---|
| `OK` | success | — |
| `INVALID_INPUT` | bad argument (sizes, non-finite values, null buffers) | fix the caller |
| `INVALID_MODEL` | the `CameraModel` failed validation | fix the model description |
| `BEHIND_CAMERA` | pinhole ray with `z ≤ 0` | expected for points behind the camera — skip |
| `OUT_OF_FOV` | ray beyond `theta_max` / the model's valid domain | expected at FOV edges — skip |
| `DOMAIN_ERROR` | math domain violation during evaluation | treat point as unprojectable |
| `NON_CONVERGED` | iterative solver hit its caps without meeting tolerance | retry with looser options, or accept best-effort where offered |
| `NUMERIC_ERROR` | numerical pathology (rank loss mid-computation, non-finite results) | investigate data/algorithm |
| `DEGENERATE_CONFIG` | input geometry *structurally* cannot identify a unique solution (collinear points, coplanar 3D sets, near-parallel board views) | provide different / more diverse data |

The split between `DEGENERATE_CONFIG` and `NUMERIC_ERROR` is intentional:
the first says "your data cannot answer this question", the second says
"something went numerically wrong while answering it". They demand
different fixes.

## Hard-to-ignore by construction

`StatusCode` itself is declared `[[nodiscard]]`, so *any* function returning
it warns when the result is dropped — intentional discards must be written
`(void)fn(...)`. Result structs are `[[nodiscard]]` too and carry `ok()` +
`explicit operator bool()` for terse checking.

## Fail closed, never garbage

- **No NaN propagation:** on failure, numeric outputs are zero-initialized,
  never left as NaN — a dropped status then produces visibly wrong (0,0)
  data instead of NaNs that surface three subsystems later. (Deliberate
  exception: `calibrate()`'s optional per-point residuals mark
  unprojectable points as NaN, documented, so plots skip them naturally.)
- **Atomic outputs:** estimators and `calibrate()` leave their out-params /
  result fields untouched on a non-OK status (only `status` is set), so a
  pre-populated result buffer is never half-overwritten.
- **Unknown model types fail, not fallback:** the projection dispatch
  returns failure for an unrecognized projection/distortion type instead of
  silently projecting distortion-free — a silent fallback would bias any
  optimizer sitting on top.
- **Type/space mismatches are caught at validation:** `DistortionSpace`
  (PLANE vs ANGLE) makes "pinhole + Kannala-Brandt coefficients" an
  `INVALID_MODEL` at `validateCameraModel` time rather than silent garbage
  at projection time. Validation is two-tier: the public
  `validateCameraModel`/`64` oracle additionally certifies (by sampled scan)
  that a polynomial-fisheye `theta_max` sits inside the polynomial's
  monotone range — the guarantee behind "every pixel the forward emits, the
  inverse can invert" — while the per-point query paths run the cheap
  structural subset, identically across scalar / batch / SIMD / Jacobian
  routes (see design/numerical-strategy.md).
- **Remap sentinels:** map builders write `-1` for unmappable pixels; the
  sampling kernel converts them to the caller's fill value and counts them
  (`border_count`), so "how much of my output is real" is always
  observable.

## Best-effort results are labeled, not hidden

`calibrate()` and `convertCameraModel()` return `NON_CONVERGED` *with* the
best-effort model and full diagnostics — usable as a warm restart — but
`ok()` stays false, and `PnpSummary::converged` distinguishes "met
tolerances" from "stopped at the iteration cap but usable". Code that
promises convergence must check `converged`, not just `solution_usable`.

The same honesty rule shapes the diagnostics: the observability check
reports degeneracy (`observability_available && !observability_ok`) as a
*successful diagnosis* naming the underdetermined parameters — camxiom
never silently regularizes, drops views, or edits the caller's lock flags.
