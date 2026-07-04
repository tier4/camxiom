# Batch projection API

Bulk forward / inverse projection over many points with one call. The model
is validated once, the per-model dispatch is hoisted out of the loop, and the
implementation uses SIMD kernels and OpenMP chunking where profitable.

Headers: `<camxiom/batch.hpp>` (float32), `<camxiom/batch64.hpp>` (float64).

## API

Each direction has an Eigen overload and a raw-pointer overload, in both
precisions. All return `int`: the **number of successfully projected
points**, or **`-1`** when the inputs themselves are invalid (mismatched
Eigen shapes, negative `count`, required pointer `nullptr`). Per-point
failures are reported through the optional `statuses_out` array (size
`count`, may be `nullptr`).

```cpp
// Forward: rays -> pixels
int rayToPixelBatch(const CameraModel &model,
                    const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,  // 3 x N
                    Eigen::Ref<Eigen::Matrix2Xf> pixels_out,                   // 2 x N (row 0 = u, row 1 = v)
                    StatusCode *statuses_out = nullptr);

int rayToPixelBatch(const CameraModel &model,
                    const float *rays_xyz,     // interleaved [x0,y0,z0, x1,y1,z1, ...], 3*N
                    int count,
                    float *u_out, float *v_out,          // planar, N each
                    StatusCode *statuses_out = nullptr);

// Inverse: pixels -> unit ray directions
int pixelToRayBatch(const CameraModel &model,
                    const Eigen::Ref<const Eigen::Matrix2Xf> &pixels,          // 2 x N
                    Eigen::Ref<Eigen::Matrix3Xf> directions_out,               // 3 x N, unit vectors
                    StatusCode *statuses_out = nullptr,
                    const SolverOptions &solver_options = SolverOptions{});

int pixelToRayBatch(const CameraModel &model,
                    const float *u_in, const float *v_in, int count,           // planar
                    float *dirs_xyz,                                           // interleaved, 3*N
                    StatusCode *statuses_out = nullptr,
                    const SolverOptions &solver_options = SolverOptions{});
```

`rayToPixelBatch64` / `pixelToRayBatch64` mirror these with `CameraModel64`,
`double`, and `SolverOptions64`.

Layout summary (note the deliberate asymmetry, matched to producer/consumer
data shapes):

| Direction | Input | Output |
|---|---|---|
| forward (raw) | interleaved `[x,y,z]...` | planar `u[]`, `v[]` |
| inverse (raw) | planar `u[]`, `v[]` | interleaved `[x,y,z]...` |
| Eigen | column-major `3×N` / `2×N` | column-major `2×N` / `3×N` |

Points that fail per-point (e.g. `OUT_OF_FOV`) get zeroed outputs and their
`StatusCode` in `statuses_out`; they simply don't count toward the return
value.

## Performance characteristics

The public contract is only "same results as calling the single-point API in
a loop". The acceleration below is implementation detail — useful for
capacity planning, not something to program against.

- **The raw-pointer float overloads are the fastest path**: they feed
  SIMD kernels directly. The Eigen overloads and most of the double API run
  scalar per-point code under OpenMP.
- **Vectorized (float32)**: pinhole, fisheye-theta (OPENCV_FISHEYE4 / KB4 /
  EQUIDISTANT distortion), omnidirectional, double sphere, EUCM — 8-wide on
  AVX2, 4-wide on SSE2. On aarch64 the same 4-wide kernels run through a
  minimal NEON compatibility layer (AArch64 only; 32-bit ARM stays scalar).
- **Vectorized (float64)**: only the undistorted-pinhole fast path (4-wide
  AVX on x86, 2-wide NEON on aarch64); everything else is scalar + OpenMP.
  This asymmetry is deliberate: float is the real-time API, double is the
  calibration API.
- **OpenMP chunking**: batches of ≥ 8192 points are split into per-thread
  chunks aligned to whole SIMD groups, so the SIMD and multi-core speedups
  compose. Below the threshold a single-threaded kernel runs (fork/join
  overhead would dominate). Requires an OpenMP-enabled build.
- **Model-specific exception**: on aarch64 the KB-family fisheye *inverse*
  (KB4 / KB8 / OPENCV_FISHEYE4) intentionally stays on the scalar path — the
  vectorized version measured slower there (round-trip verification + scalar
  trig per lane).
- SIMD lanes that a kernel flags as invalid are recomputed through the
  scalar reference path, so batch and single-point results agree.

## Choosing an API

| Situation | Use |
|---|---|
| A handful of points, or clarity first | single-point `rayToPixel` / `pixelToRay` |
| Same model, many points per frame | `rayToPixelBatch` / `pixelToRayBatch` (raw-pointer float overloads for maximum throughput) |
| Repeated single-point calls on one validated model | [`ValidatedCameraModel`](projection.md#validatedcameramodel--validated-hot-loops) |
| Repeated approximate inverse over a fixed image grid | [`InverseLut`](lut.md) |
| Warping whole images | [remap maps + kernel](remap.md) |
