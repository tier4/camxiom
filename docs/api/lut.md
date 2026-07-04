# Inverse LUT

`InverseLut` precomputes `pixelToRay` on a regular grid over the image and
answers arbitrary queries by bilinear interpolation — an O(1), allocation-free
approximate inverse for hot paths where the iterative solver per pixel would
be too expensive (per-pixel unprojection at video rate, LUT-style sensor
fusion pipelines).

Headers: `<camxiom/lut.hpp>` (`InverseLut = InverseLutT<float>`),
`<camxiom/lut64.hpp>` (`InverseLut64 = InverseLutT<double>`).

## Usage

```cpp
#include <camxiom/lut.hpp>

camxiom::InverseLut lut;
lut.build(model, 1280, 720);             // step = 1 -> full-resolution grid
lut.build(model, 1280, 720, {}, 4);      // step = 4 -> 320 x 180 grid, ~16x less memory

camxiom::RayResult r = lut.query(500.3f, 300.7f);   // O(1) bilinear interpolation
if (r.ok()) { /* r.ray.direction — approximate unit ray */ }
```

## API

```cpp
template <typename T> class InverseLutT {
 public:
  // Evaluate pixelToRay on a regular grid. Returns the number of VALID grid
  // points (grid nodes whose exact unprojection succeeded).
  int build(const CameraModelT<T> &model, int width, int height,
            const /* SolverOptions or SolverOptions64 */ &solver_options = {},
            int step = 1);

  RayResultT<T> query(T u, T v) const;   // bilinear; approximate direction

  bool isValid() const;                  // built?
  int gridWidth() const;  int gridHeight() const;
  int imageWidth() const; int imageHeight() const;
  int step() const;
  void clear();                          // free memory
};
```

- The float LUT takes `SolverOptions` (10 iterations / 1e-6 defaults), the
  double LUT takes `SolverOptions64` (15 / 1e-12) — each precision keeps its
  own default solver behaviour for the grid construction.
- Grid nodes where the exact inverse fails (outside FOV, non-converged) are
  flagged invalid; queries interpolating into invalid nodes return a non-OK
  status instead of a fabricated direction. `build`'s return value is the
  coverage indicator.
- Storage is structure-of-arrays (`x/y/z` direction components + 1-byte
  validity per node), so memory is roughly
  `(3·sizeof(T) + 1) · (W/step) · (H/step)` bytes.

## Accuracy / cost trade-off

- `query` is *approximate*: bilinear interpolation between grid samples,
  re-normalized direction. Accuracy degrades with larger `step` and with
  stronger local distortion curvature (image corners of wide fisheyes).
- Use the exact [`pixelToRay`](projection.md) (or the
  [batch inverse](batch.md), which is SIMD-accelerated) when you need
  solver-grade precision; use the LUT when a sub-pixel-interpolated
  direction is good enough and per-query latency matters.
- Build cost is one exact inverse per grid node (parallelized with OpenMP in
  OpenMP-enabled builds); rebuild only when the model changes
  (`model_compare.hpp`'s `operator==` is designed for exactly this
  cache-invalidation check).

## Thread safety

`build()` mutates the instance — one builder at a time. After `build`
completes, concurrent `query()` calls are safe (const, no internal state).
See [Thread safety](../design/thread-safety.md).
