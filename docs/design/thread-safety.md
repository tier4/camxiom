# Thread safety

## Stateless free functions — safe everywhere

The projection / inverse-projection / Jacobian free functions
(`rayToPixel`, `pixelToRay`, `rayToPixelWithJacobian`, `isRayProjectable`,
`computeFov`, the batch variants, the compat conversions, ...) are
stateless: they only read the `CameraModel` (or `CameraInfo`) passed in.
Concurrent calls from multiple threads — including on a **shared model** —
are safe.

Batch projection and the remap-map builders may parallelize internally with
OpenMP; calling them is still safe from any thread. They write only to
caller-provided or freshly returned buffers, so two concurrent batch calls
must simply target different output buffers (the natural usage).

## Stateful objects — exclusive per instance while mutating

| Object | Mutating operations | Concurrent-read rule |
|---|---|---|
| `InverseLut` / `InverseLut64` | `build()`, `clear()` | after `build()` returns, concurrent `query()` (const) is safe |
| `ValidatedCameraModel` | none after construction (immutable) | fully safe to share and call concurrently |
| `RemapCache` (OpenCV layer) | `build*()`, `clear()` | after building, concurrent `apply()` to distinct outputs is safe |
| `PnpSolver` | `solve()` mutates internal state; `lastSummary()` reads it | one solver instance per thread, or external synchronization; `lastSummary()` refers to *this instance's* most recent `solve()` |

The general rule: **one writer, then many readers**. Use one instance per
thread when in doubt; none of these objects share hidden global state, so
per-thread instances scale without contention.

## Global state

There is none: no singletons, no global caches, no lazily-initialized
tables. Library behaviour is a pure function of the arguments (plus OpenMP's
thread pool for the parallel loops).
