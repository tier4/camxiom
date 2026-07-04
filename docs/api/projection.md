# Projection API

Single-point forward (`ray → pixel`) and inverse (`pixel → ray`) projection.
This is the primary API of the library: one generic entry point per
direction, dispatched over the model's `ProjectionModelType` after
validation. Per-model entry points are an internal implementation detail —
the public surface never grows when a model is added.

Headers: `<camxiom/projection.hpp>` (float32), `<camxiom/projection64.hpp>`
(float64), `<camxiom/validated_model.hpp>`, `<camxiom/projection_template.hpp>`.

## Forward projection

```cpp
PixelResult rayToPixel(const CameraModel &model, const Eigen::Vector3f &ray_direction);
PixelResult rayToPixel(const CameraModel &model, float x, float y, float z);
PixelResult rayToPixel(const CameraModel &model, const Ray3 &ray);   // origin ignored (central projection)

PixelResult64 rayToPixel64(const CameraModel64 &model, const Eigen::Vector3d &ray_direction);
```

- Direction points from the camera center into the scene; +Z is forward.
  The direction need not be unit length.
- Result: `PixelResult{ StatusCode status; Pixel2 pixel; }` with `.ok()` and
  `explicit operator bool()`. Typical non-OK statuses: `BEHIND_CAMERA`
  (pinhole, `z <= 0`), `OUT_OF_FOV` (past `theta_max` / model domain),
  `INVALID_MODEL` (validation failed).

## Inverse projection

```cpp
RayResult pixelToRay(const CameraModel &model, const Pixel2 &pixel,
                     const SolverOptions &solver_options = SolverOptions{});
RayResult pixelToRay(const CameraModel &model, float u, float v,
                     const SolverOptions &solver_options = SolverOptions{});

RayResult64 pixelToRay64(const CameraModel64 &model, const Pixel2d &pixel,
                         const SolverOptions64 &solver_options = SolverOptions64{});
```

- Returns `RayResult{ status, ray }`; `ray.direction` is a **unit vector**.
- Undistortion is numerical for models without a closed-form inverse; the
  iterative solver is governed by `SolverOptions`:

| Field | float default | double default |
|---|---|---|
| `max_iterations` | 10 | 15 |
| `residual_tolerance` | 1e-6 | 1e-12 |
| `step_tolerance` | 1e-8 | 1e-14 |
| `skip_verify` | false | false |

  A solve that exhausts iterations without meeting tolerance returns
  `NON_CONVERGED`. See [Numerical strategy](../design/numerical-strategy.md)
  for the plane / angle solver design.

## Domain predicate

```cpp
bool isRayProjectable(const CameraModel &model, const Eigen::Vector3f &dir);
bool isRayProjectable(const CameraModel &model, const Eigen::Vector3f &dir,
                      int image_width, int image_height);
```

True iff `rayToPixel` would return `OK` (the second overload additionally
requires the pixel to land inside `[0, W) × [0, H)` — the same in-bounds rule
the remap builders apply). It performs the projection, so it is a
readability helper, not a cheaper path.

## `ValidatedCameraModel` — validated hot loops

The generic functions re-validate the model on every call — correct for a
trust-nothing free function, wasteful in a hot loop. `ValidatedCameraModel`
makes "valid by construction" a type property:

```cpp
#include <camxiom/validated_model.hpp>

auto vm = camxiom::ValidatedCameraModel::tryMake(model);   // std::optional
if (!vm) { /* validation failed; call validateCameraModel(model) for the reason */ }

for (const auto &p : cloud) {
  auto px = vm->rayToPixel(p);          // no re-validation, no dispatch switch
  ...
}
```

- `tryMake()` validates once (identical criteria to `validateCameraModel`)
  and resolves the per-model forward/inverse entry points once.
- The wrapped model is immutable (`get()` to read it); results are
  bit-for-bit identical to the generic path.
- Member `rayToPixel` / `pixelToRay` overloads mirror the free functions.
- float32 only (a `ValidatedCameraModel64` may be added later).

Measured effect (aarch64 scalar, N=65536 forward loop): roughly 3.3× for
pinhole+radtan5, 1.9× for fisheye+KB4, 2.8× for pinhole+tilted14 over the
generic per-call path. For bulk data prefer the [batch API](batch.md), which
hoists validation and dispatch out of the loop and adds SIMD/OpenMP.

## `projection_template.hpp` — AutoDiff-ready projection math

Templated forward-projection functions usable with any scalar type `T`
(`float`, `double`, or `ceres::Jet<double, N>`). Pure math: the header
depends only on `<cmath>` and the camxiom types — deliberately **not** on
Ceres or Eigen — so it can serve as the body of any cost functor. This is
the same single code path the built-in PnP solver uses for its `AUTO_DIFF`
cost.

Namespace `camxiom::projection_template`:

```cpp
// Dispatch by model (reads parameters from a CameraModel):
template <typename T>
bool projectGeneric(const CameraModel &model,
                    const T &x_cam, const T &y_cam, const T &z_cam,
                    T &u_out, T &v_out);

// Dispatch with every parameter as a separate raw block
// (for independent Ceres parameter blocks):
template <typename T>
bool projectGenericParametric(ProjectionModelType proj_type,
                              DistortionModelType dist_type,
                              const T *intrinsics,      // [fx, fy, cx, cy]
                              const T *dist, int dist_count,
                              const T &proj_xi, const T &proj_alpha, const T &proj_beta,
                              const T &x_cam, const T &y_cam, const T &z_cam,
                              T &u_out, T &v_out);
```

Per-model building blocks are also public (`projectRadtan5`,
`projectRational8`, `projectThinPrism12`, `projectTilted14`,
`projectFisheye4`, `projectKB4`, `projectKB8`, `projectEquidistant`,
`projectEquisolid`, `projectStereographic`, `projectOrthographic`,
`projectOmnidirectional`, `projectDoubleSphere`, `projectEucm`). All follow
one contract: return `bool`, write `(u_out, v_out)`, and on a domain failure
write zeros and return `false`. Unknown model/distortion types return
`false` rather than silently projecting distortion-free (which would bias an
optimizer).

Sketch of a Ceres cost functor:

```cpp
struct ReprojCost {
  template <typename T>
  bool operator()(const T *intr, const T *dist, const T *pose, T *residual) const {
    T pc[3] = /* world point transformed by pose */;
    T u, v;
    if (!camxiom::projection_template::projectGenericParametric(
            proj_type_, dist_type_, intr, dist, dist_count_,
            xi_, alpha_, beta_, pc[0], pc[1], pc[2], u, v)) {
      return false;
    }
    residual[0] = u - T(observed_u_);
    residual[1] = v - T(observed_v_);
    return true;
  }
};
```

Note that these templates apply `u = fx·x + cx` without skew (the
4-element intrinsics block is `[fx, fy, cx, cy]`).
