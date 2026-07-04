# Jacobian API

Analytical derivatives of the forward projection, for optimization and
uncertainty propagation. Two tiers are provided:

1. the **geometric (point) Jacobian** `∂(u,v)/∂(X,Y,Z)` — for pose
   optimization, feature tracking, covariance propagation;
2. the **full parameter Jacobian** — additionally `∂` with respect to
   intrinsics, distortion coefficients, and projection-level parameters —
   for camera calibration and bundle adjustment.

Supported for all five projection models (PINHOLE, FISHEYE_THETA,
OMNIDIRECTIONAL, DOUBLE_SPHERE, EUCM).

Headers: `<camxiom/jacobian.hpp>`, `<camxiom/jacobian64.hpp>`,
`<camxiom/jacobian_batch.hpp>`, `<camxiom/jacobian_batch64.hpp>`,
`<camxiom/jacobian_with_distortion_deriv64.hpp>`.

## Point Jacobian

```cpp
ProjectionJacobian   rayToPixelWithJacobian(const CameraModel &model,
                                            const Eigen::Vector3f &ray_direction);
ProjectionJacobian64 rayToPixelWithJacobian64(const CameraModel64 &model,
                                              const Eigen::Vector3d &ray_direction);
```

Result (`ProjectionJacobianT<T>`, `[[nodiscard]]`):

```cpp
StatusCode status;            // same semantics as rayToPixel
Pixel2T<T> pixel;             // the projection itself
Eigen::Matrix<T, 2, 3> J;     // d(u,v) / d(X,Y,Z), evaluated at ray_direction
```

`J` is the chain of perspective projection × distortion × intrinsics; row 0
is `[du/dX, du/dY, du/dZ]`, row 1 is `[dv/dX, dv/dY, dv/dZ]`.

## Batch point Jacobians

```cpp
int rayToPixelWithJacobianBatch(const CameraModel &model,
                                const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,
                                Eigen::Ref<Eigen::Matrix2Xf> pixels_out,
                                Eigen::Matrix<float, 2, 3> *jacobians_out,   // N matrices
                                StatusCode *statuses_out = nullptr);

int rayToPixelWithJacobianBatch(const CameraModel &model,
                                const float *rays_xyz, int count,            // interleaved
                                float *u_out, float *v_out,
                                float *jacobians_out,                        // N * 6 floats, row-major
                                StatusCode *statuses_out = nullptr);
```

Same conventions as the [batch projection API](batch.md): returns the number
of successes, `-1` on invalid inputs; `jacobians_out` and `statuses_out` may
be `nullptr`. The raw per-point Jacobian layout is row-major
`[du/dX, du/dY, du/dZ, dv/dX, dv/dY, dv/dZ]`. `rayToPixelWithJacobianBatch64`
mirrors both overloads in double precision.

## Full parameter Jacobian (double precision)

```cpp
FullProjectionJacobian64 rayToPixelWithFullJacobian64(const CameraModel64 &model,
                                                      const Eigen::Vector3d &ray_direction);
```

This is the derivative backbone of the built-in calibrator (the `ANALYTICAL`
PnP cost and the covariance / observability diagnostics). Result fields:

| Field | Meaning |
|---|---|
| `status`, `pixel` | as usual |
| `xd`, `yd` | distorted normalized coordinates *before* intrinsics |
| `J_point` | the 2×3 `∂(u,v)/∂(X,Y,Z)` |
| `dxd_ddist[14]`, `dyd_ddist[14]` | `∂(xd,yd)/∂dist[i]`; first `dist_count` entries valid |
| `dxd_dproj[3]`, `dyd_dproj[3]` | `∂(xd,yd)/∂[xi, alpha, beta]`; first `proj_param_count` entries valid |

Assembling pixel-space derivatives from these:

```
∂(u,v)/∂(fx,fy)   = [[xd, 0], [0, yd]]
∂(u,v)/∂(cx,cy)   = I
∂(u,v)/∂dist[i]   = [fx · dxd_ddist[i],  fy · dyd_ddist[i]]
∂(u,v)/∂proj[j]   = [fx · dxd_dproj[j],  fy · dyd_dproj[j]]     // proj = [xi, alpha, beta]
```

Only the parameters a model actually has are populated (e.g. Mei: `xi`
only; double sphere: `xi, alpha`; EUCM: `alpha, beta`).

## Verification

The analytical Jacobians are cross-checked in the test suite against
finite differences and against Ceres `Jet` automatic differentiation of the
[projection templates](projection.md#projection_templatehpp--autodiff-ready-projection-math),
so the three derivative paths (analytic, AD, numeric) agree on every model.
