# camxiom

**camxiom** is a unified C++17 camera-geometry library: one API across the
whole spectrum of camera models — **pinhole**, **fisheye (Kannala–Brandt)**,
**omnidirectional (Mei)**, **double sphere**, and **EUCM**, each with its full
distortion catalogue — plus a Ceres-based PnP / intrinsic-calibration layer
and ROS 2 / OpenCV interfaces built on the same model set.

Eigen is the only hard dependency of the runtime core. Everything else —
Ceres, OpenCV, ROS 2, OpenMP — is optional and detected at configure time.

## Why camxiom

Supporting several camera geometries usually multiplies code: per-model
projection paths (`cv::projectPoints` vs `cv::fisheye::*` vs
`cv::omnidir::*`), per-model undistortion, per-model calibration tooling —
and some models (double sphere, EUCM) have no mainstream tooling at all.
camxiom collapses that into one model set handled uniformly everywhere:

- **One API for every model.** The primary API is two functions —
  `rayToPixel()` and `pixelToRay()`. Forward projection, undistortion,
  rectification, Jacobians, PnP, and calibration behave identically across
  the five projection families; adding a camera model never grows the API
  surface, and supporting a new camera downstream never means a new code
  path.
- **A calibrator with the same breadth.** The calibration layer
  (`camxiom::calib`) provides initial-guess estimators per model family, a
  model-agnostic PnP solver (single templated cost, analytical Jacobians, a
  Ceres-free Gauss–Newton fast path for pose-only solves) with per-parameter
  freeze flags for staged optimization, a single-pass intrinsics calibrator
  with uncertainty and observability diagnostics, and cross-family model
  conversion.
- **Fits existing ecosystems.** `sensor_msgs/CameraInfo` adapters, `cv::Mat`
  drop-ins mirroring OpenCV signatures, OpenCV-Calib3D / Kalibr / Basalt
  parameter profiles, ROS-layout calibration YAML.
- **Light by default.** The runtime core (`camxiom::core`) links only Eigen —
  no OpenCV types in the core, no Ceres header installed. The calibration
  layer is built only when Ceres is found (or requested).

## Feature overview

| Area | What you get |
|---|---|
| [Projection](api/projection.md) | Generic `rayToPixel` / `pixelToRay` over all five models, float32 and float64 variants, `ValidatedCameraModel` for validated hot loops, Ceres-`Jet`-ready projection templates |
| [Batch projection](api/batch.md) | Bulk ray↔pixel APIs (Eigen and raw-pointer overloads) with SIMD (AVX2 / SSE2 / NEON) and OpenMP acceleration |
| [Jacobians](api/jacobians.md) | Analytical 2×3 point Jacobians, batch variants, and full parameter Jacobians (intrinsics / distortion / projection parameters) for optimization |
| [Inverse LUT](api/lut.md) | Precomputed grid + bilinear interpolation for O(1) approximate `pixelToRay` |
| [Remap & rectification](api/remap.md) | OpenCV-free undistort / rectify pipeline: virtual output-camera generation, `alpha`-blended fitting compatible with OpenCV semantics, remap-map builders, CPU sampling kernel |
| [CameraInfo interchange](api/camera-info.md) | ROS-free `camxiom::CameraInfo` POD, `makeCameraModel` string parsing, per-model import/export profiles (OpenCV-Calib3D / Kalibr / Basalt), `camera_calibration_parsers`-layout YAML writer |
| [Calibration](api/calibration.md) | Default seeds, linear + model-specific init estimators, model-agnostic `PnpSolver` with `FIX_*` staged optimization, single-pass `calibrate()` with covariance and observability diagnostics |
| [Model conversion](api/conversion.md) | `convertCameraModel`: fit a model of a different family to reproduce a source model's geometry, with honest fit-quality reporting |
| [ROS / OpenCV interop](api/interop.md) | Optional `sensor_msgs::msg::CameraInfo` conversion layer and `cv::Mat` drop-in layer (`cv::` / `cv::fisheye::` / `cv::omnidir::`-style signatures, `RemapCache`, unified `solvePnP`) |

## Architecture at a glance

camxiom ships as one package with three CMake targets, split on a
*runtime-geometry vs calibration* boundary:

```
camxiom::camxiom  (INTERFACE umbrella — link this for everything)
├── camxiom::core   runtime geometry: types, model, projection, jacobian,
│                   lut, batch, SIMD, remap, CameraInfo compat
│                   depends on Eigen3 only (+ optional OpenMP / ROS / OpenCV interop)
└── camxiom::calib  calibration: default seeds, init estimators, PnP solver,
                    intrinsics calibrator, model conversion
                    depends on camxiom::core + Ceres (built only with CAMXIOM_WITH_CERES)
```

The package builds in **dual mode**: inside a ROS 2 / colcon workspace it
registers with the ament index; outside, plain CMake generates a standalone
`camxiomConfig.cmake`. Both paths export the same `camxiom::` targets, so
downstream code always uses `find_package(camxiom)`.

See [Architecture](design/architecture.md) for the full design.

## Documentation map

- **[Getting started](getting-started.md)** — install, build (colcon and pure
  CMake), link, first projection.
- **[Camera models](camera-models.md)** — coordinate conventions, the
  `CameraModel` data model, all projection and distortion models with their
  exact parameter layouts.
- **API reference** — one page per subsystem (see the feature table above).
- **Design** — [architecture](design/architecture.md),
  [error handling](design/error-handling.md),
  [numerical strategy](design/numerical-strategy.md),
  [thread safety](design/thread-safety.md).

## Status and license

- Version **0.1.0** (pre-1.0: minor releases may contain breaking changes;
  they are listed in the README's versioning notes).
- License: **Apache-2.0**.
- Platforms exercised regularly: x86-64 (SSE2 baseline + AVX2) and
  aarch64 / NEON (NVIDIA Jetson).
