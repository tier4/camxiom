# camxiom 
### ***camxiom [ˈkæmdʒiəm] — For camera geometry thirsted for an absolute axiom.***

**Unified C++17 camera-geometry library** — one API across the whole
spectrum of camera models, from narrow-FOV pinhole to 180°+ fisheye and
catadioptric: **pinhole, fisheye (Kannala–Brandt), omnidirectional (Mei),
double sphere, and EUCM**, each with its full distortion catalogue.
On top of the same model set: a Ceres-based intrinsic calibrator and
ROS 2 / OpenCV interfaces.

- **Every camera model, one API.** The primary API is two functions —
  `rayToPixel()` / `pixelToRay()`. Projection, undistortion, rectification,
  Jacobians, PnP, and calibration all behave identically across the five
  projection families; supporting a new camera in your application never
  means a new code path.
- **A calibrator with the same breadth.** The PnP / intrinsics solver is
  model-agnostic (single templated cost, analytical Jacobians, a Ceres-free
  Gauss–Newton fast path for pose-only solves), with per-parameter `FIX_*`
  flags for staged optimization, initial-guess estimators per model family,
  covariance + observability diagnostics, and cross-family model conversion.
- **Fits the ecosystems you already use.** `sensor_msgs/CameraInfo`
  adapters, `cv::Mat` drop-ins mirroring `cv::` / `cv::fisheye::` /
  `cv::omnidir::` signatures, OpenCV-Calib3D / Kalibr / Basalt parameter
  profiles, and ROS-layout calibration YAML.
- **Light by default.** The runtime core depends only on Eigen; Ceres,
  OpenCV, ROS 2, and OpenMP are optional layers detected at configure time.
  Ament and pure-CMake packaging from the same tree.

License: [Apache-2.0](LICENSE) · Version: 0.1.0 ·
Docs: [`docs/`](docs/index.md) (MkDocs / GitHub Pages ready)

---

## Supported camera models

| Projection | Distortion | Reference |
|---|---|---|
| Pinhole | none / radtan4 / radtan5 (plumb_bob) / rational8 / thin-prism12 / tilted14 (Scheimpflug) | OpenCV-compatible coefficients |
| Fisheye (θ-based) | OpenCV-fisheye4, KB4, KB8, ideal equidistant / equisolid / stereographic / orthographic | Kannala & Brandt 2006 |
| Omnidirectional | ξ + optional radtan | Mei & Rives 2007 |
| Double sphere | ξ, α + optional radtan | Usenko et al. 2018 |
| EUCM | α, β + optional radtan | Khomutenko et al. 2015 |

All of them share the same `CameraModel` aggregate, the same generic API,
the same batch / Jacobian / LUT / remap machinery, and the same calibrator.

## Features

| | |
|---|---|
| **Unified projection** | Generic `rayToPixel` / `pixelToRay` over all five models; float32 (real-time) and float64 (calibration) variants; `ValidatedCameraModel` for validate-once hot loops; Ceres-`Jet`-ready projection templates for AutoDiff cost functors |
| **Calibration** *(optional, Ceres)* | Default seeds, linear + per-model init estimators, model-agnostic `PnpSolver` with per-parameter `FIX_*` flags for staged optimization, single-pass `calibrate()` with parameter-covariance and **observability diagnostics** (catches "low RMS but unobservable focal") |
| **Model conversion** *(optional, Ceres)* | `convertCameraModel` refits across model families (e.g. KB4 → pinhole+radtan5) with honest fit-quality reporting |
| **Bulk APIs** | Batch projection / Jacobians with SIMD (AVX2 · SSE2 · NEON) + OpenMP; `InverseLut` for O(1) approximate unprojection |
| **Jacobians** | Analytical 2×3 point Jacobians and full parameter Jacobians (intrinsics / distortion / ξ·α·β) for optimization |
| **Remap & rectify** | Undistort/rectify for every model: virtual output-camera generation, OpenCV-compatible `alpha` semantics (pinhole `alpha` and fisheye `balance` unified), map builders, CPU sampling kernel — no OpenCV required |
| **Interchange** | ROS-free `camxiom::CameraInfo` POD; `makeCameraModel` parses ROS / OpenCV distortion-model strings; per-model import/export profiles (OpenCV-Calib3D, Kalibr, Basalt); `camera_calibration_parsers`-layout YAML writer |
| **Interop** *(optional)* | `sensor_msgs::msg::CameraInfo` adapters (binning/ROI folded correctly); `cv::Mat` drop-ins, `RemapCache`, unified `solvePnP` working with all models |

## Quick start

```cpp
#include <camxiom/core.hpp>     // Eigen-only runtime geometry, one include

// Build a model from any ROS/OpenCV-style camera description (no ROS needed):
camxiom::CameraInfo info;
info.width = 1280;  info.height = 960;
info.k = {800.0, 0.0, 639.5,  0.0, 800.0, 479.5,  0.0, 0.0, 1.0};
info.distortion_model = "plumb_bob";   // or "equidistant", "double_sphere", "eucm", ...
info.d = {-0.28, 0.07, 0.0001, -0.0002, 0.0};
camxiom::CameraModel model = camxiom::makeCameraModel(info);

// Forward: 3D ray (camera frame, +Z forward) -> pixel
auto px = camxiom::rayToPixel(model, Eigen::Vector3f(0.1f, 0.05f, 1.0f));
if (px.ok()) { /* px.pixel.u, px.pixel.v */ }

// Inverse: pixel -> unit ray
auto ray = camxiom::pixelToRay(model, {640.0f, 480.0f});
if (ray.ok()) { /* ray.ray.direction */ }
```

The same two calls work unchanged whether `model` describes a pinhole, a
KB4 fisheye, or a double-sphere camera. In a ROS 2 node,
`#include <camxiom/ros.hpp>` adds
`makeCameraModel(const sensor_msgs::msg::CameraInfo&)` directly.

Every fallible call returns a `StatusCode` (`OK`, `BEHIND_CAMERA`,
`OUT_OF_FOV`, `NON_CONVERGED`, ...) — no exceptions, no silent NaNs.

## Build

```bash
# ROS 2 / colcon workspace
colcon build --packages-select camxiom

# Standalone (no ROS)
cmake -S camxiom -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && cmake --install build
```

Consume from CMake (identical in both modes):

```cmake
find_package(camxiom REQUIRED)
target_link_libraries(my_app PRIVATE camxiom::camxiom)   # or camxiom::core / camxiom::calib
```

| Dependency | Min | Role |
|---|---|---|
| Eigen3 | 3.3 | **required** — sole dependency of `camxiom::core` |
| Ceres | 2.0 | optional — enables `camxiom::calib` (`CAMXIOM_WITH_CERES`, auto-detected; Ubuntu 22.04 / JetPack apt package works) |
| OpenCV / sensor_msgs / OpenMP | any | optional — interop layers & parallelism, auto-detected |

See [Getting started](docs/getting-started.md) for details (CMake options,
pure-CMake packaging, tests, benchmarks).

## Documentation

Full documentation lives in [`docs/`](docs/index.md) and is
[MkDocs](mkdocs.yml)-ready for GitHub Pages (`mkdocs serve` to preview):

- [Getting started](docs/getting-started.md) — install, build, link, first projection
- [Camera models](docs/camera-models.md) — conventions and exact parameter layouts
- API reference — [projection](docs/api/projection.md) ·
  [batch](docs/api/batch.md) · [Jacobians](docs/api/jacobians.md) ·
  [inverse LUT](docs/api/lut.md) · [remap & rectification](docs/api/remap.md) ·
  [CameraInfo interchange](docs/api/camera-info.md) ·
  [calibration](docs/api/calibration.md) ·
  [model conversion](docs/api/conversion.md) ·
  [ROS / OpenCV interop](docs/api/interop.md)
- Design — [architecture](docs/design/architecture.md) ·
  [error handling](docs/design/error-handling.md) ·
  [numerical strategy](docs/design/numerical-strategy.md) ·
  [thread safety](docs/design/thread-safety.md)

## Non-goals

- **No OpenCV linkage in the core.** The runtime core is OpenCV-free by
  design: rectification kernels, initial-guess estimation, and remap
  utilities are implemented natively. The OpenCV layer is an optional
  convenience, never a requirement.
- **No board detection.** Chessboard / dot / AprilTag detection belongs to
  the application layer.
- **No calibration strategy.** `calibrate()` is one maximum-likelihood pass;
  view selection, outlier rejection, staged unlocking, and restarts are the
  application's policy decisions (camxiom provides the flags and diagnostics
  to implement them).

## Versioning

`<camxiom/version.hpp>` defines `CAMXIOM_VERSION_{MAJOR,MINOR,PATCH,STRING}`
and a numerically comparable `CAMXIOM_VERSION_CODE`
(`CAMXIOM_MAKE_VERSION(major, minor, patch)`), kept in lock-step with
`project(VERSION)` and `package.xml` by a configure-time check.

Pre-1.0 caveat: minor releases may contain breaking changes. Notable ones
made while still pre-0.1.0, for anyone migrating from older snapshots:
per-model projection namespaces (`camxiom::pinhole::rayToPixel` etc.) were
removed from the public headers — use the generic `rayToPixel` /
`pixelToRay`; result types are `[[nodiscard]]`; `toString(DistortionModelType)`
now round-trips through `parseDistortionModelType` (some spellings changed);
and the never-returned `StatusCode` enumerators (`NOT_IMPLEMENTED`,
`JACOBIAN_SINGULAR`) were dropped.

## Thread safety

Free functions are stateless and safe on shared models from any thread.
Stateful objects (`InverseLut`, `RemapCache`, `PnpSolver`) follow
"one writer, then many readers" per instance. Details:
[thread safety](docs/design/thread-safety.md).

## For developers

Code style is enforced by `clang-format` (config:
[`.clang-format`](.clang-format), Google-based, 100-column) through
[pre-commit](https://pre-commit.com/). One-time setup:

```bash
pip install pre-commit        # or: pipx install pre-commit
pre-commit install            # from the repo root — hooks then run on every `git commit`
```

Requires `clang-format` (LLVM 14+) on PATH
(`sudo apt install clang-format` on Ubuntu).

Usage:

```bash
pre-commit run --all-files    # format the whole tree now
pre-commit run                # format staged files only
```

The hook rewrites files in place: if a commit is rejected, the offending
files have already been reformatted — review the changes, `git add` them,
and commit again.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (commit conventions, test
expectations). The public API surface is guarded by a snapshot test —
intentional API changes must update the golden file.

## License

[Apache License 2.0](LICENSE)
