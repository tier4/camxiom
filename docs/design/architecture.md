# Architecture

## Layer model

camxiom is one package with three CMake targets, split on a **runtime
geometry vs calibration** boundary — a semantic split, deliberately *not*
"does it happen to include Ceres":

| Target | Alias | Depends on | Contents |
|---|---|---|---|
| `camxiom_core` | `camxiom::core` | **Eigen3 only** (+ optional OpenMP; optional ROS / OpenCV interop) | types, model parsing/validation, generic + per-model projection (float & double), Jacobians, LUT, batch + SIMD kernels, remap/rectify, CameraInfo compat + YAML |
| `camxiom_calib` | `camxiom::calib` | `camxiom::core` + **Ceres** (PUBLIC) | default seeds, all init estimators, PnP optimizer, `calibrate()`, `convertCameraModel()`. Built only when `CAMXIOM_WITH_CERES`. |
| `camxiom` | `camxiom::camxiom` | whatever was built | INTERFACE umbrella: core + calib when present. The backward-compatible "link everything" target. |

Two boundary decisions worth spelling out:

- The **linear** init estimators (homography, Zhang, pinhole-OpenCV) are
  Eigen-only, yet they live in `calib`, not `core`. The boundary is drawn by
  *meaning* (they are calibration-time tools), keeping core's contract crisp:
  "runtime ray ↔ pixel geometry".
- The ROS and OpenCV **interop layers compile into `core`** (they are
  Ceres-free adapters), guarded by `CAMXIOM_HAS_ROS` / `CAMXIOM_HAS_OPENCV`
  and compiled only when the dependency is found.

## Dependency policy

- **OpenCV-free core.** The library reasons in camera-frame rays and its own
  `CameraModel`; no `cv::` type appears outside the opt-in interop layer.
  Rectification kernels, initial-guess estimation, and remap utilities are
  implemented natively.
- **Ceres optional, and never in installed headers.** Consumers that only
  project / undistort / remap must not pull in an optimizer. Beyond the
  target split, `PnpSolver` hides Ceres behind a PIMPL: the public header
  exposes plain structs (`PnpOptimizerOptions`, `PnpSummary`), all Ceres
  state lives in the `.cpp`, and internal cost-function headers are not
  installed. **No installed header includes Ceres** — including any camxiom
  header never requires Ceres; only *linking* `camxiom::calib` does (which
  is exactly what the package config's `find_dependency(Ceres)` expresses).
  The Ceres floor is 2.0 (the 2.0 vs 2.1+ Manifold API difference is
  absorbed at the single call site).
- **ROS optional.** The canonical interchange type is the ROS-free
  `camxiom::CameraInfo` POD; `ros.hpp` only copies message fields into it.
  Calibration YAML files are written by the core without ROS.
- **No exceptions across the API.** Fallible operations return `StatusCode`
  or result structs; see [Error handling](error-handling.md).

## Public API design

- **Two-function primary API.** `rayToPixel` / `pixelToRay` (+ `*64`)
  subsume what OpenCV spreads across `projectPoints`, `undistortPoints`,
  `fisheye::*`, `omnidir::*`. Per-model entry points
  (`camxiom::pinhole::rayToPixel` and siblings) are **internal** — declared
  once in a private header and used only by dispatch, batch, and SIMD
  layers. Adding a model changes dispatch tables, not the API.
- **Composition over inheritance.** `CameraModel` =
  intrinsics ∘ distortion ∘ projection, three plain aggregates in a
  fixed-size (~200 B), stack-allocatable, vtable-free container. Hot paths
  use a single `switch` per call (hoisted out of loops in batch), never
  virtual dispatch, never per-point heap allocation. Derived state (tilt
  matrices, distortion flags, `theta_max`) is precomputed at model build.
- **Forward-only model authoring.** A new model implements forward
  projection (+ Jacobians); the inverse comes from shared numerical solvers
  (see [Numerical strategy](numerical-strategy.md)).
- **Single-source float/double.** All types and per-model math are `T`
  templates instantiated exactly twice; the float and double APIs cannot
  drift apart by construction. Batch SIMD is the one deliberate exception
  (per-precision kernels — different optimization targets).

## Header layout

```
include/camxiom/
├── core.hpp  calibration.hpp  camxiom.hpp     # layer umbrellas (pure aggregation)
├── types.hpp / types64.hpp                    # data model (float / double aliases)
├── model.hpp  model_compare.hpp  validated_model.hpp  version.hpp
├── projection.hpp / projection64.hpp / projection_template.hpp
├── jacobian*.hpp  lut*.hpp  batch*.hpp
├── remap.hpp  remap_kernel.hpp
├── compat.hpp  camera_info_yaml.hpp           # ROS-free interchange
├── ros.hpp  opencv.hpp  opencv/pnp.hpp        # optional interop (self-guarded)
├── default_seed.hpp  init/*.hpp               # calibration layer
├── optimizer/pnp_flag.hpp  optimizer/pnp_solver.hpp
├── calib/intrinsics.hpp  calib/convert.hpp
└── internal/constants.hpp                     # installed but NOT part of the stable API
```

`internal/` is installed (public headers reference it in default member
initializers) but carries **no compatibility guarantee** and is excluded
from the public-API snapshot test.

## Dual-mode packaging

One `CMakeLists.txt` serves both worlds; `find_package(ament_cmake QUIET)`
picks the branch:

- **ament mode** (ROS 2 / colcon): registers with the ament index,
  `ament_export_targets` / `ament_export_dependencies`.
- **pure CMake mode**: generates `camxiomConfig.cmake` +
  `camxiomConfigVersion.cmake` (SameMajorVersion) +
  `camxiomTargets.cmake`. The config `find_dependency()`s only what was
  actually built (Eigen3 always; OpenMP / Ceres / OpenCV / sensor_msgs
  conditionally, with the Ceres version pinned to what camxiom was compiled
  against).

Both paths export the same `camxiom::core` / `camxiom::calib` /
`camxiom::camxiom` targets, so downstream usage is identical. A
configure-time check keeps `project(VERSION)`, `version.hpp`, and
`package.xml` in lock-step (drift is a configure error).

## Quality gates

- GoogleTest suites per subsystem: core geometry (types, projection smoke,
  batch↔scalar parity, error paths, property/fuzz invariants, remap, LUT,
  Jacobians, compat, YAML, validated model, umbrella-header include-safety)
  plus calibration suites (per-estimator, solver, uncertainty, conversion,
  integration with real-correspondence fixtures). Interop and calibration
  tests are presence-gated, so every build configuration tests exactly what
  it built.
- A **public-API snapshot test** (Python, stdlib-only) diffs the public
  symbol surface of `include/camxiom/**` against a golden file, so
  accidental API changes fail CI.
- CI builds SSE2-baseline, AVX2, and ASan+UBSan variants on x86-64;
  scalar ↔ SIMD parity tests run the real vector kernels; the NEON path is
  validated natively on aarch64.
- Benchmarks (`CAMXIOM_BUILD_BENCHMARKS=ON`) baseline the projection hot
  path and the calibrate-diagnostics overhead.
