# Getting started

## Requirements

| Dependency | Minimum | Required? | Enables |
|---|---|---|---|
| C++17 compiler, CMake ≥ 3.14 | — | **required** | — |
| Eigen3 | 3.3 | **required** | the runtime core (`camxiom::core`) — its only dependency. apt: `libeigen3-dev` |
| Ceres | 2.0 | optional | the calibration layer (`camxiom::calib`). apt: `libceres-dev` — the 2.0.0 shipped by Ubuntu 22.04 / JetPack works (the 2.0 vs 2.1+ Manifold API difference is absorbed inside the solver implementation). |
| OpenCV | any | optional | the `cv::Mat` interop layer (`opencv.hpp`, `opencv/pnp.hpp`) |
| sensor_msgs (ROS 2) | any | optional | the ROS interop layer (`ros.hpp`) |
| OpenMP | — | optional | parallel batch projection / Jacobian / LUT / remap-map builds |

Optional dependencies are **presence-driven**: they are auto-detected at
configure time and the matching layer is compiled in only when found. The one
user-facing switch is `CAMXIOM_WITH_CERES` (defaults to ON when Ceres is
found; forcing it ON without Ceres is a configure error).

## Building

### In a ROS 2 / colcon workspace (ament mode)

```bash
cd your_ws
colcon build --packages-select camxiom
```

The package registers with the ament index; downstream ROS 2 packages depend
on it the usual way (`<depend>camxiom</depend>` + `find_package(camxiom REQUIRED)`).

### Standalone (pure CMake mode)

No ROS required. The same `CMakeLists.txt` detects the absence of
`ament_cmake` and installs a standalone CMake package config:

```bash
cmake -S camxiom -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/camxiom
cmake --build build -j
cmake --install build
```

To build the runtime core only (no Ceres, no calibration layer):

```bash
cmake -S camxiom -B build -DCMAKE_BUILD_TYPE=Release -DCAMXIOM_WITH_CERES=OFF
```

### CMake options

| Option | Default | Effect |
|---|---|---|
| `CAMXIOM_WITH_CERES` | auto (`ON` when Ceres ≥ 2.0 is found) | Build `camxiom::calib` and add it to the umbrella target. `OFF` gives a core-only build. |
| `CAMXIOM_BUILD_BENCHMARKS` | `OFF` | Build the projection benchmark (and the calibrate benchmark when Ceres is enabled). Use with `CMAKE_BUILD_TYPE=Release`. |
| `BUILD_TESTING` | (CTest default; colcon sets it ON) | Build the GoogleTest suite. In standalone mode tests additionally require a findable GTest package. |

## Linking

Both build modes export identical targets, so consumption is always:

```cmake
find_package(camxiom REQUIRED)

# Everything that was built (core + calib when available):
target_link_libraries(my_app PRIVATE camxiom::camxiom)

# Or pick a layer explicitly:
target_link_libraries(my_runtime PRIVATE camxiom::core)    # Eigen-only runtime geometry
target_link_libraries(my_calib   PRIVATE camxiom::calib)   # calibration (needs a Ceres-enabled build)
```

The generated package config re-resolves the public dependencies for you
(`find_dependency` for Eigen3, and — only when the corresponding layer was
built — OpenMP, Ceres, OpenCV, sensor_msgs). Consumers do not need to find
those packages themselves.

When a ROS- or OpenCV-enabled camxiom build is consumed, the exported targets
propagate the `CAMXIOM_HAS_ROS` / `CAMXIOM_HAS_OPENCV` compile definitions
automatically. Non-CMake consumers must define these macros themselves to see
the interop declarations.

## First projection

```cpp
#include <camxiom/core.hpp>   // one include for all runtime geometry

int main() {
  // Describe the camera. CameraModel is a plain aggregate; you can fill it
  // by hand, or parse it from a CameraInfo description (see below).
  camxiom::CameraModel model{};
  model.intrinsics = {800.0f, 800.0f, 639.5f, 479.5f, 0.0f};  // fx, fy, cx, cy, skew
  model.projection.type = camxiom::ProjectionModelType::PINHOLE;

  // Forward: 3D ray (camera frame, +Z forward) -> pixel
  auto px = camxiom::rayToPixel(model, Eigen::Vector3f(0.1f, 0.05f, 1.0f));
  if (px.ok()) {
    // px.pixel.u, px.pixel.v
  }

  // Inverse: pixel -> unit ray direction
  auto ray = camxiom::pixelToRay(model, {640.0f, 480.0f});
  if (ray.ok()) {
    // ray.ray.direction  (Eigen::Vector3f, unit length)
  }
}
```

Every result carries a `StatusCode` (`OK`, `BEHIND_CAMERA`, `OUT_OF_FOV`,
`NON_CONVERGED`, ...) instead of silently producing NaNs — see
[Error handling](design/error-handling.md).

### From a camera-info description

The usual way to obtain a `CameraModel` is the string-driven factory, which
understands ROS / OpenCV distortion-model names (`plumb_bob`, `equidistant`,
`rational_polynomial`, `double_sphere`, `eucm`, ...):

```cpp
#include <camxiom/compat.hpp>

camxiom::CameraInfo info;             // ROS-free POD, mirrors sensor_msgs/CameraInfo
info.width = 1280; info.height = 960;
info.k = {800.0, 0.0, 639.5,  0.0, 800.0, 479.5,  0.0, 0.0, 1.0};
info.distortion_model = "plumb_bob";
info.d = {-0.28, 0.07, 0.0001, -0.0002, 0.0};

camxiom::CameraModel model = camxiom::makeCameraModel(info);
if (camxiom::validateCameraModel(model) != camxiom::StatusCode::OK) { /* reject */ }
```

In a ROS 2 node, `#include <camxiom/ros.hpp>` adds the overload
`makeCameraModel(const sensor_msgs::msg::CameraInfo&)` (it also folds
binning/ROI into K correctly). See
[CameraInfo interchange](api/camera-info.md).

### A first calibration

```cpp
#include <camxiom/calibration.hpp>

using camxiom::calib::CalibrationView;

std::vector<CalibrationView> views = /* detected corners: world (Z=0) + pixels */;

camxiom::CameraModel seed = camxiom::getDefaultSeed(
    camxiom::ProjectionModelType::FISHEYE_THETA, 1280, 960);

camxiom::calib::CalibrationOptions opts;
opts.image_width = 1280;
opts.image_height = 960;

// Per-model identifiability locks are the caller's responsibility:
// KB4 fisheye -> fix k3/k4; MEI/DS/EUCM -> fix projection params; pinhole -> NONE.
auto result = camxiom::calib::calibrate(
    views, seed,
    camxiom::calib::PnpFlag::FIX_DIST_2 | camxiom::calib::PnpFlag::FIX_DIST_3,
    opts);

if (result.ok()) {
  // result.camera_model, result.rms_reprojection_error_px,
  // result.parameter_std / observability diagnostics, ...
}
```

See [Calibration](api/calibration.md) for the full pipeline (seeds, init
estimators, staged optimization, diagnostics).

## Convenience headers

Individual headers are always available; three umbrella headers aggregate the
API by layer:

| Header | Pulls in |
|---|---|
| `<camxiom/core.hpp>` | the Eigen-only runtime geometry layer (types, model, projection, jacobian, lut, batch, remap, compat). No Ceres / OpenCV / ROS. |
| `<camxiom/calibration.hpp>` | the calibration layer (`default_seed`, `init/*`, `optimizer/*`, `calib/intrinsics`, `calib/convert`). Ceres-free to *include*; *linking* needs a Ceres-enabled build. |
| `<camxiom/camxiom.hpp>` | everything: `core.hpp` + `calibration.hpp` + the optional ROS / OpenCV interop (self-guarded, harmless when absent). |

## Running the tests and benchmarks

```bash
# colcon
colcon test --packages-select camxiom && colcon test-result --verbose

# standalone (requires a findable GTest)
ctest --test-dir build
```

Benchmarks (numbers are only meaningful in Release):

```bash
cmake -S camxiom -B build-bench -DCMAKE_BUILD_TYPE=Release -DCAMXIOM_BUILD_BENCHMARKS=ON
cmake --build build-bench -j
./build-bench/camxiom_projection_benchmark
```

## Next steps

- [Camera models](camera-models.md) — the data model and every supported
  projection / distortion model.
- [Projection API](api/projection.md) — the full single-point API surface.
- [Architecture](design/architecture.md) — layering, packaging, and the
  dependency policy behind it.
