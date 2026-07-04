# ROS / OpenCV interop

Two optional adapter layers compile into the core library when their
dependency is present at configure time. Both are thin by design: all
parsing, packing, and math stay in the dependency-free core; the adapters
only copy fields.

| Layer | Header(s) | Guard |
|---|---|---|
| ROS 2 | `<camxiom/ros.hpp>` | `CAMXIOM_HAS_ROS` **and** `__has_include(<sensor_msgs/msg/camera_info.hpp>)` |
| OpenCV | `<camxiom/opencv.hpp>`, `<camxiom/opencv/pnp.hpp>` | `CAMXIOM_HAS_OPENCV` **and** `__has_include(<opencv2/core.hpp>)` |

When the guard is not satisfied the headers expand to nothing, so including
them unconditionally is always safe. The `CAMXIOM_HAS_*` macros are PUBLIC
compile definitions on the exported CMake targets; non-CMake consumers of an
interop-enabled build must define them manually. (Both the macro *and* the
include check are required — `__has_include` alone would expose declarations
with no compiled implementation and turn every use into a link error.)

## ROS layer (`camxiom/ros.hpp`)

```cpp
// The one-liner most nodes need:
CameraModel makeCameraModel(const sensor_msgs::msg::CameraInfo &camera_info);

// Strict per-profile message conversion (same profiles as compat.hpp):
StatusCode importPinholeCameraInfo(const sensor_msgs::msg::CameraInfo &, PinholeCompatProfile, CameraModel &out);
StatusCode exportPinholeCameraInfo(const CameraModel &, PinholeCompatProfile, sensor_msgs::msg::CameraInfo &out);
// ... Fisheye / Omnidirectional / DoubleSphere / Eucm variants
```

Semantics worth knowing:

- **Import folds binning/ROI into K/P** (the `image_geometry` adjustment),
  so the resulting model maps the *published* image's pixels, not the full
  sensor's.
- **Export resets `binning_x/binning_y/roi`** to "no binning / full image" —
  the exported K/P describe the model's own pixel frame, and stale metadata
  would otherwise be applied a second time by the next importer.
  `width`/`height` are left to the caller.
- Everything else delegates to the core POD path
  ([CameraInfo interchange](camera-info.md)); no parsing logic is duplicated
  in the ROS layer.

## OpenCV layer (`camxiom/opencv.hpp`)

Namespace `camxiom::opencv`. The API mirrors OpenCV's helpers so it can act
as a drop-in replacement, but **every model is handled by the unified
camxiom core** — no OpenCV math is used for projection or undistortion.

### `RemapCache` — compute maps once, apply per frame

```cpp
camxiom::opencv::RemapCache cache;
auto rr = cache.buildRectify(src_model, {width, height}, options);   // or build / buildUndistort
for (;;) {
  cache.apply(frame, rectified /*, interpolation, border_mode, border_value */);
}
```

`build*` wrap the core [remap builders](remap.md) and store `cv::Mat`
maps; `apply` runs `cv::remap`. Accessors: `isValid()`, `width()`,
`height()`, `map1()`, `map2()`, `clear()`.

### One-shot helpers and map builders

```cpp
RemapResult        buildRemapMapCV(src_model, dst_model, w, h, map1, map2, solver_options);
RemapResult        buildUndistortRemapMapCV(src_model, w, h, map1, map2, solver_options);
RectifyRemapResult buildRectifyRemapMapCV(src_model, src_size, options, map1, map2);

bool undistortImage(src, dst, src_model, ...);   // same-projection undistort
bool rectifyImage(src, dst, src_model, options, output_model_out, ...);
bool remapImage(src, dst, src_model, dst_model, ...);
bool distortImage(src, dst, src_model, dst_model, ...);   // inverse mapping — argument order matters

bool initCameraMatrix2D(object_points_per_view, image_points_per_view, image_size, model_out);

// Point operations (batch, SIMD/OpenMP-backed; return success count or -1):
int projectPoints(model, object_points, rvec, tvec, image_points_out);
int undistortPoints(model, src, dst, R = {}, P = {}, solver_options = {});
int distortPoints(model, src, dst);
```

### Drop-in signature namespaces

For code being migrated from OpenCV with `cv::Mat camera_matrix /
dist_coeffs` arguments instead of `CameraModel`:

- `camxiom::opencv::pinhole::*` — mirrors `cv::` calib3d
  (`projectPoints`, `undistortPoints`, `undistortImage`, ...)
- `camxiom::opencv::fisheye::*` — mirrors `cv::fisheye::*`
- `camxiom::opencv::omnidirectional::*` — mirrors `cv::omnidir::*`
  (each function takes the extra `double xi`)

### Unified PnP (`camxiom/opencv/pnp.hpp`)

```cpp
struct SolvePnPConfig {
  int method{6};            // cv::SOLVEPNP_IPPE — best for planar targets
  int fallback_method{0};   // cv::SOLVEPNP_ITERATIVE
  bool use_fallback{true};
};

bool solvePnP(const CameraModel &model,
              const std::vector<cv::Point3f> &object_points,
              const std::vector<cv::Point2f> &image_points,
              cv::Vec3d &rvec_out, cv::Vec3d &tvec_out,
              const SolvePnPConfig &config = {}, const SolverOptions &solver_options = {});

double reprojectRmse(const CameraModel &model, object_points, image_points, rvec, tvec);
```

Works with **all** camxiom models: observed pixels are normalized through
camxiom's `undistortPoints`, then `cv::solvePnP` runs with an identity
camera matrix. `reprojectRmse` averages over successfully projected points
only, and returns infinity when nothing projects.

For a PnP without OpenCV at all, use
[`init::estimatePoseRefined`](calibration.md#init-estimators-camxiominit)
from the calibration layer instead.
