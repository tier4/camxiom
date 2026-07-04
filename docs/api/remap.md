# Remap & rectification

An OpenCV-free pipeline from a `CameraModel` to an undistorted / rectified
image:

```
CameraModel ──► (1) design a virtual output camera        makeRectifiedOutputModel / alpha fit
            ──► (2) build remap maps (map_x, map_y)       buildRemapMap family
            ──► (3) sample pixels                         remapImage kernel  (or cv::remap via the OpenCV layer)
```

Headers: `<camxiom/remap.hpp>` (map builders, FOV, output-camera
generation), `<camxiom/remap_kernel.hpp>` (CPU sampling kernel). The
optional [OpenCV layer](interop.md) wraps the same builders in `cv::Mat`
form (`RemapCache`, `undistortImage`, ...).

## Map semantics

All builders fill two `float` buffers `map_x` / `map_y` of `width × height`
(row-major, index `i = v·width + u`), defined so that

```
dst(v, u) = src( map_y[i], map_x[i] )
```

i.e. for each destination pixel: `dst_pixel → pixelToRay(dst_model) →
rayToPixel(src_model)`. Invalid destination pixels (outside the source
model's domain or out of source bounds) are written as the sentinel `-1`,
which the sampling kernel turns into the fill value. `status == OK` means
the map build succeeded even if some pixels are sentinels.

## Basic builders

```cpp
RemapResult buildRemapMap(const CameraModel &src_model, const CameraModel &dst_model,
                          int width, int height, float *map_x, float *map_y,
                          const SolverOptions &solver_options = {});

RemapResult buildUndistortRemapMap(const CameraModel &src_model,
                                   int width, int height, float *map_x, float *map_y,
                                   const SolverOptions &solver_options = {});
```

- `buildRemapMap` is the general model-to-model reprojection.
- `buildUndistortRemapMap` is the **same-projection undistort**: removes
  distortion but keeps the projection model. For non-pinhole models
  (fisheye, omnidirectional, DS, EUCM) this does **not** produce a
  rectilinear image — use `buildRectifyRemapMap` for that.
- `RemapResult{ status, valid_count, total_count }`.

## Image-aware builder (bounds + rotation)

```cpp
ImageRemapResult buildImageRemapMap(const CameraModel &src_model, ImageSize src_size,
                                    const CameraModel &dst_model, ImageSize dst_size,
                                    float *map_x, float *map_y,
                                    const ImageRemapOptions &options = {});
```

Adds to the basic builder: an optional rotation between destination and
source frames (`options.src_from_dst_rotation`, with
`ray_in_source = R · ray_in_dst` — the hook for stereo rectification),
source-bounds checking (`require_source_in_bounds`), a policy for invalid
pixels (`WRITE_NEGATIVE_ONE` sentinel vs `WRITE_RAW_COORDINATE`), and
separate diagnostics (`model_valid_count` vs `source_in_bounds_count`).
`buildUndistortImageRemapMap` is the same-size undistort convenience.

## Field of view

```cpp
FovResult computeFov(const CameraModel &model, ImageSize image_size,
                     const SolverOptions &solver_options = {});
```

Horizontal / vertical / diagonal FOV in degrees, measured between the
unprojected rays of boundary pixels — works for any model. A boundary pixel
that cannot be unprojected reports that single component as `0.0` with
overall status `OK`.

## Rectification

### Virtual output camera

```cpp
RectifiedOutputModelResult makeRectifiedOutputModel(
    const CameraModel &src_model, ImageSize src_size,
    const RectifiedOutputModelOptions &options = {});
```

Designs a distortion-free virtual camera for the rectified output.
`RectifiedOutputModelOptions`:

| Field | Default | Meaning |
|---|---|---|
| `projection_type` | `PINHOLE` | output projection: `PINHOLE`, `CYLINDRICAL`, `STEREOGRAPHIC`, `LONGITUDE_LATITUDE` |
| `output_size` | source size | output image size |
| `fit_policy` | `MAX_INSCRIBED_VALID` | `MAX_INSCRIBED_VALID` (largest all-valid region — OpenCV alpha=0), `ALL_SOURCE_CONTAINED` (widest FOV — alpha=1), `ALLOW_INVALID_BORDER` (manual `focal_scale`) |
| `focal_scale` | 1.0 | manual focal multiplier for `ALLOW_INVALID_BORDER` |
| `src_from_output_rotation` | identity | stereo-rectification pre-rotation |
| `boundary_sample_count` | 2048 | boundary sampling density for the fit search |
| `source_margin_px` | 1.0 | safety margin at the source border |

For `PINHOLE` the generated model is a pure pinhole (`distortion NONE`,
`skew 0`, `fx/fy` fitted independently, principal point at the pixel-center
image middle `((W−1)/2, (H−1)/2)`). The non-pinhole projections
(cylindrical, stereographic — usable to roughly 220° FOV, longitude/latitude
— full sphere) cannot be expressed as a `CameraModel`; they are available
through `buildRectifyRemapMap`, which leaves `output_model` as `UNKNOWN` for
them.

### One-call rectify map

```cpp
RectifyRemapResult buildRectifyRemapMap(const CameraModel &src_model, ImageSize src_size,
                                        const RectifiedOutputModelOptions &options,
                                        float *map_x, float *map_y);
```

`makeRectifiedOutputModel()` + `buildImageRemapMap()` in one call; the result
carries the generated `output_model`, output size, and achieved FOV.

### Alpha-blended rectify (OpenCV-compatible semantics)

```cpp
BuildRectifyMapResult buildRectifyMap(const CameraModel &src_model,
                                      ImageSize src_size, ImageSize dst_size,
                                      float alpha, float *map_x, float *map_y,
                                      const BuildRectifyMapOptions &options = {});
```

One `alpha ∈ [0, 1]` covers what OpenCV splits into pinhole `alpha`
(`getOptimalNewCameraMatrix`) and fisheye `balance`:

- `alpha = 0` — output contains only valid pixels (inscribed fit);
- `alpha = 1` — output keeps as much source FOV as a pinhole target can
  represent, with black borders; rays at ≥ (π/2 − margin) are skipped
  because no rectilinear target can represent them (so for >180° sources,
  alpha=1 intentionally does not contain literally every pixel);
- intermediate — the focal length is interpolated linearly between the two
  extremes (matching OpenCV's rectangle interpolation).

Works for any source model. Out-of-range alpha is clamped and echoed back as
`alpha_used`. `fx` and `fy` are fitted independently (OpenCV-compatible).

## Sampling kernel

```cpp
#include <camxiom/remap_kernel.hpp>

template <typename PixelT>              // explicit instantiations: std::uint8_t, float
RemapImageResult remapImage(const PixelT *src, ImageSize src_size,
                            const float *map_x, const float *map_y,
                            PixelT *dst, ImageSize dst_size,
                            InterpolationMode mode,   // NEAREST or BILINEAR
                            PixelT fill_value);
```

Applies precomputed maps to a single-channel image buffer: sentinel or
out-of-bounds source positions produce `fill_value` (counted in
`border_count`); `BILINEAR` interpolates 4 taps in float (rounded for
`uint8_t`). Multi-channel images can be processed plane-by-plane, or via
`cv::remap` with the [OpenCV interop layer](interop.md), which shares the
same map builders.
