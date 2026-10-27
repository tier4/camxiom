# Remap & rectification

An OpenCV-free pipeline from a `CameraModel` to an undistorted / rectified
image:

```
CameraModel ──► (1) design a virtual output camera        makeRectifiedOutputModel (template + alpha)
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
  rectilinear image — use `buildRectifyRemapMap` with a pinhole output
  template for that.
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

Horizontal / vertical / diagonal FOV in degrees.

- Pinhole, fisheye laws, omnidirectional, double sphere, EUCM: the angle
  between the unprojected rays of two boundary pixels — `(0, cy)`..`(W−1, cy)`,
  `(cx, 0)`..`(cx, H−1)` and the two opposite corners. A boundary pixel that
  cannot be unprojected reports that single component as `0.0` with overall
  status `OK`.
- `EQUIRECTANGULAR`, `CYLINDRICAL`: angular spans of valid pixels, edge to
  edge and clipped to the sphere, since the angle between two rays cannot
  exceed 180°. Horizontal is the longitude span of the principal row (from
  the left edge of column 0, `u = −0.5`, to the right edge of the last
  column); vertical is the latitude span of the principal column `u = cx`,
  top edge to bottom edge. With skew that column's longitude changes from
  row to row, and rows where it passes the seam are not counted. A
  whole-sphere equirectangular image reports 360 × 180. `diagonal_fov_deg`
  is 0: these projections have no diagonal angle.

## Rectification

Rectification fits a distortion-free output camera to a source and builds
the map into it. The output can be any projection: you describe it with a
**template** camera, and camxiom chooses its scale.

### Output camera

```cpp
RectifiedOutputModelResult makeRectifiedOutputModel(
    const CameraModel &src_model, ImageSize src_size,
    const RectifiedOutputModelOptions &options = {});
```

`RectifiedOutputModelOptions`:

| Field | Default | Meaning |
|---|---|---|
| `output_template` | square-pixel `PINHOLE` (`fx = fy = 1`) | projection and pixel shape of the output; distortion must be `NONE` |
| `output_size` | source size | output image size |
| `alpha` | 0 | 0 = no invalid output pixel, 1 = the whole source kept; must be in `[0, 1]` |
| `src_from_output_rotation` | identity | `ray_in_source = R · ray_in_output` (stereo-rectification pre-rotation, or pointing a latitude-longitude output) |
| `source_margin_px` | 1.0 | margin the `alpha = 0` view keeps clear of the source image edge (`alpha = 1` keeps the source out to its outermost pixel centres regardless) |
| `solver_options` | default | for the `pixelToRay` calls of the fit and the map build |

The output camera is the template scaled by **one** factor `s`:

- its projection is kept — type, `xi` / `alpha` / `beta`, and `theta_max`
  (a fisheye-law template shows at most `theta_max` off axis; the
  `ProjectionModel` default is π/2);
- its pixel shape is kept — `fx`, `fy` and `skew` are all multiplied by `s`,
  so `fx : fy : skew` is the template's (square pixels by default);
- the principal point is the image centre `((W−1)/2, (H−1)/2)`; the
  template's `cx`, `cy` are not used.

`alpha` chooses `s`:

- `alpha = 0` — the widest view in which every output pixel shows source
  content (no invalid border): every ray the output shows between its
  outermost pixel centres is a valid source direction (see below) and
  lands at least `source_margin_px` inside the source image edge. For an
  equirectangular source that wraps all the way around, the view also
  stays off the source's seam, the image edge where longitude wraps (remap
  does not wrap, so a view across it would sample the gap);
- `alpha = 1` — the narrowest view in which every valid source ray that
  lands between the source's outermost pixel centres, and that the output
  can represent, lands inside the output image (an invalid border where
  the source does not reach);
- in between — `s` interpolates linearly between the two.

**Horizon margin.** Some outputs have coordinates that grow without bound
toward an edge of what they can show: the pinhole 90° horizon, the
stereographic antipode, the cylinder's poles, the end of the Mei /
double-sphere / EUCM domain. The equidistant and equisolid laws stay
finite at their antipode, but their coordinates are singular there: they
draw it as a whole circle, every direction landing on it, so a pixel there
has no single ray. The fit keeps 0.05 rad (about 3°) from those places in
two cases only:

- `alpha = 1` leaves source rays inside the margin out — near an unbounded
  edge, containing them would push `s` to zero — so those source pixels
  may fall outside the output. A pinhole output of a source wider than
  180° therefore contains the source up to about 3° short of the horizon.
- `alpha = 0` goes as close to the edge as the source allows (a 178°
  pinhole source rectifies to itself, and so does one reaching within
  2e-7 rad of its horizon). Only when the source covers the output all the
  way to the edge — e.g. a fisheye whose whole image circle lies inside the
  frame, rectified to pinhole — is there no widest view; the view then
  stops at the margin.

**Valid source rays and the fold.** A distortion polynomial can fold over
inside the frame: with a strong negative radtan `k1`, the distorted radius
peaks and turns back before the image corners. Past that fold a pixel
unprojects onto the folded branch, to a ray the lens does not see at that
pixel (e.g. a left-edge pixel returning a ray far to the right of the
axis). A source ray is valid when it lies before the first fold: along its
direction out from the source axis, the forward projection must still
succeed and keep the orientation it has near the axis. The fold is found
from the distortion polynomials themselves — the Jacobian determinant of a
plane distortion, its rational denominator and the tilt's third
component, or the slope of the Kannala–Brandt angle polynomial — with
certified Bernstein bounds, not by stepping out from the axis, so even a
fold that turns back for a fraction of a degree counts. A root where a
condition only touches zero (a double root, e.g. `k1 = −10, k2 = 45`)
cannot be told from a crossing one; the rays past it count as possibly
invalid, so `alpha = 1` keeps them and `alpha = 0` stops before them. For a
lens without a fold every ray the model projects is valid, out to the
edge of its domain.

**How the fit decides.** Both ends are decided over the continuous sets of
rays, not on pixel or other samples. With `q` the output's coordinates of
a ray at `s = 1`, a ray lands inside the output exactly when `s · |q|_K ≤ 1`
(`|q|_K` the larger of its offsets from the image centre in units of half
the image size), so

- `alpha = 1` takes `s = 1 / max |q|_K` over the valid source rays landing
  in the source image that the output keeps, and
- `alpha = 0` takes `s = 1 / min |q|_K` over the rays the output can show
  that are not valid or land outside the source image (or within the
  margin of its edge), and keeps the output's corners inside its own
  domain.

Both extremes come from a branch-and-bound search over patches of
directions: interval arithmetic encloses each patch's source pixel, its
validity and its `|q|_K`; patches that cannot hold the extreme are dropped
and the rest split, until the bound is within 2e-4 of a value an actual ray
attains. The bound itself is returned, with 1e-4 more of headroom. A
patch the search cannot settle — at a double root, or once it has spent its
work budget — keeps the bound it has, which is less tight.

**Scope of the guarantee.** It holds for the model formulas evaluated in
double precision, up to floating-point rounding: the 1e-4 of headroom
covers that rounding, and the float evaluation of the public projections,
for ordinary models. Models near singular parameters — e.g. a Mei model
with `xi` close to −1, where the float and double evaluations of its
projection differ greatly — are outside it. For `alpha = 0` the output's
own domain is also checked with the returned model itself: its corners
must project. `alpha = 1` makes no such check — keeping every valid source
ray can need output corners past the output's limit (e.g. beyond the poles
of an equirectangular output).

On `OK`, `output_model` is a valid `CameraModel` of the template's
projection (the latitude-longitude projections included), the `*_fov_deg` fields
are `computeFov(output_model, output_size)`, and `estimated_*` count a coarse
grid over the output. `INVALID_MODEL` means an invalid source model;
`INVALID_INPUT` an invalid source size, a source or output image less than
two pixels wide or high (no span between outermost pixel centres), an
`alpha` outside `[0, 1]`, or a
template that is not a valid model with distortion `NONE`; `DOMAIN_ERROR`
that the fit has no solution: for `alpha < 1`, no output view shows the
source everywhere (e.g. the rotation points the output away from the
source, or the output is centred on the seam of an equirectangular source
that wraps around); for `alpha > 0`, no source ray is representable.

Examples:

```cpp
// A square-pixel pinhole with no invalid pixels (the defaults):
camxiom::RectifiedOutputModelOptions perspective;
perspective.output_size = {1280, 720};

// A distortion-free equidistant fisheye up to 97 degrees off axis that
// keeps the whole source:
camxiom::RectifiedOutputModelOptions fisheye;
fisheye.output_template.projection.type = camxiom::ProjectionModelType::EQUIDISTANT;
fisheye.output_template.projection.theta_max = 1.7f;
fisheye.alpha = 1.0f;

// A cylindrical image of the source:
camxiom::RectifiedOutputModelOptions cylinder;
cylinder.output_template.projection.type = camxiom::ProjectionModelType::CYLINDRICAL;
cylinder.output_size = {2048, 768};
cylinder.alpha = 1.0f;
```

A whole-sphere equirectangular image needs no fit: build it with
`makeEquirectangularModel(size, model)` and call `buildImageRemapMap`.

### One-call rectify map

```cpp
RectifyRemapResult buildRectifyRemapMap(const CameraModel &src_model, ImageSize src_size,
                                        const RectifiedOutputModelOptions &options,
                                        float *map_x, float *map_y);
```

`makeRectifiedOutputModel()` + `buildImageRemapMap()` (with
`options.src_from_output_rotation` and source-bounds checking) in one call.
The result carries the fitted `output_model`, the output size, and its FOV.
The map is exactly what `buildImageRemapMap` builds from `output_model`, so
the model can be stored and the map rebuilt later.

### Compared with OpenCV

One `alpha` covers what OpenCV splits into pinhole `alpha`
(`getOptimalNewCameraMatrix`) and fisheye `balance`, with the same meaning
at 0 and 1 for every source and every output projection. Unlike OpenCV, the
output keeps the template's pixel shape: OpenCV fits `fx` and `fy`
separately, which stretches the image when the source's horizontal and
vertical extents differ. The results are not meant to equal OpenCV's.

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
