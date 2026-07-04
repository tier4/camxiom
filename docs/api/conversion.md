# Model conversion

`calib::convertCameraModel` fits a camera model of a (typically different)
family so it reproduces a source model's pixel ↔ ray geometry — e.g. KB4
fisheye → pinhole+radtan5 for a consumer that only understands plumb_bob,
pinhole → double sphere, MEI → EUCM.

Header: `<camxiom/calib/convert.hpp>` (calibration layer — needs a
Ceres-enabled build).

## API

```cpp
struct ModelConversionOptions {
  int grid_cols{24}, grid_rows{18};   // synthetic-correspondence grid over the source image
  int max_iterations{200};
};

struct [[nodiscard]] ModelConversionResult {
  StatusCode status;                  // OK = the fit CONVERGED (not a quality certificate!)
  CameraModel camera_model;           // fitted model, same family as dst_seed
  double rms_fit_error_px;            // residual vs the source geometry
  double max_fit_error_px;
  int used_point_count;               // grid points the source could unproject
  int representable_point_count;      // of those, how many the fitted model can reproject
  bool ok() const; explicit operator bool() const;
};

[[nodiscard]] ModelConversionResult convertCameraModel(
    const CameraModel &src_model, int image_width, int image_height,
    const CameraModel &dst_seed,
    const ModelConversionOptions &options = {});
```

## How it works

A grid of source pixels is unprojected through `src_model` into unit rays,
and the destination model is fitted (single synthetic view, pose fixed to
identity, ordinary least squares — synthetic correspondences have no
outliers) to project those rays back onto the same pixels. Source pixels
outside the source FOV are skipped automatically.

`dst_seed` selects the destination family **and** provides the initial
guess — pass `getDefaultSeed(type, width, height)` for the standard warm
start, or a hand-tuned model when converting between wildly different
geometries. The fitted model's `theta_max` is refreshed from its final
coefficients.

## Reading the result honestly

`status == OK` means the optimizer converged — it does **not** certify that
the destination family can represent the source geometry. Converting a 190°
fisheye to a pinhole converges to the best pinhole there is, with a large
residual. Judge quality from:

- `rms_fit_error_px` / `max_fit_error_px` — the geometric error of the
  conversion itself (exact pairs drive this to numerical precision);
- `representable_point_count` vs `used_point_count` — a large gap means part
  of the source FOV is simply outside what the destination family can
  express.

`NON_CONVERGED` carries the best-effort model; `INVALID_INPUT` /
`INVALID_MODEL` / `DEGENERATE_CONFIG` (fewer than 8 usable grid
correspondences) report bad inputs.

```cpp
auto seed = camxiom::getDefaultSeed(camxiom::ProjectionModelType::PINHOLE, w, h);
auto conv = camxiom::calib::convertCameraModel(fisheye_model, w, h, seed);
if (conv.ok() && conv.rms_fit_error_px < 0.5 &&
    conv.representable_point_count == conv.used_point_count) {
  usePinhole(conv.camera_model);
} else {
  // the conversion is lossy for this camera — decide explicitly what to do
}
```
