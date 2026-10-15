# camxiom real-correspondence fixture format

## Why correspondences, not images

camxiom is OpenCV-free and cannot decode raw image files. A "real-image
regression fixture" is therefore a detected-correspondence text file: the
world-to-image point pairs that a real camera and detector produced, written
as plain numbers. Raw-image to corner detection is the application layer's
responsibility (MS4-new, out of scope for camxiom).

## Grammar

Lines starting with `#` and blank lines are ignored. All other lines are
parsed token by token, whitespace-separated. The following directives must
appear in this order (with the view blocks at the end):

```
# Model names are lowercase (matching camxiom::toString output); uppercase also accepted.
model: <pinhole|fisheye_theta|omnidirectional|double_sphere|eucm>
image_size: <w> <h>

# optional: pinhole distortion family to fit from zero coefficients
pinhole_distortion: <none|radtan5|rational8>
# optional: acceptance threshold for training and held-out reprojection RMS
max_rms_px: <positive float>
# optional: reserve the last N views for evaluation, leaving >= 3 training views
held_out_views: <non-negative integer>

# optional: present only when you want the test to assert recovery
expected_fx: <float>
expected_fy: <float>
expected_cx: <float>
expected_cy: <float>

view: <N>
<wx> <wy> <wz> <u> <v>
... (exactly N rows)

view: <N>
<wx> <wy> <wz> <u> <v>
... (exactly N rows)
```

Constraints enforced by the parser:
- At least 3 views required.
- Each view must have at least 6 points.
- Declared `N` must match the actual number of rows that follow.
- Unknown model strings are rejected with a descriptive error.
- Non-numeric tokens in numeric fields are rejected.

## Short worked example (2 views, 6 points each — add a third view before use)

Model names match the output of `camxiom::toString(ProjectionModelType)` (lowercase).
Both lowercase and uppercase are accepted by the parser.

```
# Example fixture — fisheye camera, 640x480
model: fisheye_theta
image_size: 640 480

# Intrinsic regression expectations
expected_fx: 200.0
expected_fy: 200.0
expected_cx: 320.0
expected_cy: 240.0

# View 0: board front-on at 30 cm
view: 6
0.00 0.00 0.0  298.5  218.3
0.05 0.00 0.0  309.7  218.3
0.10 0.00 0.0  320.8  218.3
0.00 0.05 0.0  298.5  229.4
0.05 0.05 0.0  309.7  229.4
0.10 0.05 0.0  320.8  229.4

# View 1: board tilted 20 degrees about X
view: 6
0.00 0.00 0.0  302.1  210.5
0.05 0.00 0.0  313.2  211.0
0.10 0.00 0.0  324.3  211.5
0.00 0.05 0.0  302.3  221.7
0.05 0.05 0.0  313.4  222.2
0.10 0.05 0.0  324.5  222.8
```

## How to export from the application/detector (MS4-new)

After detecting corners with your detector (chessboard, dotboard, AprilTag),
for each accepted frame write:

```
view: <N>
```

followed by one `<wx> <wy> <wz> <u> <v>` line per matched world-image pair.
World points are in metres (Z = 0 for a flat board). Image points are in
pixels (origin at top-left, u right, v down).

Use at least 15 significant digits for floating-point values to avoid
quantisation noise in the regression check.

## Discovery rules

The test `Ms2IntegrationFixture/RealCorrespondenceRegression` searches for
`*.fixture` files as follows:

1. If `CAMXIOM_TEST_FIXTURE_DIR` is set and non-empty, that directory is used.
2. Otherwise the directory containing this README
   (`tests/fixtures/` inside the camxiom source tree) is used.

Files are processed in lexicographic order (deterministic). If no `*.fixture`
file is found the test **fails**, so removing the bundled data cannot silently
disable recorded-data coverage. Every fit must retain all correspondence
projections and have finite RMS. When `held_out_views` is set, those last
views do not enter the camera fit; only their poses are estimated under the
recovered camera. Global and per-view held-out RMS must meet `max_rms_px`.

## Bundled c2_120 recorded correspondences

`c2_120.fixture` is taken from the corner export of the intrinsic camera
calibrator in tier4/CalibrationToolkit,
`calibrators/intrinsic_camera_calibrator/tests/c2_120/detected_corners.csv`
(branch `refactor/intrinsic_camera_calibrator/architecture-overhaul`). The
CSV holds 847 samples of 48 detected chessboard corners each, from recorded
images; the images are not bundled. The data is recorded, not synthesised
by camxiom.

- Source CSV SHA-256:
  `c743965461d7a58b44a9ab1d5c7b3026674e66ba4d60f1e44ce06c5ac8395754`.
- Select 24 sample indices deterministically as `floor(i * 846 / 23)`,
  `i=0..23`; the exact indices are also recorded in the fixture header.
- Preserve the CSV's numeric strings in the order
  `object_x object_y object_z image_u image_v`. Image paths are omitted.
- Image size and reference intrinsics come from the adjacent
  `camera_info_gt.yaml`: 2880x1860, pinhole with Rational8 distortion.
  These are reference calibration values, not independently measured exact
  ground truth. The seed uses default intrinsics and zero distortion; it
  never reads the reference values used for assertions.
- Fit the first 20 selected views and evaluate the last four separately.
  The 0.5 px RMS bound applies to the training fit, the held-out aggregate,
  and every held-out view. The existing 5% focal / 10 px principal-point
  checks also apply against the reference calibration.

This fixture covers one recorded pinhole/Rational8 camera. Recorded fisheye,
Mei, Double Sphere and EUCM data still need to be added.

## Provisional regression tolerances

Until real fixture data is available with known noise characteristics the
following provisional scaffold tolerances apply:

| Parameter | Tolerance | Type |
|-----------|-----------|------|
| fx, fy    | 5 %       | relative |
| cx, cy    | 10 px     | absolute |

Tighten these literals in `integration_test.cpp` (the `RealCorrespondenceRegression`
test) once you have real fixtures and can characterise the noise floor.

## Notes

- All five projection models are supported: `PINHOLE`, `FISHEYE_THETA`,
  `OMNIDIRECTIONAL`, `DOUBLE_SPHERE`, `EUCM`.
- The calibration seed starts from `getDefaultSeed(model, w, h)`, with the
  optional `pinhole_distortion` family and zero coefficients. For pinhole
  this is a cold start (fx = h/2); for fisheye models projection/distortion
  params are locked at their seed values (D29/D30).
- The expected_* fields are optional. Omit them if you only want a smoke test
  (status == OK) without regression assertions on intrinsic values.
