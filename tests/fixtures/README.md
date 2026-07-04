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

## Short worked example (2 views, 6 points each — minimum valid fixture)

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
file is found the test is **skipped** (not failed); the skip message prints
the directory that was searched.

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
- The calibration seed is always `getDefaultSeed(model, w, h)`. For pinhole
  this is a cold start (fx = h/2); for fisheye models projection/distortion
  params are locked at their seed values (D29/D30).
- The expected_* fields are optional. Omit them if you only want a smoke test
  (status == OK) without regression assertions on intrinsic values.
