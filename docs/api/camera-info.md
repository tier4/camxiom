# CameraInfo interchange

How camera descriptions enter and leave camxiom. The canonical interchange
type is a **ROS-free POD** — every conversion (ROS messages, OpenCV K/D
pairs, Kalibr / Basalt conventions, YAML files) funnels through it, so the
parsing and packing logic exists exactly once, in the dependency-free core.

Headers: `<camxiom/compat.hpp>`, `<camxiom/camera_info_yaml.hpp>` (both
core, no ROS / no OpenCV). The ROS message adapters live in
[`<camxiom/ros.hpp>`](interop.md).

## `camxiom::CameraInfo` — the interchange POD

```cpp
struct CameraInfo {
  std::array<double, 9>  k;               // 3x3 intrinsic matrix, row-major (default identity)
  std::vector<double>    d;               // distortion coefficients (model-dependent length)
  std::string            distortion_model;
  std::array<double, 9>  r;               // rectification matrix (default identity)
  std::array<double, 12> p;               // 3x4 projection matrix, row-major
  std::uint32_t          width{0}, height{0};
};
```

Mirrors the field layout of `sensor_msgs/CameraInfo` without the ROS-only
fields (`header`, `binning_*`, `roi` — the [ROS layer](interop.md) folds
those into K/P before producing this POD).

## `makeCameraModel` — the single parsing entry point

```cpp
CameraModel makeCameraModel(const CameraInfo &camera_info);
```

Builds a `CameraModel` from K / D / `distortion_model`. Parsing is
case-insensitive and accepts both camxiom's canonical names and the common
ROS / OpenCV / third-party spellings:

The name selects both slots of the model — the projection law and the
correction on top of it (see
[Projection law and lens correction](../camera-models.md#projection-law-and-lens-correction)):

| Input string(s) | Projection law | Distortion |
|---|---|---|
| `none` (or empty string + empty `d`) | `PINHOLE` | `NONE` |
| `radtan4` | `PINHOLE` | `RADTAN4` |
| `radtan5`, `plumb_bob`, `brown_conrady` | `PINHOLE` | `RADTAN5` |
| `rational8`, `rational_polynomial` | `PINHOLE` | `RATIONAL8` |
| `thin_prism12`, `thin_prism_fisheye` | `PINHOLE` | `THIN_PRISM12` |
| `tilted14`, `tilted` | `PINHOLE` | `TILTED14` |
| `kb4`, `kannala_brandt`, `kannala-brandt`, `equidistant`, `fisheye`, `opencv_fisheye4` | `EQUIDISTANT` | `KB4` |
| `kb8`, `kannala_brandt8`, `kannala-brandt8` | `EQUIDISTANT` | `KB8` |
| `ideal_equidistant` | `EQUIDISTANT` | `NONE` |
| `equisolid[_angle]`, `stereographic[_angle]`, `orthographic[_angle]` | `EQUISOLID`, `STEREOGRAPHIC`, `ORTHOGRAPHIC` | `NONE` |
| `omnidirectional`, `omni` | `OMNIDIRECTIONAL`; `d = [xi, plane coeffs...]` | by the remaining count |
| `double_sphere`, `ds` | `DOUBLE_SPHERE`; `d = [xi, alpha, plane coeffs...]` | by the remaining count |
| `eucm` | `EUCM`; `d = [alpha, beta, plane coeffs...]` | by the remaining count |
| anything else | `UNKNOWN` (rejected by `validateCameraModel`) | — |

The laws without a correction take no coefficients: an all-zero `d` (as
some writers pad it) is read as none, and any non-zero entry yields a model
that `validateCameraModel` rejects.

Two pitfalls worth knowing:

- ROS's `equidistant` means the **4-coefficient fisheye polynomial**
  (OpenCV's fisheye model, which is the same function as KB4), not the
  plain equidistant law. camxiom reads it as `EQUIDISTANT` + `KB4`, writes
  `KB4` back under that name, and reads `ideal_equidistant` as the plain
  law (`NONE`). ROS has no name for the plain law, so the fisheye exporters
  (`CANONICAL` included) write it under `equidistant` with an empty `D`,
  which `makeCameraModel` reads back as a KB4 model without coefficients
  and `validateCameraModel` rejects. Write `ideal_equidistant` yourself to
  keep the plain law in a file camxiom reads back.
- For Mei / DS / EUCM, the projection-level parameters ride in the *front*
  of `d` (see the table). They are unpacked into
  `CameraModel::projection`, not kept as distortion coefficients.

`makeCameraModel` never throws; a description it cannot understand yields a
model that `validateCameraModel` rejects. Always validate after parsing.

## Strict per-profile import / export

For interchange with specific ecosystems, `compat.hpp` defines profile enums
and symmetric import/export functions per model family, returning a
`StatusCode`. Unlike the lenient `makeCameraModel`, the import functions
**enforce the profile's exact coefficient count** and validate the result.

| Family | External struct | Profiles |
|---|---|---|
| pinhole | `PinholeExternalModel{K, D}` | `CANONICAL`, `OPENCV_CALIB3D_D4/D5/D8/D12/D14`, `ROS_CAMERA_INFO_RAW` |
| fisheye | `FisheyeExternalModel{K, D, distortion_type, projection_type}` | `CANONICAL`, `OPENCV_FISHEYE_D4`, `ROS_CAMERA_INFO_EQUIDISTANT` |
| omnidirectional | `OmnidirectionalExternalModel{K, D, xi}` | `CANONICAL`, `OPENCV_OMNIDIR_D4/D5/D8` |
| double sphere | `DoubleSphereExternalModel{K, D, xi, alpha}` | `CANONICAL`, `BASALT_D0`, `BASALT_D4` |
| EUCM | `EucmExternalModel{K, D, alpha, beta}` | `CANONICAL`, `KALIBR_D0`, `KALIBR_D4` |

```cpp
StatusCode importPinholeModel(const PinholeExternalModel &, PinholeCompatProfile, CameraModel &out);
StatusCode exportPinholeModel(const CameraModel &, PinholeCompatProfile, PinholeExternalModel &out);
// ... importFisheyeModel / exportFisheyeModel, importOmnidirectionalModel / ...,
//     importDoubleSphereModel / ..., importEucmModel / ...
```

In the external structs, `xi` / `alpha` / `beta` are **named fields** and `D`
carries only the plane/angle coefficients — the prefix packing only happens
in the `CameraInfo` POD / ROS-message direction. `FisheyeExternalModel`
names the fisheye law in `projection_type` (default `EQUIDISTANT`) and the
correction in `distortion_type` (`NONE` with an empty `D`, `KB4`, `KB8`;
`UNKNOWN` reads `D` as KB4); the `CANONICAL` profile reads both, the OpenCV
and ROS profiles always mean `EQUIDISTANT` + `KB4`. `toString()` overloads
exist for every profile enum.

### What export does not check

Export packs the model into the profile's layout; it does not check that
the profile can represent the camera:

- A profile with a fixed coefficient count (`OPENCV_CALIB3D_D*`,
  `OPENCV_FISHEYE_D4`, `OPENCV_OMNIDIR_D*`, `BASALT_D0/D4`, `KALIBR_D0/D4`)
  zero-fills the coefficients the model lacks and drops the ones past the
  count, non-zero or not: `RATIONAL8` under `OPENCV_CALIB3D_D5` keeps
  `k1, k2, p1, p2, k3`, `KB8` under `OPENCV_FISHEYE_D4` keeps `k1..k4`, and
  `BASALT_D0` / `KALIBR_D0` drop the plane correction.
- The OpenCV and ROS fisheye profiles write any fisheye law as
  `equidistant` (KB4), so an equisolid, stereographic or orthographic model
  comes back as the equidistant law. `ROS_CAMERA_INFO_EQUIDISTANT` refuses
  `KB8` (`INVALID_MODEL`).
- `CANONICAL` and `ROS_CAMERA_INFO_RAW` write every coefficient the model
  has.

To know whether an export kept the camera, import the result again and
compare it with the source model.

### Exporting to a `CameraInfo` POD

```cpp
StatusCode exportPinholeCameraInfo(const CameraModel &, PinholeCompatProfile, CameraInfo &out);
// ... exportFisheyeCameraInfo, exportOmnidirectionalCameraInfo,
//     exportDoubleSphereCameraInfo, exportEucmCameraInfo
```

Packs K/D per the profile, sets the family's canonical
`distortion_model` string, `r = I`, `p = [K | 0]`. `width`/`height` are left
untouched (the caller owns image geometry). `exportOmnidirectionalCameraInfo`
supports only `CANONICAL` (the OpenCV omnidir profiles return
`INVALID_MODEL` here; `exportOmnidirectionalModel` still writes them). The
reverse direction for a raw POD is simply `makeCameraModel`.

## YAML output (`camera_calibration_parsers` layout)

```cpp
[[nodiscard]] std::string toCameraInfoYaml(const CameraInfo &info,
                                           const std::string &camera_name = "camera");
[[nodiscard]] StatusCode  saveCameraInfoYaml(const std::string &path,
                                             const CameraInfo &info,
                                             const std::string &camera_name = "camera");
```

Serializes the POD in the standard ROS calibration-file layout —
`image_width`, `image_height`, `camera_name`, `camera_matrix`,
`distortion_model`, `distortion_coefficients`, `rectification_matrix`,
`projection_matrix` — accepted by `camera_calibration_parsers` and camera
drivers' `camera_info_url`, **without camxiom depending on ROS** (the writer
is plain C++ in the core).

Formatting is deterministic and locale-independent (matrices at 6 decimals
in aligned blocks, distortion coefficients at 12 decimals on one line;
numbers always use `.`). Strings are quoted/escaped only when not
identifier-like, so the output is always valid YAML. `saveCameraInfoYaml`
overwrites `path` and returns `INVALID_INPUT` when the file cannot be
opened or fully written.

The writer emits what it is given — it does not re-validate K/D consistency
(use `validateCameraModel(makeCameraModel(info))` for that).

## Round-trip picture

```
              lenient, string-driven                strict, profile-driven
  CameraInfo ───── makeCameraModel ────► CameraModel ── export*Model ──► {K, D, xi, ...}
      ▲                                        │                      (OpenCV / Kalibr / Basalt)
      │                                        │
      └────── export*CameraInfo(profile) ◄─────┘        import*Model ──► CameraModel
      │
      ├── toCameraInfoYaml / saveCameraInfoYaml         (ROS-free calibration YAML)
      └── ros.hpp: sensor_msgs ⇄ CameraInfo POD         (optional layer, binning/ROI folded)
```

### Saving and reloading in double precision

`makeCameraModel64` and `import*Model64` / `export*Model64` /
`export*CameraInfo64` pass intrinsics, distortion coefficients and projection
parameters through without rounding them to float. To keep that precision in
YAML, pass `CameraInfoYamlPrecision::ROUND_TRIP` after the camera name.
Without it, the default format applies: 6 decimal places for the matrices
and 12 for the distortion coefficients.

```cpp
camxiom::CameraInfo info;
auto status = camxiom::exportFisheyeCameraInfo64(
    fit.camera_model, camxiom::FisheyeCompatProfile::CANONICAL, info);
if (status == camxiom::StatusCode::OK) {
  status = camxiom::saveCameraInfoYaml(
      path, info, "front_camera", camxiom::CameraInfoYamlPrecision::ROUND_TRIP);
}
```

`ROUND_TRIP` writes finite doubles in scientific notation with 17 significant
digits, so negative zero and subnormal values survive as well. To reload,
read the numbers into a `CameraInfo` as doubles with an existing YAML reader
and pass it to `makeCameraModel64`; camxiom itself adds no YAML reader
dependency. `theta_max` is not part of the CameraInfo format and is not
saved, so keep any application-specific field-of-view limit separately.
