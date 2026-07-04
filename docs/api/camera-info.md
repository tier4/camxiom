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

| Input string(s) | Distortion model | Projection family |
|---|---|---|
| `none` (or empty string + empty `d`) | `NONE` | pinhole |
| `radtan4` | `RADTAN4` | pinhole |
| `radtan5`, `plumb_bob`, `brown_conrady` | `RADTAN5` | pinhole |
| `rational8`, `rational_polynomial` | `RATIONAL8` | pinhole |
| `thin_prism12`, `thin_prism_fisheye` | `THIN_PRISM12` | pinhole |
| `tilted14`, `tilted` | `TILTED14` | pinhole |
| `equidistant`, `fisheye`, `opencv_fisheye4` | `OPENCV_FISHEYE4` | fisheye |
| `kb4`, `kannala_brandt`, `kannala-brandt` | `KB4` | fisheye |
| `kb8`, `kannala_brandt8`, `kannala-brandt8` | `KB8` | fisheye |
| `ideal_equidistant` | `EQUIDISTANT` (coefficient-free) | fisheye |
| `equisolid[_angle]`, `stereographic[_angle]`, `orthographic[_angle]` | trigonometric fisheye variants | fisheye |
| `omnidirectional`, `omni` | — | Mei; `d = [xi, plane coeffs...]` |
| `double_sphere`, `ds` | — | double sphere; `d = [xi, alpha, plane coeffs...]` |
| `eucm` | — | EUCM; `d = [alpha, beta, plane coeffs...]` |
| anything else | `UNKNOWN` (rejected by `validateCameraModel`) | — |

Two pitfalls worth knowing:

- ROS's `equidistant` means the **4-coefficient OpenCV fisheye**, not the
  ideal equidistant mapping — camxiom follows ROS here and provides
  `ideal_equidistant` for the coefficient-free model.
- For Mei / DS / EUCM, the projection-level parameters ride in the *front*
  of `d` (see the table). They are unpacked into
  `CameraModel::projection`, not kept as distortion coefficients.

`makeCameraModel` never throws; a description it cannot understand yields a
model that `validateCameraModel` rejects. Always validate after parsing.

## Strict per-profile import / export

For interchange with specific ecosystems, `compat.hpp` defines profile enums
and symmetric import/export functions per model family. Unlike the lenient
`makeCameraModel`, these **enforce the profile's exact coefficient count**
and validate the result, returning a `StatusCode`.

| Family | External struct | Profiles |
|---|---|---|
| pinhole | `PinholeExternalModel{K, D}` | `CANONICAL`, `OPENCV_CALIB3D_D4/D5/D8/D12/D14`, `ROS_CAMERA_INFO_RAW` |
| fisheye | `FisheyeExternalModel{K, D, distortion_type}` | `CANONICAL`, `OPENCV_FISHEYE_D4`, `ROS_CAMERA_INFO_EQUIDISTANT` |
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
in the `CameraInfo` POD / ROS-message direction. `toString()` overloads
exist for every profile enum.

### Exporting to a `CameraInfo` POD

```cpp
StatusCode exportPinholeCameraInfo(const CameraModel &, PinholeCompatProfile, CameraInfo &out);
// ... exportFisheyeCameraInfo, exportOmnidirectionalCameraInfo,
//     exportDoubleSphereCameraInfo, exportEucmCameraInfo
```

Packs K/D per the profile, sets the family's canonical
`distortion_model` string, `r = I`, `p = [K | 0]`. `width`/`height` are left
untouched (the caller owns image geometry). The reverse direction for a raw
POD is simply `makeCameraModel`.

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
