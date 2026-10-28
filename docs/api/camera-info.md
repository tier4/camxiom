# CameraInfo interchange

How camera descriptions enter and leave camxiom. There are two formats:

- **ROS CameraInfo** (`camxiom::CameraInfo`): for ROS and every tool that reads
  `sensor_msgs/CameraInfo` or its calibration YAML. It can hold only the
  lenses ROS has distortion models for, and camxiom writes a model to it only
  when it holds the same lens.
- **camxiom's own description** (`camxiom::CamxiomCameraInfo`): holds every
  valid `CameraModel`, field by field, and reads back as the same model.

Header: `<camxiom/compat.hpp>` (core, no ROS / no OpenCV). The ROS message
adapters live in [`<camxiom/ros.hpp>`](interop.md). camxiom does not read or
write files: storing a description is left to the application.

## ROS CameraInfo

```cpp
struct CameraInfo {
  std::array<double, 9>  k;               // 3x3 intrinsic matrix, row-major (default identity)
  std::vector<double>    d;               // distortion coefficients
  std::string            distortion_model;
  std::array<double, 9>  r;               // rectification matrix (default identity)
  std::array<double, 12> p;               // 3x4 projection matrix, row-major
  std::uint32_t          width{0}, height{0};
};

CameraModel   makeCameraModel(const CameraInfo &camera_info);
CameraModel64 makeCameraModel64(const CameraInfo &camera_info);
StatusCode    exportCameraInfo(const CameraModel &model, CameraInfo &camera_info_out);
StatusCode    exportCameraInfo64(const CameraModel64 &model, CameraInfo &camera_info_out);
```

The POD mirrors `sensor_msgs/CameraInfo` without the ROS-only fields
(`header`, `binning_*`, `roi` — the [ROS layer](interop.md) folds those into
K / P before producing it).

### Writing

`exportCameraInfo` picks the ROS distortion model from the camera model and
writes it only when that name holds the same lens:

| Model | `distortion_model` | `d` |
|---|---|---|
| `PINHOLE` + `NONE` / `RADTAN4` / `RADTAN5` | `plumb_bob` | 5 (`k1 k2 p1 p2 k3`, zeros for what the model lacks; a `RADTAN4` that carries `k3` in a fifth coefficient writes it) |
| `PINHOLE` + `RATIONAL8` | `rational_polynomial` | 8 |
| `PINHOLE` + `THIN_PRISM12` / `TILTED14` with the prism and tilt terms all zero | `rational_polynomial` | 8 |
| `EQUIDISTANT` + `NONE` / `KB4` | `equidistant` | 4 (`k1..k4`, zeros without a correction) |
| `EQUIDISTANT` + `KB8` with `k5..k8` all zero | `equidistant` | 4 |

Zeros are added only where the ROS name fixes the length, and no non-zero
coefficient is dropped. Every other model returns `INVALID_MODEL` and leaves
`camera_info_out` unchanged: a non-zero prism or tilt term, KB8 terms past
`k4`, a non-zero skew (ROS defines K without one, and receivers such as
`image_geometry` project without it), and every projection other than
pinhole and the equidistant law (equisolid, stereographic, orthographic,
omnidirectional, double sphere, EUCM, equirectangular, cylindrical). Use
camxiom's own description for those. An invalid model returns its
validation status, also without writing.

On `OK`, `k` is `[fx 0 cx; 0 fy cy; 0 0 1]`, `r` the identity and
`p = [K | 0]`; `width` / `height` are left to the caller.

What a receiver does with the file is up to the receiver. `image_geometry`,
for one, treats a `D` whose coefficients are all zero as no distortion for
every distortion model, `equidistant` included, and then leaves points
unchanged. That is right for pinhole, but the plain equidistant law (written
as `equidistant` with `[0, 0, 0, 0]`) is not a pinhole camera, so such a
receiver handles it as a different lens. camxiom reads it back correctly.

### Reading

`makeCameraModel` reads only the names `exportCameraInfo` writes, with the
coefficient counts they define:

| `distortion_model` | `d` length | Model |
|---|---|---|
| `plumb_bob` | 0 / 4 / 5 | `PINHOLE` + `NONE` / `RADTAN4` / `RADTAN5` |
| `rational_polynomial` | 8 / 12 / 14 | `PINHOLE` + `RATIONAL8` / `THIN_PRISM12` / `TILTED14` |
| `equidistant` | 4 | `EQUIDISTANT` + `KB4` |

K gives `fx = k[0]`, `skew = k[1]`, `cx = k[2]`, `fy = k[4]`, `cy = k[5]`.
Any other name or length, including an empty name, yields a model that
`validateCameraModel` rejects; `makeCameraModel` never throws, so validate
after reading.

ROS's `equidistant` is the four-coefficient fisheye polynomial (OpenCV's
fisheye model, the same function as KB4), not the plain equidistant law.
A model without a correction is written with zero coefficients and so reads
back as `KB4` (or `RADTAN5` for pinhole) with zero coefficients: the same
lens, projecting every ray to the same pixel, but a correction with free
coefficients rather than none. camxiom's own description keeps that
distinction.

## camxiom's own description

```cpp
struct CamxiomCameraInfo {
  std::uint32_t         image_width{0}, image_height{0};
  std::string           projection_model;        // toString(ProjectionModelType)
  std::vector<double>   projection_parameters;   // xi | xi, alpha | alpha, beta | none
  std::string           distortion_model;        // toString(DistortionModelType), "none" without one
  std::vector<double>   distortion_parameters;   // exactly the correction's coefficients
  std::array<double, 5> intrinsic_parameters;    // fx, fy, cx, cy, skew
};

CameraModel   makeCameraModel(const CamxiomCameraInfo &camera_info);
CameraModel64 makeCameraModel64(const CamxiomCameraInfo &camera_info);
StatusCode    exportCameraInfo(const CameraModel &model, CamxiomCameraInfo &camera_info_out);
StatusCode    exportCameraInfo64(const CameraModel64 &model, CamxiomCameraInfo &camera_info_out);
```

It keeps the projection law and its own parameters apart from the lens
correction (see
[Projection law and lens correction](../camera-models.md#projection-law-and-lens-correction)):
`projection_parameters` holds `xi` for omnidirectional, `xi, alpha` for double
sphere, `alpha, beta` for EUCM and nothing for the other laws, and
`distortion_parameters` holds exactly the correction's coefficients (4 for
`radtan4` and `kb4`, 5 for `radtan5`, 8 for `rational8` and `kb8`, 12 for
`thin_prism12`, 14 for `tilted14`, none for `none`). A correction whose
coefficients are zero stays that correction.

`exportCameraInfo` writes every valid model, the latitude-longitude
projections included; a float model's values are written as the shortest
decimals that read back as the same floats. `makeCameraModel` reads the
toString names only, requires exactly the law's parameters and the
correction's coefficients, and derives `theta_max` again (`updateThetaMax`):
a description that is not a valid model yields one `validateCameraModel`
rejects. A model therefore reads back as itself, apart from a `theta_max`
narrower than the derived one, which is not stored, and a `radtan4` holding
the five-term layout, which is written as `radtan5` and reads back as that
(the same lens, all five coefficients kept). `image_width` /
`image_height` are left to the caller on export.

## Round-trip picture

```
                     exportCameraInfo (ROS lenses only)
  CameraModel ──────────────────────────────────────────► CameraInfo
       ▲  ▲                                                   │  ▲
       │  └────────────────── makeCameraModel ────────────────┘  └── ros.hpp: sensor_msgs (binning/ROI folded)
       │
       │             exportCameraInfo (every model)
       ├────────────────────────────────────────────────► CamxiomCameraInfo
       └───────────────────── makeCameraModel ◄───────────────┘
```

`makeCameraModel64` and `exportCameraInfo64` pass intrinsics, coefficients
and projection parameters through without rounding them to float.
