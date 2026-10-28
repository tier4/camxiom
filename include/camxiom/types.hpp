// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CAMXIOM__TYPES_HPP
#define CAMXIOM__TYPES_HPP

#include <Eigen/Core>

#include <array>
#include <cstdint>

namespace camxiom
{

// ===========================================================================
// Conventions (apply to the ENTIRE camxiom API)
// ===========================================================================
//
// Camera coordinate frame (right-handed):
//   +Z forward (optical axis, points out of the camera into the scene),
//   +X right, +Y down. Rays with z <= 0 are BEHIND_CAMERA for pinhole;
//   wide-angle models (fisheye / omnidirectional / double-sphere / EUCM)
//   accept rays up to their theta_max; the lat-long projections
//   (equirectangular / cylindrical) accept every direction except the two
//   poles straight up and straight down.
//
// Pixel coordinates:
//   u along +X (columns, increases rightward), v along +Y (rows, increases
//   downward). PIXEL-CENTER origin: integer coordinate (0, 0) is the CENTER
//   of the top-left pixel, so the geometric center of a W x H image is
//   ((W-1)/2, (H-1)/2) (this is what the rectify output models use for
//   cx/cy).
//
// Intrinsics (IntrinsicsModelT):
//   fx, fy, cx, cy, skew are in pixels. With distorted normalized
//   coordinates (x_d, y_d):
//     u = fx * x_d + skew * y_d + cx
//     v = fy * y_d            + cy
//   (skew multiplies y_d in the u equation only; most cameras have skew = 0.)
//
// Angles:
//   Every angle in the API is in RADIANS unless the name says otherwise
//   (only the *_fov_deg fields are degrees). ProjectionModelT::theta_max is
//   the maximum accepted angle between a ray and the optical axis, in
//   radians. PINHOLE and the lat-long projections ignore it.
//
// Lat-long projections (EQUIRECTANGULAR, CYLINDRICAL):
//   For a ray (X, Y, Z), with h = sqrt(X^2 + Z^2):
//     longitude  lambda = atan2(X, Z) in [-pi, pi): 0 straight ahead,
//                positive to the right, the seam directly behind (-Z)
//                belongs to the left edge (-pi);
//     latitude   phi = atan2(Y, h) in (-pi/2, pi/2): positive downward,
//                so it grows with v like the rest of the frame.
//   EQUIRECTANGULAR uses (x_d, y_d) = (lambda, phi); CYLINDRICAL uses
//   (x_d, y_d) = (lambda, tan(phi)) = (lambda, Y / h). The intrinsics then
//   apply as for every other model. A pole (h = 0) has no longitude and is
//   a DOMAIN_ERROR in both directions: rayToPixel for the ray, pixelToRay
//   for a pixel that unprojects onto it (the equirectangular rows at
//   phi = +-pi/2). pixelToRay rejects pixels whose longitude falls outside
//   [-pi, pi) or (equirectangular) whose latitude falls past +-pi/2 with
//   OUT_OF_FOV instead of wrapping them. A longitude within a few float
//   ulps (about 6e-6 rad, in both precisions) below -pi is the seam itself
//   (an image edge recovers it only to rounding), and the left edge always
//   unprojects to the ray behind the camera and projects back onto the left
//   edge, in float as in double.
//   Orientation is the caller's rotation, as for every model: there is no
//   separate axis setting.
//
// Projection law and lens correction:
//   ProjectionModelT::type names the law (pinhole, a fisheye law, or a
//   sphere-based model); DistortionModelT::type names the correction applied
//   on top of it. Plane corrections (radtan family) combine with PINHOLE,
//   OMNIDIRECTIONAL, DOUBLE_SPHERE and EUCM; the angle corrections KB4/KB8
//   combine with EQUIDISTANT only; EQUISOLID, STEREOGRAPHIC, ORTHOGRAPHIC,
//   EQUIRECTANGULAR and CYLINDRICAL take no correction. validateCameraModel
//   rejects other combinations.
//
// Distortion coefficients (DistortionModelT::coeffs, OpenCV-compatible
// ordering; `count` says how many leading entries are meaningful):
//   PLANE space (pinhole-family), full TILTED14 layout:
//     [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]
//     RADTAN4 uses the first 4, RADTAN5 the first 5, RATIONAL8 the first 8
//     (k4..k6 form the denominator), THIN_PRISM12 the first 12, TILTED14
//     all 14 (tau_x/tau_y in radians).
//   ANGLE space (on the equidistant law): KB4 [k1, k2, k3, k4] on the
//     incidence angle, theta_d = theta * (1 + k1*theta^2 + k2*theta^4
//     + k3*theta^6 + k4*theta^8) (Kannala-Brandt / OpenCV fisheye); KB8
//     continues the series to k8*theta^16.
//   Projection-level parameters (omnidirectional xi, double-sphere xi/alpha,
//   EUCM alpha/beta) live in ProjectionModelT, NOT in coeffs; camxiom's own
//   camera description (CamxiomCameraInfo) keeps them apart from the
//   coefficients as well.
// ===========================================================================

/// [[nodiscard]] on the enum makes every function returning a StatusCode by
/// value warn when the caller silently drops the result (the cheap,
/// non-breaking alternative adopted instead of a Result<T> migration).
/// Intentional discards must be explicit: `(void)fn(...)`.
enum class [[nodiscard]] StatusCode : std::uint8_t{
  OK = 0U, INVALID_INPUT, INVALID_MODEL, BEHIND_CAMERA, OUT_OF_FOV, DOMAIN_ERROR, NON_CONVERGED,
  NUMERIC_ERROR,
  // Input data's geometric configuration is structurally unable to identify
  // a unique solution (e.g. collinear points for homography, coplanar 3D
  // points for PnP, fewer-than-minimum distinct samples). Distinct from
  // NUMERIC_ERROR, which signals a numerical pathology during computation.
  DEGENERATE_CONFIG};

enum class DistortionModelType : std::uint8_t {
  NONE = 0U,
  RADTAN4,
  RADTAN5,
  RATIONAL8,
  THIN_PRISM12,
  TILTED14,
  KB4,
  KB8,
  UNKNOWN
};

enum class DistortionSpace : std::uint8_t { NONE = 0U, PLANE, ANGLE };

/// The space a distortion model acts in. It is a property of the type, not a
/// separate setting: the radtan family corrects the normalized image plane
/// (PLANE), the Kannala-Brandt family corrects the incidence angle of the
/// equidistant law (ANGLE), and NONE (or an unusable type) corrects nothing.
constexpr DistortionSpace distortionSpaceOf(const DistortionModelType type) noexcept
{
  switch (type)
  {
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
      return DistortionSpace::PLANE;
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
      return DistortionSpace::ANGLE;
    case DistortionModelType::NONE:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return DistortionSpace::NONE;
}

/// The projection law: how a ray direction maps to the normalized image
/// plane before any lens correction. Fisheye laws are functions of the angle
/// theta between the ray and the optical axis (equidistant r = theta,
/// equisolid r = 2 sin(theta/2), stereographic r = 2 tan(theta/2),
/// orthographic r = sin(theta)); then come pinhole, the sphere-based Mei,
/// double-sphere and EUCM models, and the lat-long projections, which map
/// longitude and latitude to the two image axes (see the conventions above).
/// Lens corrections live in DistortionModelType.
enum class ProjectionModelType : std::uint8_t {
  PINHOLE = 0U,
  EQUIDISTANT,
  EQUISOLID,
  STEREOGRAPHIC,
  ORTHOGRAPHIC,
  OMNIDIRECTIONAL,
  DOUBLE_SPHERE,
  EUCM,
  EQUIRECTANGULAR,
  CYLINDRICAL,
  UNKNOWN
};

/// True for the fisheye laws, which map the angle theta to an image radius:
/// EQUIDISTANT, EQUISOLID, STEREOGRAPHIC and ORTHOGRAPHIC.
constexpr bool isFisheyeProjection(const ProjectionModelType type) noexcept
{
  return type == ProjectionModelType::EQUIDISTANT || type == ProjectionModelType::EQUISOLID ||
         type == ProjectionModelType::STEREOGRAPHIC || type == ProjectionModelType::ORTHOGRAPHIC;
}

/// True for the lat-long projections, which map longitude and latitude to the
/// image axes: EQUIRECTANGULAR and CYLINDRICAL. They project and unproject
/// like any other model but take no lens correction, and the calibration
/// layer does not estimate them.
constexpr bool isLatLongProjection(const ProjectionModelType type) noexcept
{
  return type == ProjectionModelType::EQUIRECTANGULAR || type == ProjectionModelType::CYLINDRICAL;
}

/// Image dimensions in pixels.
struct ImageSize
{
  int width{0};
  int height{0};

  constexpr bool isValid() const { return width > 0 && height > 0; }
};

// ---------------------------------------------------------------------------
// Scalar-templated geometry types (#1 float/double unification).
//
// The structs below are written once as `*T<T>` templates and aliased to the
// float32 runtime types (`Pixel2`, `CameraModel`, ...) here, and to their
// double counterparts (`Pixel2d`, `CameraModel64`, ...) in types64.hpp. They
// stay plain aggregates (no user-declared constructors), so aggregate /
// brace initialisation and the POD layout are preserved bit-for-bit per type.
// `T` is float or double for normal evaluation.
// ---------------------------------------------------------------------------

template <typename T>
struct Pixel2T
{
  T u{T(0)};
  T v{T(0)};
};

template <typename T>
struct Ray3T
{
  Eigen::Matrix<T, 3, 1> origin{Eigen::Matrix<T, 3, 1>::Zero()};
  Eigen::Matrix<T, 3, 1> direction{Eigen::Matrix<T, 3, 1>(T(0), T(0), T(1))};
};

template <typename T>
struct IntrinsicsModelT
{
  T fx{T(1)};
  T fy{T(1)};
  T cx{T(0)};
  T cy{T(0)};
  T skew{T(0)};
};

template <typename T>
struct ProjectionModelT
{
  ProjectionModelType type{ProjectionModelType::UNKNOWN};
  T theta_max{T(1.5707963267948966)};
  T xi{T(0)};
  T alpha{T(0)};
  T beta{T(1)};
};

template <typename T>
struct DistortionModelT
{
  DistortionModelType type{DistortionModelType::NONE};
  std::array<T, 14> coeffs{};
  std::array<T, 9> tilt_matrix{{T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1)}};
  std::array<T, 9> inv_tilt_matrix{{T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1)}};
  std::uint8_t count{0U};
  bool is_rational{false};
  bool has_thin_prism{false};
  bool has_tilt{false};
};

template <typename T>
struct CameraModelT
{
  IntrinsicsModelT<T> intrinsics{};
  ProjectionModelT<T> projection{};
  DistortionModelT<T> distortion{};
};

template <typename T>
struct [[nodiscard]] PixelResultT
{
  StatusCode status{StatusCode::INVALID_INPUT};
  Pixel2T<T> pixel{};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

template <typename T>
struct [[nodiscard]] RayResultT
{
  StatusCode status{StatusCode::INVALID_INPUT};
  Ray3T<T> ray{};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

// --- float32 runtime aliases (primary real-time API) -----------------------
using Pixel2 = Pixel2T<float>;
using Ray3 = Ray3T<float>;
using IntrinsicsModel = IntrinsicsModelT<float>;
using ProjectionModel = ProjectionModelT<float>;
using DistortionModel = DistortionModelT<float>;
using CameraModel = CameraModelT<float>;
using PixelResult = PixelResultT<float>;
using RayResult = RayResultT<float>;

struct SolverOptions
{
  int max_iterations{10};
  float residual_tolerance{1e-6f};
  float step_tolerance{1e-8f};
  bool skip_verify{false};
};

const char *toString(StatusCode status_code);
const char *toString(DistortionModelType model_type);
const char *toString(ProjectionModelType model_type);
const char *toString(DistortionSpace space);

}  // namespace camxiom

#endif  // CAMXIOM__TYPES_HPP
