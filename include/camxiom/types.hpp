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
//   accept rays up to their theta_max.
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
//   radians.
//
// Distortion coefficients (DistortionModelT::coeffs, OpenCV-compatible
// ordering; `count` says how many leading entries are meaningful):
//   PLANE space (pinhole-family), full TILTED14 layout:
//     [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]
//     RADTAN4 uses the first 4, RADTAN5 the first 5, RATIONAL8 the first 8
//     (k4..k6 form the denominator), THIN_PRISM12 the first 12, TILTED14
//     all 14 (tau_x/tau_y in radians).
//   ANGLE space (fisheye-family): [k1, k2, k3, k4] on the incidence angle,
//     theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6
//                          + k4*theta^8) (Kannala-Brandt / OpenCV fisheye).
//   Projection-level parameters (omnidirectional xi, double-sphere xi/alpha,
//   EUCM alpha/beta) live in ProjectionModelT, NOT in coeffs. (Interchange
//   formats that pack them into D — e.g. sensor_msgs — are unpacked by the
//   compat layer.)
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
  OPENCV_FISHEYE4,
  KB4,
  KB8,
  EQUIDISTANT,
  EQUISOLID,
  STEREOGRAPHIC,
  ORTHOGRAPHIC,
  OMNIDIRECTIONAL,
  UNKNOWN
};

enum class DistortionSpace : std::uint8_t { NONE = 0U, PLANE, ANGLE };

enum class ProjectionModelType : std::uint8_t {
  PINHOLE = 0U,
  FISHEYE_THETA,
  OMNIDIRECTIONAL,
  DOUBLE_SPHERE,
  EUCM,
  UNKNOWN
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
  DistortionSpace space{DistortionSpace::NONE};
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
