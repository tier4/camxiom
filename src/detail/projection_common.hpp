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

#ifndef CAMXIOM__DETAIL__PROJECTION_COMMON_HPP
#define CAMXIOM__DETAIL__PROJECTION_COMMON_HPP

// Scalar-templated helpers shared by every per-model projection core
// (#1 step 3). These were previously hand-duplicated as the float
// camxiom::detail:: helpers (src/detail/internal.hpp) and the double
// camxiom::detail64:: helpers (src/projection64/internal.hpp); the per-model
// projection_impl.hpp cores now call these <T> templates so the result/
// intrinsics plumbing has a single source of truth across precisions.

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "detail/ds_forward.hpp"      // dsNegW2 (shared bijectivity bound)
#include "distortion/plane_impl.hpp"  // isFinite2 (shared finite check)

#include <Eigen/Core>

#include <cmath>

namespace camxiom::detail_impl
{

/// True when the model caps the FOV below the wide-angle default of pi.
/// static_cast<float>(kPi) rounds to kPiF, so both precisions short-circuit
/// at their own default theta_max and pay nothing on the hot path.
template <typename T>
inline bool hasThetaMaxCap(const T theta_max)
{
  return theta_max < static_cast<T>(constants::kPi);
}

/// theta <= theta_max via cosine comparison (no atan2 needed):
/// theta <= theta_max  <=>  z >= norm * cos(theta_max) for theta_max in
/// (0, pi). Callers guard with PreparedProjectionT::has_theta_cap so the ray
/// norm is only required when a cap is actually active.
template <typename T>
inline bool withinThetaMaxCos(const T cos_theta_max, const T z, const T norm)
{
  return z >= norm * cos_theta_max;
}

// ---------------------------------------------------------------------------
// Model-derived projection constants.
//
// Two validity checks in the per-model cores depend only on the CameraModel,
// never on the ray or pixel being projected:
//
//   * the FOV cap needs cos(theta_max) — double-sphere, EUCM and
//     omnidirectional, forward and inverse;
//   * the double-sphere bijectivity bound needs w2, a square root and two
//     divisions away from xi and alpha.
//
// Evaluating them inside the per-point kernel makes every projected point pay
// a transcendental (and, for double-sphere, a square root) to re-derive a
// value that is constant across the whole point cloud. PreparedProjectionT
// carries them so a caller projecting many points against one model computes
// them once.
//
// The generic single-point API (rayToPixel / pixelToRay) keeps deriving them
// per call: it accepts an unvalidated CameraModel by reference and has nowhere
// to cache anything, so its cost is unchanged. ValidatedCameraModel and the
// batch / SIMD layers, which already resolve the model once, reuse a prepared
// instance instead.
// ---------------------------------------------------------------------------
template <typename T>
struct PreparedProjectionT
{
  /// theta_max caps the FOV below the wide-angle default of pi.
  bool has_theta_cap{false};
  /// cos(theta_max), read only when has_theta_cap. Defaults to cos(pi) = -1 so
  /// an uncapped model accepts every forward direction even if a caller
  /// ignores the flag.
  T cos_theta_max{T(-1)};
  /// -w2 of the double-sphere bijectivity region (Usenko et al. 2018,
  /// eq. 43-45). Zero for every other projection type, which never reads it.
  T ds_neg_w2{T(0)};
};

using PreparedProjection = PreparedProjectionT<float>;
using PreparedProjection64 = PreparedProjectionT<double>;

/// Derive the model-constant projection state. Cheap enough to call per point
/// (the generic API does), but the whole point of the type is that callers
/// with a fixed model call it once.
template <typename T>
inline PreparedProjectionT<T> prepareProjection(const ProjectionModelT<T> &projection)
{
  PreparedProjectionT<T> prepared;
  prepared.has_theta_cap = hasThetaMaxCap(projection.theta_max);
  if (prepared.has_theta_cap)
  {
    prepared.cos_theta_max = std::cos(projection.theta_max);
  }
  if (projection.type == ProjectionModelType::DOUBLE_SPHERE)
  {
    prepared.ds_neg_w2 = detail::dsNegW2(projection.xi, projection.alpha);
  }
  return prepared;
}

template <typename T>
inline PixelResultT<T> invalidPixelResult(const StatusCode status)
{
  PixelResultT<T> result;
  result.status = status;
  result.pixel = Pixel2T<T>{};
  return result;
}

template <typename T>
inline RayResultT<T> invalidRayResult(const StatusCode status)
{
  RayResultT<T> result;
  result.status = status;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = Eigen::Matrix<T, 3, 1>::Zero();
  return result;
}

// Undo the affine intrinsics (fx, fy, cx, cy, skew) to recover the distorted
// normalised image coordinates. Matches the historical detail::removeIntrinsics
// (float) and detail64::removeIntrinsics64 (double) line for line.
template <typename T>
inline bool removeIntrinsics(
  const IntrinsicsModelT<T> &intrinsics, const Pixel2T<T> &pixel, T &x_distorted, T &y_distorted
)
{
  y_distorted = (pixel.v - intrinsics.cy) / intrinsics.fy;
  x_distorted = (pixel.u - intrinsics.cx - intrinsics.skew * y_distorted) / intrinsics.fx;
  return isFinite2(x_distorted, y_distorted);
}

// ---------------------------------------------------------------------------
// Out-of-line iterative inverse solvers (defined once in projection_solve.cpp).
//
// The forward distortion (distortPlaneModel / distortTheta) is cheap and stays
// inlined into each per-model rayToPixel. The inverse solvers, however, are
// heavy (plane: Newton + Levenberg-Marquardt with verification; theta:
// bracketed Newton with bisection fallback). Before #1 step 3 the float path
// reached them through the out-of-line camxiom::detail wrappers, so the solver
// body was compiled once and merely *called* from pixelToRay. Calling the
// detail_impl templates directly would instead inline the whole solver into
// every per-model pixelToRay translation unit and measurably regressed the
// float inverse hot path. These non-template overloads restore that single
// out-of-line definition while keeping one shared source per precision.
StatusCode undistortPlaneSolve(
  const DistortionModelT<float> &model, const NormPointT<float> &observed, int max_iterations,
  float residual_tolerance, float step_tolerance, bool skip_verify, NormPointT<float> &undistorted
);
StatusCode undistortPlaneSolve(
  const DistortionModelT<double> &model, const NormPointT<double> &observed, int max_iterations,
  double residual_tolerance, double step_tolerance, bool skip_verify,
  NormPointT<double> &undistorted
);

StatusCode undistortThetaSolve(
  const DistortionModelT<float> &model, const ProjectionModelT<float> &projection, float radius_d,
  int max_iterations, float residual_tolerance, float step_tolerance, float &theta_out
);
StatusCode undistortThetaSolve(
  const DistortionModelT<double> &model, const ProjectionModelT<double> &projection,
  double radius_d, int max_iterations, double residual_tolerance, double step_tolerance,
  double &theta_out
);

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__DETAIL__PROJECTION_COMMON_HPP
