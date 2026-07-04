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
/// (0, pi). Callers guard with hasThetaMaxCap so the ray norm is only
/// required when a cap is actually active.
template <typename T>
inline bool withinThetaMax(const T theta_max, const T z, const T norm)
{
  return z >= norm * std::cos(theta_max);
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
