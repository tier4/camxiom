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

#ifndef CAMXIOM__PROJECTION64__INTERNAL_HPP
#define CAMXIOM__PROJECTION64__INTERNAL_HPP

#include "camxiom/projection64.hpp"
#include "detail/internal.hpp"
#include "distortion/angle_impl.hpp"
#include "distortion/plane_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace camxiom
{

namespace detail64
{

inline constexpr double kEpsilon = 1e-15;
inline constexpr double kPi = camxiom::constants::kPi;
inline constexpr double kHalfPi = camxiom::constants::kHalfPi;

inline bool isFinite2(const double x, const double y)
{
  return std::isfinite(x) && std::isfinite(y);
}

inline bool isFinite3(const double x, const double y, const double z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

inline PixelResult64 invalidPixelResult(const StatusCode status)
{
  PixelResult64 r;
  r.status = status;
  return r;
}

inline RayResult64 invalidRayResult(const StatusCode status)
{
  RayResult64 r;
  r.status = status;
  r.ray.origin = Eigen::Vector3d::Zero();
  r.ray.direction = Eigen::Vector3d::Zero();
  return r;
}

// ---------------------------------------------------------------------------
// Plane distortion (double) — thin wrappers over the scalar-templated
// detail_impl core (#1 step 2a). NormPoint64 aliases NormPointT<double>; the
// math is shared with the float detail:: path in distortion/plane_impl.hpp.
// ---------------------------------------------------------------------------

using NormPoint64 = detail_impl::NormPointT<double>;

inline StatusCode distortPlaneNoTilt64(
  const DistortionModel64 &model, const double x, const double y, NormPoint64 &out_xy
)
{
  return detail_impl::distortPlaneNoTilt<double>(model, x, y, out_xy);
}

inline StatusCode applyProjectiveMatrix64(
  const std::array<double, 9> &matrix, const double x_in, const double y_in, double &x_out,
  double &y_out
)
{
  return detail_impl::applyProjectiveMatrix<double>(matrix, x_in, y_in, x_out, y_out);
}

inline StatusCode distortPlaneModel64(
  const DistortionModel64 &model, const NormPoint64 &in_xy, NormPoint64 &out_xy
)
{
  return detail_impl::distortPlaneModel<double>(model, in_xy, out_xy);
}

inline StatusCode distortPlaneNoTiltWithJ64(
  const DistortionModel64 &model, const double x, const double y, NormPoint64 &out_xy, double &j00,
  double &j01, double &j10, double &j11
)
{
  return detail_impl::distortPlaneNoTiltWithJacobian<double>(
    model, x, y, out_xy, j00, j01, j10, j11
  );
}

inline bool solveNewtonStep64(
  const double j00, const double j01, const double j10, const double j11, const double residual_x,
  const double residual_y, double &delta_x, double &delta_y
)
{
  return detail_impl::solveNewtonStep<double>(
    j00, j01, j10, j11, residual_x, residual_y, delta_x, delta_y
  );
}

inline bool solveLmStep64(
  const double j00, const double j01, const double j10, const double j11, const double residual_x,
  const double residual_y, const double lambda, double &delta_x, double &delta_y
)
{
  return detail_impl::solveLmStep<double>(
    j00, j01, j10, j11, residual_x, residual_y, lambda, delta_x, delta_y
  );
}

inline StatusCode undistortPlaneNoTiltLm64(
  const DistortionModel64 &model, const NormPoint64 &observed_xy, const SolverOptions64 &opts,
  NormPoint64 &undistorted
)
{
  return detail_impl::undistortPlaneNoTiltLm<double>(
    model, observed_xy, opts.max_iterations, opts.residual_tolerance, opts.step_tolerance,
    undistorted
  );
}

inline StatusCode undistortPlaneModel64(
  const DistortionModel64 &model, const NormPoint64 &observed, const SolverOptions64 &opts,
  NormPoint64 &undistorted
)
{
  return detail_impl::undistortPlaneModel<double>(
    model, observed, opts.max_iterations, opts.residual_tolerance, opts.step_tolerance,
    opts.skip_verify, undistorted
  );
}

// ---------------------------------------------------------------------------
// Angle distortion (double)
// ---------------------------------------------------------------------------

inline StatusCode distortTheta64(
  const DistortionModel64 &model, const double theta, double &theta_d
)
{
  return detail_impl::distortTheta<double>(model, theta, theta_d);
}

inline StatusCode distortThetaDerivative64(
  const DistortionModel64 &model, const double theta, double &derivative_out
)
{
  return detail_impl::distortThetaDerivative<double>(model, theta, derivative_out);
}

inline StatusCode undistortThetaHybrid64(
  const DistortionModel64 &model, const ProjectionModel64 &proj, const double radius_d,
  const SolverOptions64 &opts, double &theta_out
)
{
  return detail_impl::undistortThetaHybrid<double>(
    model, proj, radius_d, opts.max_iterations, opts.residual_tolerance, opts.step_tolerance,
    theta_out
  );
}

inline bool removeIntrinsics64(
  const IntrinsicsModel64 &intr, const Pixel2d &pixel, double &x_d, double &y_d
)
{
  y_d = (pixel.v - intr.cy) / intr.fy;
  x_d = (pixel.u - intr.cx - intr.skew * y_d) / intr.fx;
  return isFinite2(x_d, y_d);
}

// requiredFiniteCoefficientCount64 removed (#1 step 4): the double validator now
// uses the shared, type-only detail::requiredFiniteCoefficientCount in
// detail/internal.hpp.

/// Double counterpart of detail::validateCameraModelQuery (see
/// detail/internal.hpp): the per-call query-tier guard shared by every
/// double-precision query path. Defined in src/projection64/validate.cpp.
StatusCode validateCameraModelQuery64(const CameraModel64 &model);

}  // namespace detail64

}  // namespace camxiom

#endif  // CAMXIOM__PROJECTION64__INTERNAL_HPP
