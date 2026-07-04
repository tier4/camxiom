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

#ifndef CAMXIOM__DISTORTION__PLANE_IMPL_HPP
#define CAMXIOM__DISTORTION__PLANE_IMPL_HPP

// Scalar-templated plane (projective / pinhole-family) distortion core
// (#1 step 2a). This is the single source of truth for the forward distortion,
// its analytic Jacobian, the tilt 3x3 application and the Newton+LM
// undistortion that were previously hand-duplicated between the float
// implementation (src/distortion/plane.cpp, camxiom::detail) and the double
// implementation (src/projection64/internal.hpp, camxiom::detail64).
//
// The float and double thin wrappers instantiate these templates for <float>
// and <double> respectively, preserving the existing detail::/detail64:: API
// and the NormPoint / NormPoint64 types (now aliases of NormPointT<T>).
//
// Only <cmath>/<array>/<algorithm> and the camxiom types are required; this
// header has no Ceres/Eigen-solver dependency.

#include "camxiom/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace camxiom::detail_impl
{

// Precision-specific epsilon: matches the historical detail::kEpsilon (float)
// and detail64::kEpsilon (double) so degenerate-denominator / singular-matrix
// rejections trigger at the same thresholds as before.
template <typename T>
struct PlaneTraits;

template <>
struct PlaneTraits<float>
{
  static constexpr float kEpsilon = 1e-8f;
};

template <>
struct PlaneTraits<double>
{
  static constexpr double kEpsilon = 1e-15;
};

template <typename T>
struct NormPointT
{
  T x{T(0)};
  T y{T(0)};
};

template <typename T>
inline bool isFinite2(const T x, const T y)
{
  return std::isfinite(x) && std::isfinite(y);
}

template <typename T>
inline bool isFinite3(const T x, const T y, const T z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

template <typename T>
inline StatusCode applyProjectiveMatrix(
  const std::array<T, 9> &matrix, const T x_in, const T y_in, T &x_out, T &y_out
)
{
  const T mx = matrix[0] * x_in + matrix[1] * y_in + matrix[2];
  const T my = matrix[3] * x_in + matrix[4] * y_in + matrix[5];
  const T mz = matrix[6] * x_in + matrix[7] * y_in + matrix[8];
  if (!isFinite3(mx, my, mz) || std::abs(mz) <= PlaneTraits<T>::kEpsilon)
  {
    x_out = T(0);
    y_out = T(0);
    return StatusCode::DOMAIN_ERROR;
  }

  x_out = mx / mz;
  y_out = my / mz;
  if (!isFinite2(x_out, y_out))
  {
    x_out = T(0);
    y_out = T(0);
    return StatusCode::NUMERIC_ERROR;
  }
  return StatusCode::OK;
}

// Forward plane distortion (no Jacobian).
template <typename T>
inline StatusCode distortPlaneNoTilt(
  const DistortionModelT<T> &model, const T x, const T y, NormPointT<T> &out_xy
)
{
  if (!isFinite2(x, y))
  {
    return StatusCode::INVALID_INPUT;
  }

  const T r2 = x * x + y * y;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const auto &c = model.coeffs;
  const T k1 = c[0];
  const T k2 = c[1];
  const T p1 = c[2];
  const T p2 = c[3];
  const T k3 = c[4];

  const T num = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
  T radial = num;
  if (model.is_rational)
  {
    const T k4 = c[5];
    const T k5 = c[6];
    const T k6 = c[7];
    const T den = T(1) + k4 * r2 + k5 * r4 + k6 * r6;
    if (std::abs(den) <= PlaneTraits<T>::kEpsilon)
    {
      return StatusCode::DOMAIN_ERROR;
    }
    radial = num / den;
  }

  const T x_tan = T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
  const T y_tan = p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

  T x_d = x * radial + x_tan;
  T y_d = y * radial + y_tan;

  if (model.has_thin_prism)
  {
    const T s1 = c[8];
    const T s2 = c[9];
    const T s3 = c[10];
    const T s4 = c[11];
    x_d += s1 * r2 + s2 * r4;
    y_d += s3 * r2 + s4 * r4;
  }

  if (!isFinite2(x_d, y_d))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  out_xy.x = x_d;
  out_xy.y = y_d;
  return StatusCode::OK;
}

// Forward plane distortion together with the 2x2 distortion Jacobian.
template <typename T>
inline StatusCode distortPlaneNoTiltWithJacobian(
  const DistortionModelT<T> &model, const T x, const T y, NormPointT<T> &out_xy, T &j00, T &j01,
  T &j10, T &j11
)
{
  if (!isFinite2(x, y))
  {
    return StatusCode::INVALID_INPUT;
  }

  const T r2 = x * x + y * y;
  const T r4 = r2 * r2;
  const T r6 = r4 * r2;

  const auto &c = model.coeffs;
  const T k1 = c[0];
  const T k2 = c[1];
  const T p1 = c[2];
  const T p2 = c[3];
  const T k3 = c[4];

  const T num = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
  const T dnum_dx = T(2) * x * (k1 + T(2) * k2 * r2 + T(3) * k3 * r4);
  const T dnum_dy = T(2) * y * (k1 + T(2) * k2 * r2 + T(3) * k3 * r4);

  T radial = num;
  T drad_dx = dnum_dx;
  T drad_dy = dnum_dy;
  if (model.is_rational)
  {
    const T k4 = c[5];
    const T k5 = c[6];
    const T k6 = c[7];
    const T den = T(1) + k4 * r2 + k5 * r4 + k6 * r6;
    if (std::abs(den) <= PlaneTraits<T>::kEpsilon)
    {
      return StatusCode::DOMAIN_ERROR;
    }
    const T dden_dx = T(2) * x * (k4 + T(2) * k5 * r2 + T(3) * k6 * r4);
    const T dden_dy = T(2) * y * (k4 + T(2) * k5 * r2 + T(3) * k6 * r4);
    const T inv_den = T(1) / den;
    radial = num * inv_den;
    const T inv_den2 = inv_den * inv_den;
    drad_dx = (dnum_dx * den - num * dden_dx) * inv_den2;
    drad_dy = (dnum_dy * den - num * dden_dy) * inv_den2;
  }

  const T x_tan = T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
  const T y_tan = p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

  T dx_tan_dx = T(2) * p1 * y + T(6) * p2 * x;
  T dx_tan_dy = T(2) * p1 * x + T(2) * p2 * y;
  T dy_tan_dx = T(2) * p1 * x + T(2) * p2 * y;
  T dy_tan_dy = T(6) * p1 * y + T(2) * p2 * x;

  T x_d = x * radial + x_tan;
  T y_d = y * radial + y_tan;

  if (model.has_thin_prism)
  {
    const T s1 = c[8];
    const T s2 = c[9];
    const T s3 = c[10];
    const T s4 = c[11];
    x_d += s1 * r2 + s2 * r4;
    y_d += s3 * r2 + s4 * r4;

    dx_tan_dx += T(2) * x * (s1 + T(2) * s2 * r2);
    dx_tan_dy += T(2) * y * (s1 + T(2) * s2 * r2);
    dy_tan_dx += T(2) * x * (s3 + T(2) * s4 * r2);
    dy_tan_dy += T(2) * y * (s3 + T(2) * s4 * r2);
  }

  if (!isFinite2(x_d, y_d))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  out_xy.x = x_d;
  out_xy.y = y_d;

  j00 = radial + x * drad_dx + dx_tan_dx;
  j01 = x * drad_dy + dx_tan_dy;
  j10 = y * drad_dx + dy_tan_dx;
  j11 = radial + y * drad_dy + dy_tan_dy;

  if (!isFinite2(j00, j01) || !isFinite2(j10, j11))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode distortPlaneModel(
  const DistortionModelT<T> &model, const NormPointT<T> &in_xy, NormPointT<T> &out_xy
)
{
  if (!isFinite2(in_xy.x, in_xy.y))
  {
    return StatusCode::INVALID_INPUT;
  }

  switch (model.space)
  {
    case DistortionSpace::NONE:
      out_xy = in_xy;
      return StatusCode::OK;
    case DistortionSpace::ANGLE:
      return StatusCode::INVALID_MODEL;
    case DistortionSpace::PLANE:
      break;
    default:
      return StatusCode::INVALID_MODEL;
  }

  NormPointT<T> out_no_tilt{};
  const StatusCode no_tilt_status = distortPlaneNoTilt(model, in_xy.x, in_xy.y, out_no_tilt);
  if (no_tilt_status != StatusCode::OK)
  {
    return no_tilt_status;
  }

  if (!model.has_tilt)
  {
    out_xy = out_no_tilt;
    return StatusCode::OK;
  }

  T x_tilt = T(0);
  T y_tilt = T(0);
  const StatusCode tilt_status =
    applyProjectiveMatrix(model.tilt_matrix, out_no_tilt.x, out_no_tilt.y, x_tilt, y_tilt);
  if (tilt_status != StatusCode::OK)
  {
    return tilt_status;
  }
  out_xy.x = x_tilt;
  out_xy.y = y_tilt;
  return StatusCode::OK;
}

template <typename T>
inline StatusCode distortPlaneModelWithJacobian(
  const DistortionModelT<T> &model, const NormPointT<T> &in_xy, NormPointT<T> &out_xy, T &j00,
  T &j01, T &j10, T &j11
)
{
  if (!isFinite2(in_xy.x, in_xy.y))
  {
    return StatusCode::INVALID_INPUT;
  }

  switch (model.space)
  {
    case DistortionSpace::NONE:
      out_xy = in_xy;
      j00 = T(1);
      j01 = T(0);
      j10 = T(0);
      j11 = T(1);
      return StatusCode::OK;
    case DistortionSpace::ANGLE:
      return StatusCode::INVALID_MODEL;
    case DistortionSpace::PLANE:
      break;
    default:
      return StatusCode::INVALID_MODEL;
  }

  T nj00 = T(0);
  T nj01 = T(0);
  T nj10 = T(0);
  T nj11 = T(0);
  NormPointT<T> out_no_tilt{};
  const StatusCode no_tilt_status =
    distortPlaneNoTiltWithJacobian(model, in_xy.x, in_xy.y, out_no_tilt, nj00, nj01, nj10, nj11);
  if (no_tilt_status != StatusCode::OK)
  {
    return no_tilt_status;
  }

  if (!model.has_tilt)
  {
    out_xy = out_no_tilt;
    j00 = nj00;
    j01 = nj01;
    j10 = nj10;
    j11 = nj11;
    return StatusCode::OK;
  }

  T x_tilt = T(0);
  T y_tilt = T(0);
  const StatusCode tilt_status =
    applyProjectiveMatrix(model.tilt_matrix, out_no_tilt.x, out_no_tilt.y, x_tilt, y_tilt);
  if (tilt_status != StatusCode::OK)
  {
    return tilt_status;
  }
  out_xy.x = x_tilt;
  out_xy.y = y_tilt;

  const auto &M = model.tilt_matrix;
  const T mz = M[6] * out_no_tilt.x + M[7] * out_no_tilt.y + M[8];
  const T inv_mz = T(1) / mz;
  const T inv_mz2 = inv_mz * inv_mz;

  const T t00 = (M[0] * mz - (M[0] * out_no_tilt.x + M[1] * out_no_tilt.y + M[2]) * M[6]) * inv_mz2;
  const T t01 = (M[1] * mz - (M[0] * out_no_tilt.x + M[1] * out_no_tilt.y + M[2]) * M[7]) * inv_mz2;
  const T t10 = (M[3] * mz - (M[3] * out_no_tilt.x + M[4] * out_no_tilt.y + M[5]) * M[6]) * inv_mz2;
  const T t11 = (M[4] * mz - (M[3] * out_no_tilt.x + M[4] * out_no_tilt.y + M[5]) * M[7]) * inv_mz2;

  j00 = t00 * nj00 + t01 * nj10;
  j01 = t00 * nj01 + t01 * nj11;
  j10 = t10 * nj00 + t11 * nj10;
  j11 = t10 * nj01 + t11 * nj11;
  return StatusCode::OK;
}

template <typename T>
inline bool solveNewtonStep(
  const T j00, const T j01, const T j10, const T j11, const T residual_x, const T residual_y,
  T &delta_x, T &delta_y
)
{
  const T det = j00 * j11 - j01 * j10;
  if (!std::isfinite(det) || std::abs(det) <= PlaneTraits<T>::kEpsilon)
  {
    delta_x = T(0);
    delta_y = T(0);
    return false;
  }

  const T inv_det = T(1) / det;
  delta_x = (-j11 * residual_x + j01 * residual_y) * inv_det;
  delta_y = (j10 * residual_x - j00 * residual_y) * inv_det;
  return isFinite2(delta_x, delta_y);
}

template <typename T>
inline bool solveLmStep(
  const T j00, const T j01, const T j10, const T j11, const T residual_x, const T residual_y,
  const T lambda, T &delta_x, T &delta_y
)
{
  const T a00 = (j00 * j00 + j10 * j10) + lambda;
  const T a01 = (j00 * j01 + j10 * j11);
  const T a11 = (j01 * j01 + j11 * j11) + lambda;
  const T b0 = -(j00 * residual_x + j10 * residual_y);
  const T b1 = -(j01 * residual_x + j11 * residual_y);

  const T det = a00 * a11 - a01 * a01;
  if (!std::isfinite(det) || std::abs(det) <= PlaneTraits<T>::kEpsilon)
  {
    delta_x = T(0);
    delta_y = T(0);
    return false;
  }

  const T inv_det = T(1) / det;
  delta_x = (a11 * b0 - a01 * b1) * inv_det;
  delta_y = (-a01 * b0 + a00 * b1) * inv_det;
  return isFinite2(delta_x, delta_y);
}

template <typename T>
inline StatusCode undistortPlaneNoTiltLm(
  const DistortionModelT<T> &model, const NormPointT<T> &observed_xy, const int max_iterations,
  const T residual_tolerance, const T step_tolerance, NormPointT<T> &undistorted_xy
)
{
  T x = observed_xy.x;
  T y = observed_xy.y;
  const int max_iter = std::max(1, max_iterations);

  for (int iter = 0; iter < max_iter; ++iter)
  {
    NormPointT<T> distorted{};
    T j00 = T(0);
    T j01 = T(0);
    T j10 = T(0);
    T j11 = T(0);
    const StatusCode status =
      distortPlaneNoTiltWithJacobian(model, x, y, distorted, j00, j01, j10, j11);
    if (status != StatusCode::OK)
    {
      return status;
    }

    const T residual_x = distorted.x - observed_xy.x;
    const T residual_y = distorted.y - observed_xy.y;
    const T residual_norm = std::max(std::abs(residual_x), std::abs(residual_y));
    if (residual_norm <= residual_tolerance)
    {
      undistorted_xy.x = x;
      undistorted_xy.y = y;
      return StatusCode::OK;
    }

    T delta_x = T(0);
    T delta_y = T(0);
    bool accepted = false;

    if (solveNewtonStep(j00, j01, j10, j11, residual_x, residual_y, delta_x, delta_y))
    {
      const T x_candidate = x + delta_x;
      const T y_candidate = y + delta_y;
      NormPointT<T> distorted_candidate{};
      const StatusCode candidate_status =
        distortPlaneNoTilt(model, x_candidate, y_candidate, distorted_candidate);
      if (candidate_status == StatusCode::OK)
      {
        const T cand_residual_x = distorted_candidate.x - observed_xy.x;
        const T cand_residual_y = distorted_candidate.y - observed_xy.y;
        const T cand_residual_norm = std::max(std::abs(cand_residual_x), std::abs(cand_residual_y));
        if (cand_residual_norm < residual_norm)
        {
          x = x_candidate;
          y = y_candidate;
          accepted = true;
        }
      }
    }

    if (!accepted)
    {
      T lambda = T(1e-4);
      for (int lm_iter = 0; lm_iter < 8; ++lm_iter)
      {
        if (!solveLmStep(j00, j01, j10, j11, residual_x, residual_y, lambda, delta_x, delta_y))
        {
          lambda *= T(10);
          continue;
        }

        const T x_candidate = x + delta_x;
        const T y_candidate = y + delta_y;
        NormPointT<T> distorted_candidate{};
        const StatusCode candidate_status =
          distortPlaneNoTilt(model, x_candidate, y_candidate, distorted_candidate);
        if (candidate_status != StatusCode::OK)
        {
          lambda *= T(10);
          continue;
        }

        const T cand_residual_x = distorted_candidate.x - observed_xy.x;
        const T cand_residual_y = distorted_candidate.y - observed_xy.y;
        const T cand_residual_norm = std::max(std::abs(cand_residual_x), std::abs(cand_residual_y));
        if (cand_residual_norm < residual_norm)
        {
          x = x_candidate;
          y = y_candidate;
          accepted = true;
          break;
        }

        lambda *= T(10);
      }
    }

    if (!accepted)
    {
      undistorted_xy.x = x;
      undistorted_xy.y = y;
      return StatusCode::NON_CONVERGED;
    }

    const T step_norm = std::max(std::abs(delta_x), std::abs(delta_y));
    if (step_norm <= step_tolerance)
    {
      NormPointT<T> distorted_check{};
      const StatusCode check_status = distortPlaneNoTilt(model, x, y, distorted_check);
      if (check_status != StatusCode::OK)
      {
        return check_status;
      }
      const T residual_check_x = distorted_check.x - observed_xy.x;
      const T residual_check_y = distorted_check.y - observed_xy.y;
      const T residual_check_norm =
        std::max(std::abs(residual_check_x), std::abs(residual_check_y));
      if (residual_check_norm <= T(10) * residual_tolerance)
      {
        undistorted_xy.x = x;
        undistorted_xy.y = y;
        return StatusCode::OK;
      }
    }
  }

  undistorted_xy.x = x;
  undistorted_xy.y = y;
  return StatusCode::NON_CONVERGED;
}

template <typename T>
inline StatusCode undistortPlaneModel(
  const DistortionModelT<T> &model, const NormPointT<T> &observed_xy, const int max_iterations,
  const T residual_tolerance, const T step_tolerance, const bool skip_verify,
  NormPointT<T> &undistorted_xy
)
{
  switch (model.space)
  {
    case DistortionSpace::NONE:
      undistorted_xy = observed_xy;
      return StatusCode::OK;
    case DistortionSpace::ANGLE:
      return StatusCode::INVALID_MODEL;
    case DistortionSpace::PLANE:
      break;
    default:
      return StatusCode::INVALID_MODEL;
  }

  NormPointT<T> observed_target = observed_xy;
  if (model.has_tilt)
  {
    T x_untilt = T(0);
    T y_untilt = T(0);
    const StatusCode untilt_status = applyProjectiveMatrix(
      model.inv_tilt_matrix, observed_xy.x, observed_xy.y, x_untilt, y_untilt
    );
    if (untilt_status != StatusCode::OK)
    {
      return untilt_status;
    }
    observed_target.x = x_untilt;
    observed_target.y = y_untilt;
  }

  const StatusCode solve_status = undistortPlaneNoTiltLm(
    model, observed_target, max_iterations, residual_tolerance, step_tolerance, undistorted_xy
  );
  if (solve_status != StatusCode::OK)
  {
    return solve_status;
  }

  if (!skip_verify)
  {
    NormPointT<T> verify{};
    const StatusCode verify_status = distortPlaneModel(model, undistorted_xy, verify);
    if (verify_status != StatusCode::OK)
    {
      return verify_status;
    }
    const T verify_residual =
      std::max(std::abs(verify.x - observed_xy.x), std::abs(verify.y - observed_xy.y));
    if (verify_residual > T(20) * residual_tolerance)
    {
      return StatusCode::NON_CONVERGED;
    }
  }
  return StatusCode::OK;
}

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__DISTORTION__PLANE_IMPL_HPP
