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

#ifndef CAMXIOM__MODEL__DISTORTION_AUX_HPP
#define CAMXIOM__MODEL__DISTORTION_AUX_HPP

// Single source of truth for the DERIVED distortion state of
// DistortionModelT<T>: is_rational / has_thin_prism / has_tilt and the
// TILTED14 tilt matrices.
//
// Historically this logic existed as three independent copies (float factory
// path in model/internal.hpp, float solver write-back in pnp_solver.cpp,
// double solver path in pnp_cost_analytical_batch.cpp) whose "non-zero
// coefficient" and tilt-r22 thresholds had drifted apart (strict !=0 vs
// >1e-12f vs >1e-15, and r22 eps 1e-8f vs 1e-7f vs 1e-12). A coefficient
// near one of those thresholds could make the solver-internal model disagree
// with the factory-built model, silently breaking the "diagnostics use the
// same equations as the solver" premise of the calibration uncertainty code.
// All callers now share this header.
//
// Canonical semantics (= the original factory behaviour):
//   * A coefficient is "active" iff it is exactly non-zero. Coefficients
//     freed by the optimizer are practically never exactly zero, and the
//     projection math is continuous in the coefficients, so no epsilon is
//     needed to decide activation.
//   * computeTiltMatrices fails (identity fallback, caller clears has_tilt)
//     when the tilt rotation is degenerate (|r22| ~ 0) or any input/output
//     value is non-finite.

#include "camxiom/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace camxiom::detail
{

template <typename T>
inline void setIdentity3x3(std::array<T, 9> &matrix)
{
  matrix = {T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1)};
}

/// True iff any coefficient in [start_index, start_index + max_count),
/// clamped to the model's active coefficient count, is exactly non-zero.
template <typename T>
inline bool hasAnyNonZero(
  const DistortionModelT<T> &model, const int start_index, const int max_count
)
{
  const int begin = std::max(0, start_index);
  const int end =
    std::max(begin, std::min(static_cast<int>(model.count), begin + std::max(0, max_count)));
  for (int index = begin; index < end; ++index)
  {
    if (model.coeffs[static_cast<std::size_t>(index)] != T(0))
    {
      return true;
    }
  }
  return false;
}

/// Degenerate-tilt guard on r22 = cos(tau_y)*cos(tau_x), scale-appropriate
/// per precision. The float value matches the historical factory epsilon.
template <typename T>
inline constexpr T tiltR22Epsilon()
{
  return std::is_same_v<T, float> ? T(1e-8) : T(1e-12);
}

/// Build the TILTED14 tilt matrix (proj_z * Ry(tau_y) * Rx(tau_x)) and its
/// inverse. Returns false (and writes identities) when the tilt is
/// degenerate or any value is non-finite.
template <typename T>
inline bool computeTiltMatrices(
  const T tau_x, const T tau_y, std::array<T, 9> &tilt_matrix, std::array<T, 9> &inv_tilt_matrix
)
{
  const T c_tx = std::cos(tau_x);
  const T s_tx = std::sin(tau_x);
  const T c_ty = std::cos(tau_y);
  const T s_ty = std::sin(tau_y);

  if (!std::isfinite(c_tx) || !std::isfinite(s_tx) || !std::isfinite(c_ty) || !std::isfinite(s_ty))
  {
    setIdentity3x3(tilt_matrix);
    setIdentity3x3(inv_tilt_matrix);
    return false;
  }

  // rot_xy = Ry(tau_y) * Rx(tau_x)
  const std::array<T, 9> rot_xy{c_ty, s_ty * s_tx, -s_ty * c_tx, T(0),       c_tx,
                                s_tx, s_ty,        -c_ty * s_tx, c_ty * c_tx};

  const T r22 = rot_xy[8];
  if (!std::isfinite(r22) || std::abs(r22) <= tiltR22Epsilon<T>())
  {
    setIdentity3x3(tilt_matrix);
    setIdentity3x3(inv_tilt_matrix);
    return false;
  }

  const std::array<T, 9> proj_z{r22, T(0), -rot_xy[2], T(0), r22, -rot_xy[5], T(0), T(0), T(1)};

  // tilt_matrix = proj_z * rot_xy
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      T value = T(0);
      for (int mid = 0; mid < 3; ++mid)
      {
        value += proj_z[row * 3 + mid] * rot_xy[mid * 3 + col];
      }
      tilt_matrix[row * 3 + col] = value;
    }
  }

  const T inv_r22 = T(1) / r22;
  const std::array<T, 9> inv_proj_z{
    inv_r22, T(0), rot_xy[2] * inv_r22, T(0), inv_r22, rot_xy[5] * inv_r22, T(0), T(0), T(1)};

  const std::array<T, 9> rot_xy_t{rot_xy[0], rot_xy[3], rot_xy[6], rot_xy[1], rot_xy[4],
                                  rot_xy[7], rot_xy[2], rot_xy[5], rot_xy[8]};

  // inv_tilt_matrix = rot_xy^T * inv_proj_z
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      T value = T(0);
      for (int mid = 0; mid < 3; ++mid)
      {
        value += rot_xy_t[row * 3 + mid] * inv_proj_z[mid * 3 + col];
      }
      inv_tilt_matrix[row * 3 + col] = value;
    }
  }

  for (const T value : tilt_matrix)
  {
    if (!std::isfinite(value))
    {
      setIdentity3x3(tilt_matrix);
      setIdentity3x3(inv_tilt_matrix);
      return false;
    }
  }
  for (const T value : inv_tilt_matrix)
  {
    if (!std::isfinite(value))
    {
      setIdentity3x3(tilt_matrix);
      setIdentity3x3(inv_tilt_matrix);
      return false;
    }
  }
  return true;
}

/// Rebuild every derived member (is_rational / has_thin_prism / has_tilt /
/// tilt_matrix / inv_tilt_matrix) from type + count + coeffs. All other
/// members are left untouched.
template <typename T>
inline void rebuildDistortionAuxState(DistortionModelT<T> &d)
{
  d.is_rational = d.type == DistortionModelType::RATIONAL8 ||
                  d.type == DistortionModelType::THIN_PRISM12 ||
                  d.type == DistortionModelType::TILTED14;
  d.has_thin_prism =
    (d.type == DistortionModelType::THIN_PRISM12 || d.type == DistortionModelType::TILTED14) &&
    hasAnyNonZero(d, 8, 4);
  d.has_tilt = d.type == DistortionModelType::TILTED14 && hasAnyNonZero(d, 12, 2);
  if (d.has_tilt)
  {
    if (!computeTiltMatrices(d.coeffs[12], d.coeffs[13], d.tilt_matrix, d.inv_tilt_matrix))
    {
      d.has_tilt = false;
    }
  }
  else
  {
    setIdentity3x3(d.tilt_matrix);
    setIdentity3x3(d.inv_tilt_matrix);
  }
}

}  // namespace camxiom::detail

#endif  // CAMXIOM__MODEL__DISTORTION_AUX_HPP
