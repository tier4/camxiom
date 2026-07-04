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

#ifndef CAMXIOM__DETAIL__DS_FORWARD_HPP
#define CAMXIOM__DETAIL__DS_FORWARD_HPP

#include "camxiom/types.hpp"

#include <cmath>

namespace camxiom::detail
{

/// Compute Double Sphere forward projection intermediates.
/// Validity is the bijectivity region of Usenko et al. 2018 (eq. 43-45),
/// z > -w2 * d1, plus a denom > eps numerical guard. denom > 0 alone is NOT
/// a FOV check: for alpha > 0.5 it holds for every direction.
///
/// Template parameter T: float or double.
/// eps: numerical epsilon appropriate for the precision (e.g. 1e-8f or 1e-15).
///
/// On success, d1_out, r_sq_out, xi_d1_z_out, d2_out, denom_out are all set.
/// Returns StatusCode::OK on success.
template <typename T>
inline StatusCode computeDsForward(
  const T xi, const T alpha, const T x, const T y, const T z, const T eps, T &d1_out, T &r_sq_out,
  T &xi_d1_z_out, T &d2_out, T &denom_out
)
{
  const T r_sq = x * x + y * y;
  r_sq_out = r_sq;

  d1_out = std::sqrt(r_sq + z * z);
  if (!std::isfinite(d1_out) || d1_out <= eps)
  {
    return StatusCode::INVALID_INPUT;
  }

  xi_d1_z_out = xi * d1_out + z;

  d2_out = std::sqrt(r_sq + xi_d1_z_out * xi_d1_z_out);
  if (!std::isfinite(d2_out) || d2_out <= eps)
  {
    return StatusCode::NUMERIC_ERROR;
  }

  denom_out = alpha * d2_out + (static_cast<T>(1) - alpha) * xi_d1_z_out;
  if (!std::isfinite(denom_out) || denom_out <= eps)
  {
    return StatusCode::OUT_OF_FOV;
  }

  // Bijectivity region (Usenko et al. 2018, eq. 43-45): z > -w2 * d1. For
  // alpha > 0.5 the denominator is positive for *every* direction (|xi*d1+z|
  // <= d2 makes denom >= (2*alpha-1)*d2), so the check above alone lets rays
  // behind the camera alias onto valid-looking pixels.
  const T w1 = (alpha <= static_cast<T>(0.5)) ? alpha / (static_cast<T>(1) - alpha)
                                              : (static_cast<T>(1) - alpha) / alpha;
  const T w2 = (w1 + xi) / std::sqrt(static_cast<T>(2) * w1 * xi + xi * xi + static_cast<T>(1));
  if (z <= -w2 * d1_out)
  {
    return StatusCode::OUT_OF_FOV;
  }

  return StatusCode::OK;
}

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__DS_FORWARD_HPP
