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

#include "detail/simd_inverse.hpp"

#include "camxiom/projection.hpp"
#include "detail/projection_common.hpp"
#include "detail/projection_models.hpp"
#include "detail/simd_ds_eucm.hpp"
#include "detail/simd_fisheye.hpp"
#include "detail/simd_pinhole.hpp"

#include <Eigen/Core>

#include <cmath>

namespace camxiom::detail
{

namespace
{

inline bool requiresSimdRoundTripVerification(const CameraModel &model)
{
  return model.projection.type == ProjectionModelType::FISHEYE_THETA &&
         (model.distortion.type == DistortionModelType::OPENCV_FISHEYE4 ||
          model.distortion.type == DistortionModelType::KB4 ||
          model.distortion.type == DistortionModelType::KB8);
}

/// Distortion families rayToPixelFisheyeSse4 implements — must stay in sync
/// with supportsFisheyeSseDistortion (batch/float.cpp). Other configs verify
/// through the scalar projection instead.
inline bool fisheyeSseForwardSupports(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::EQUIDISTANT:
      return true;
    case DistortionModelType::NONE:
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::KB8:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return false;
}

inline void storeScalarRayResult(const RayResult &result, const int off, float *dirs_xyz)
{
  if (result.status == StatusCode::OK)
  {
    dirs_xyz[off] = result.ray.direction.x();
    dirs_xyz[off + 1] = result.ray.direction.y();
    dirs_xyz[off + 2] = result.ray.direction.z();
  }
  else
  {
    dirs_xyz[off] = 0.0f;
    dirs_xyz[off + 1] = 0.0f;
    dirs_xyz[off + 2] = 0.0f;
  }
}

/// All-scalar batch loop for models the SIMD inverse cannot solve exactly
/// (currently KB8: the SIMD Newton carries 4 polynomial coefficients only,
/// so k5..k8 would be silently dropped and the 1 px round-trip guard can
/// accept a systematically biased theta).
template <typename ScalarInvFn>
inline int pixelToRayBatchScalarOnly(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options, ScalarInvFn scalar_fn
)
{
  int valid_count = 0;
  for (int i = 0; i < count; ++i)
  {
    const RayResult sr = scalar_fn(model, Pixel2{u_in[i], v_in[i]}, solver_options);
    storeScalarRayResult(sr, i * 3, dirs_xyz);
    if (statuses_out != nullptr) statuses_out[i] = sr.status;
    if (sr.status == StatusCode::OK) ++valid_count;
  }
  return valid_count;
}

}  // namespace

#ifdef CAMXIOM_HAS_SSE2

// ---------------------------------------------------------------------------
// Helper: store 4 ray results from SSE registers + write statuses
// ---------------------------------------------------------------------------

namespace
{

inline void storeRaysSse4(
  const __m128 dx, const __m128 dy, const __m128 dz, const int valid_mask, float *dirs_xyz
)
{
  // Interleave x,y,z for 4 rays
  alignas(16) float xs[4], ys[4], zs[4];
  _mm_store_ps(xs, dx);
  _mm_store_ps(ys, dy);
  _mm_store_ps(zs, dz);
  for (int i = 0; i < 4; ++i)
  {
    const int off = i * 3;
    if (valid_mask & (1 << i))
    {
      dirs_xyz[off] = xs[i];
      dirs_xyz[off + 1] = ys[i];
      dirs_xyz[off + 2] = zs[i];
    }
    else
    {
      dirs_xyz[off] = 0.0f;
      dirs_xyz[off + 1] = 0.0f;
      dirs_xyz[off + 2] = 0.0f;
    }
  }
}

// normalize 4 direction vectors in SSE
inline void normalizeSse4(__m128 &dx, __m128 &dy, __m128 &dz, __m128 &valid)
{
  const __m128 norm_sq =
    _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), _mm_mul_ps(dz, dz));
  const __m128 norm = _mm_sqrt_ps(norm_sq);
  const __m128 eps = _mm_set1_ps(1e-8f);
  const __m128 norm_ok = _mm_cmpgt_ps(norm, eps);
  valid = _mm_and_ps(valid, _mm_and_ps(norm_ok, finiteMaskSseInv(norm)));
  const __m128 safe_norm = selectSseInv(_mm_set1_ps(1.0f), norm, norm_ok);
  const __m128 inv_norm = _mm_div_ps(_mm_set1_ps(1.0f), safe_norm);
  dx = _mm_mul_ps(dx, inv_norm);
  dy = _mm_mul_ps(dy, inv_norm);
  dz = _mm_mul_ps(dz, inv_norm);
}

// Check if distortion is NONE (no distortion to undo)
inline bool isDistortionNone(const DistortionModel &dist)
{
  return dist.type == DistortionModelType::NONE;
}

inline bool isDefaultSolverOptionsForSimd(const SolverOptions &solver_options)
{
  constexpr float kTol = 1e-12f;
  return solver_options.max_iterations == 10 &&
         std::abs(solver_options.residual_tolerance - 1e-6f) <= kTol &&
         std::abs(solver_options.step_tolerance - 1e-8f) <= kTol && !solver_options.skip_verify;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pinhole SSE4: removeIntrinsics → undistort → (x, y, 1) → normalize
// ---------------------------------------------------------------------------

int pixelToRayPinholeSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m128 x_d, y_d, valid;
  removeIntrinsicsSse4(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m128 ux, uy;
  if (isDistortionNone(model.distortion))
  {
    ux = x_d;
    uy = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonSse4(model.distortion, x_d, y_d, ux, uy, valid);
  }
  else
  {
    // tilt: fall back, return 0 valid
    for (int i = 0; i < 12; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  __m128 dz = _mm_set1_ps(1.0f);
  normalizeSse4(ux, uy, dz, valid);

  const int vm = _mm_movemask_ps(valid);
  storeRaysSse4(ux, uy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// DoubleSphere SSE4: removeIntrinsics → undistort → DS unproject → normalize
// ---------------------------------------------------------------------------

int pixelToRayDsphSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m128 x_d, y_d, valid;
  removeIntrinsicsSse4(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m128 mx, my;
  if (isDistortionNone(model.distortion))
  {
    mx = x_d;
    my = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonSse4(model.distortion, x_d, y_d, mx, my, valid);
  }
  else
  {
    for (int i = 0; i < 12; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 two = _mm_set1_ps(2.0f);
  const __m128 eps = _mm_set1_ps(1e-8f);
  const float alpha = model.projection.alpha;
  const float xi = model.projection.xi;
  const __m128 alpha_v = _mm_set1_ps(alpha);
  const __m128 xi_v = _mm_set1_ps(xi);

  const __m128 r2 = _mm_add_ps(_mm_mul_ps(mx, mx), _mm_mul_ps(my, my));

  // inner = 1 - (2*alpha - 1) * r2
  const __m128 inner = _mm_sub_ps(one, _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(two, alpha_v), one), r2));
  valid = _mm_and_ps(valid, _mm_cmpge_ps(inner, _mm_setzero_ps()));
  const __m128 sqrt_inner = _mm_sqrt_ps(_mm_max_ps(inner, _mm_setzero_ps()));

  // mz_num = 1 - alpha^2 * r2
  const __m128 alpha_sq = _mm_mul_ps(alpha_v, alpha_v);
  const __m128 mz_num = _mm_sub_ps(one, _mm_mul_ps(alpha_sq, r2));

  // mz_den = alpha * sqrt_inner + (1 - alpha)
  const __m128 mz_den = _mm_add_ps(_mm_mul_ps(alpha_v, sqrt_inner), _mm_sub_ps(one, alpha_v));
  const __m128 den_ok = _mm_cmpgt_ps(absSseInv(mz_den), eps);
  valid = _mm_and_ps(valid, den_ok);
  const __m128 safe_den = selectSseInv(one, mz_den, den_ok);
  const __m128 mz = _mm_div_ps(mz_num, safe_den);

  // norm_sq = r2 + mz^2
  const __m128 norm_sq = _mm_add_ps(r2, _mm_mul_ps(mz, mz));
  // inner_xi = mz^2 + (1 - xi^2) * r2
  const __m128 xi_sq = _mm_mul_ps(xi_v, xi_v);
  const __m128 inner_xi = _mm_add_ps(_mm_mul_ps(mz, mz), _mm_mul_ps(_mm_sub_ps(one, xi_sq), r2));
  valid = _mm_and_ps(valid, _mm_cmpge_ps(inner_xi, _mm_setzero_ps()));

  const __m128 sqrt_inner_xi = _mm_sqrt_ps(_mm_max_ps(inner_xi, _mm_setzero_ps()));
  const __m128 scalar =
    _mm_div_ps(_mm_add_ps(_mm_mul_ps(mz, xi_v), sqrt_inner_xi), _mm_max_ps(norm_sq, eps));

  __m128 dx = _mm_mul_ps(scalar, mx);
  __m128 dy = _mm_mul_ps(scalar, my);
  __m128 dz = _mm_sub_ps(_mm_mul_ps(scalar, mz), xi_v);

  normalizeSse4(dx, dy, dz, valid);

  // Round-trip consistency with the forward theta_max cap (scalar parity:
  // double_sphere/projection_impl.hpp). dz is normalised here, so
  // theta <= theta_max reduces to dz >= cos(theta_max). Without this the
  // no-status fast path trusts the SIMD valid mask alone and hands out
  // beyond-FOV directions the scalar path rejects as OUT_OF_FOV.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m128 cos_tmax = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(dz, cos_tmax));
  }

  const int vm = _mm_movemask_ps(valid);
  storeRaysSse4(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// EUCM SSE4: removeIntrinsics → undistort → EUCM unproject → normalize
// ---------------------------------------------------------------------------

int pixelToRayEucmSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m128 x_d, y_d, valid;
  removeIntrinsicsSse4(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m128 mx, my;
  if (isDistortionNone(model.distortion))
  {
    mx = x_d;
    my = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonSse4(model.distortion, x_d, y_d, mx, my, valid);
  }
  else
  {
    for (int i = 0; i < 12; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 two = _mm_set1_ps(2.0f);
  const __m128 eps = _mm_set1_ps(1e-8f);
  const float alpha = model.projection.alpha;
  const float beta = model.projection.beta;
  const __m128 alpha_v = _mm_set1_ps(alpha);
  const __m128 beta_v = _mm_set1_ps(beta);

  const __m128 r2 = _mm_add_ps(_mm_mul_ps(mx, mx), _mm_mul_ps(my, my));

  // inner = 1 - (2*alpha - 1) * beta * r2
  const __m128 inner =
    _mm_sub_ps(one, _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(two, alpha_v), one), beta_v), r2));
  valid = _mm_and_ps(valid, _mm_cmpge_ps(inner, _mm_setzero_ps()));
  const __m128 sqrt_inner = _mm_sqrt_ps(_mm_max_ps(inner, _mm_setzero_ps()));

  // mz_num = 1 - alpha^2 * beta * r2
  const __m128 alpha_sq = _mm_mul_ps(alpha_v, alpha_v);
  const __m128 mz_num = _mm_sub_ps(one, _mm_mul_ps(_mm_mul_ps(alpha_sq, beta_v), r2));

  // mz_den = alpha * sqrt_inner + (1 - alpha)
  const __m128 mz_den = _mm_add_ps(_mm_mul_ps(alpha_v, sqrt_inner), _mm_sub_ps(one, alpha_v));
  const __m128 den_ok = _mm_cmpgt_ps(absSseInv(mz_den), eps);
  valid = _mm_and_ps(valid, den_ok);
  const __m128 safe_den = selectSseInv(one, mz_den, den_ok);
  const __m128 mz = _mm_div_ps(mz_num, safe_den);

  __m128 dx = mx;
  __m128 dy = my;
  __m128 dz = mz;

  normalizeSse4(dx, dy, dz, valid);

  // Scalar parity: eucm/projection_impl.hpp's forward theta_max cap
  // (normalised dz vs cos(theta_max)); see the DoubleSphere kernel note.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m128 cos_tmax = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(dz, cos_tmax));
  }

  const int vm = _mm_movemask_ps(valid);
  storeRaysSse4(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// Omni SSE4: removeIntrinsics → undistort → Omni unproject → normalize
// ---------------------------------------------------------------------------

int pixelToRayOmniSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m128 x_d, y_d, valid;
  removeIntrinsicsSse4(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m128 ux, uy;
  if (isDistortionNone(model.distortion))
  {
    ux = x_d;
    uy = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonSse4(model.distortion, x_d, y_d, ux, uy, valid);
  }
  else
  {
    for (int i = 0; i < 12; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 eps = _mm_set1_ps(1e-8f);
  const float xi = model.projection.xi;
  const __m128 xi_v = _mm_set1_ps(xi);
  const __m128 xi_sq = _mm_mul_ps(xi_v, xi_v);

  const __m128 r2 = _mm_add_ps(_mm_mul_ps(ux, ux), _mm_mul_ps(uy, uy));

  // inside_sqrt = 1 + (1 - xi^2) * r2
  const __m128 inside = _mm_add_ps(one, _mm_mul_ps(_mm_sub_ps(one, xi_sq), r2));
  valid = _mm_and_ps(valid, _mm_cmpge_ps(inside, _mm_setzero_ps()));
  const __m128 sqrt_term = _mm_sqrt_ps(_mm_max_ps(inside, _mm_setzero_ps()));

  // denom = 1 + r2
  const __m128 denom = _mm_add_ps(one, r2);
  const __m128 denom_ok = _mm_cmpgt_ps(denom, eps);
  valid = _mm_and_ps(valid, denom_ok);
  const __m128 safe_denom = selectSseInv(one, denom, denom_ok);

  // lambda = (xi + sqrt_term) / denom
  const __m128 lambda = _mm_div_ps(_mm_add_ps(xi_v, sqrt_term), safe_denom);
  valid = _mm_and_ps(valid, finiteMaskSseInv(lambda));

  __m128 dx = _mm_mul_ps(lambda, ux);
  __m128 dy = _mm_mul_ps(lambda, uy);
  __m128 dz = _mm_sub_ps(lambda, xi_v);

  normalizeSse4(dx, dy, dz, valid);

  // Scalar parity: omnidirectional/projection_impl.hpp's forward theta_max
  // cap (normalised dz vs cos(theta_max)); see the DoubleSphere kernel note.
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m128 cos_tmax = _mm_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm_and_ps(valid, _mm_cmpge_ps(dz, cos_tmax));
  }

  const int vm = _mm_movemask_ps(valid);
  storeRaysSse4(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// Fisheye SSE4: removeIntrinsics → undistortTheta → spherical → ray
// For EQUIDISTANT: theta = radius_d (direct)
// For polynomial: SIMD Newton to invert theta_d(theta)
// ---------------------------------------------------------------------------

int pixelToRayFisheyeSse4(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m128 x_d, y_d, valid;
  removeIntrinsicsSse4(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  const __m128 r_d = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(x_d, x_d), _mm_mul_ps(y_d, y_d)));
  const __m128 eps = _mm_set1_ps(1e-8f);
  const __m128 one = _mm_set1_ps(1.0f);
  const __m128 zero = _mm_setzero_ps();

  // theta estimation: invert theta_d(theta)
  __m128 theta;
  const auto dtype = model.distortion.type;
  if (dtype == DistortionModelType::EQUIDISTANT || dtype == DistortionModelType::NONE)
  {
    theta = r_d;
  }
  else if (dtype == DistortionModelType::EQUISOLID)
  {
    // theta_d = 2*sin(theta/2) → theta = 2*asin(r_d/2)
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 arg = _mm_mul_ps(r_d, half);
    // clamp to [-1,1]
    const __m128 clamped = _mm_min_ps(_mm_max_ps(arg, _mm_set1_ps(-1.0f)), one);
    // scalar fallback for asin - store, compute, reload
    alignas(16) float arg_arr[4], theta_arr[4];
    _mm_store_ps(arg_arr, clamped);
    for (int i = 0; i < 4; ++i) theta_arr[i] = 2.0f * std::asin(arg_arr[i]);
    theta = _mm_load_ps(theta_arr);
  }
  else if (dtype == DistortionModelType::STEREOGRAPHIC)
  {
    // theta_d = 2*tan(theta/2) → theta = 2*atan(r_d/2)
    alignas(16) float rd_arr[4], theta_arr[4];
    _mm_store_ps(rd_arr, r_d);
    for (int i = 0; i < 4; ++i) theta_arr[i] = 2.0f * std::atan(rd_arr[i] * 0.5f);
    theta = _mm_load_ps(theta_arr);
  }
  else if (dtype == DistortionModelType::ORTHOGRAPHIC)
  {
    // theta_d = sin(theta) → theta = asin(r_d)
    const __m128 clamped = _mm_min_ps(_mm_max_ps(r_d, _mm_set1_ps(-1.0f)), one);
    alignas(16) float rd_arr[4], theta_arr[4];
    _mm_store_ps(rd_arr, clamped);
    for (int i = 0; i < 4; ++i) theta_arr[i] = std::asin(rd_arr[i]);
    theta = _mm_load_ps(theta_arr);
  }
  else
  {
    // Polynomial (OPENCV_FISHEYE4 / KB4): Newton iteration
    // theta_d = theta*(1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8)
    // Start from theta = r_d as initial guess
    theta = r_d;
    const auto &c = model.distortion.coeffs;
    const __m128 k1 = _mm_set1_ps(c[0]);
    const __m128 k2 = _mm_set1_ps(c[1]);
    const __m128 k3 = _mm_set1_ps(c[2]);
    const __m128 k4 = _mm_set1_ps(c[3]);

    for (int iter = 0; iter < 15; ++iter)
    {
      const __m128 t2 = _mm_mul_ps(theta, theta);
      const __m128 t4 = _mm_mul_ps(t2, t2);
      const __m128 t6 = _mm_mul_ps(t4, t2);
      const __m128 t8 = _mm_mul_ps(t4, t4);

      // f(t) = t*(1 + k1*t2 + k2*t4 + k3*t6 + k4*t8) - r_d
      __m128 poly = _mm_add_ps(one, _mm_mul_ps(k1, t2));
      poly = _mm_add_ps(poly, _mm_mul_ps(k2, t4));
      poly = _mm_add_ps(poly, _mm_mul_ps(k3, t6));
      poly = _mm_add_ps(poly, _mm_mul_ps(k4, t8));
      const __m128 f_val = _mm_sub_ps(_mm_mul_ps(theta, poly), r_d);

      // f'(t) = 1 + 3*k1*t2 + 5*k2*t4 + 7*k3*t6 + 9*k4*t8
      __m128 dpoly = _mm_add_ps(one, _mm_mul_ps(_mm_set1_ps(3.0f), _mm_mul_ps(k1, t2)));
      dpoly = _mm_add_ps(dpoly, _mm_mul_ps(_mm_set1_ps(5.0f), _mm_mul_ps(k2, t4)));
      dpoly = _mm_add_ps(dpoly, _mm_mul_ps(_mm_set1_ps(7.0f), _mm_mul_ps(k3, t6)));
      dpoly = _mm_add_ps(dpoly, _mm_mul_ps(_mm_set1_ps(9.0f), _mm_mul_ps(k4, t8)));

      const __m128 dpoly_abs_safe = _mm_max_ps(absSseInv(dpoly), eps);
      const __m128 dpoly_neg = _mm_cmplt_ps(dpoly, zero);
      const __m128 dpoly_safe =
        selectSseInv(dpoly_abs_safe, _mm_sub_ps(zero, dpoly_abs_safe), dpoly_neg);
      const __m128 delta = _mm_div_ps(f_val, dpoly_safe);
      theta = _mm_sub_ps(theta, delta);
      theta = _mm_max_ps(theta, zero);

      // Early exit once every lane's Newton step is negligible (theta is
      // O(1) rad, so 1e-7 is float-ulp territory). The finite/theta_max
      // checks below and the KB4 round-trip guard are unaffected.
      const __m128 big_step = _mm_cmpgt_ps(absSseInv(delta), _mm_set1_ps(1e-7f));
      if (!anyLaneSetSseInv(big_step))
      {
        break;
      }
    }
    // finite check BEFORE theta_max: NaN from overflow must not be swallowed by max(NaN,0)=0
    valid = _mm_and_ps(valid, finiteMaskSseInv(theta));
  }

  // theta_max check (conservative margin to avoid boundary false positives)
  const __m128 t_max = _mm_set1_ps(model.projection.theta_max - 1e-5f);
  valid = _mm_and_ps(valid, finiteMaskSseInv(theta));
  valid = _mm_and_ps(valid, _mm_cmple_ps(theta, t_max));
  valid = _mm_and_ps(valid, _mm_cmpge_ps(theta, zero));

  // on-axis: radius_d ≈ 0 → direction = (0, 0, 1)
  const __m128 on_axis = _mm_cmple_ps(r_d, eps);

  // cos_phi = x_d / r_d, sin_phi = y_d / r_d
  const __m128 safe_rd = _mm_max_ps(r_d, eps);
  const __m128 inv_rd = _mm_div_ps(one, safe_rd);
  const __m128 cos_phi = _mm_mul_ps(x_d, inv_rd);
  const __m128 sin_phi = _mm_mul_ps(y_d, inv_rd);

  // sin_theta, cos_theta (scalar fallback for sincos)
  alignas(16) float theta_arr[4], sin_arr[4], cos_arr[4];
  _mm_store_ps(theta_arr, theta);
  for (int i = 0; i < 4; ++i)
  {
    sin_arr[i] = std::sin(theta_arr[i]);
    cos_arr[i] = std::cos(theta_arr[i]);
  }
  const __m128 sin_t = _mm_load_ps(sin_arr);
  const __m128 cos_t = _mm_load_ps(cos_arr);

  __m128 dx = _mm_mul_ps(sin_t, cos_phi);
  __m128 dy = _mm_mul_ps(sin_t, sin_phi);
  __m128 dz = cos_t;

  // on-axis override: (0, 0, 1)
  dx = selectSseInv(dx, zero, on_axis);
  dy = selectSseInv(dy, zero, on_axis);
  dz = selectSseInv(dz, one, on_axis);

  normalizeSse4(dx, dy, dz, valid);

  const int vm = _mm_movemask_ps(valid);
  storeRaysSse4(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// Batch driver template: SSE → scalar fallback
// ---------------------------------------------------------------------------

namespace
{

using ScalarInvFn = RayResult (*)(const CameraModel &, const Pixel2 &, const SolverOptions &);
using Sse4InvFn = int (*)(const CameraModel &, const float *, const float *, float *);
using Sse4FwdFn = int (*)(const CameraModel &, const float *, float *, float *);

constexpr float kStatusVerifyThresholdSq =
  1.0f;  // 1 px round-trip tolerance for SIMD-OK verification

int pixelToRayBatchSseGeneric(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options, Sse4InvFn sse_fn, Sse4FwdFn fwd_fn,
  ScalarInvFn scalar_fn
)
{
  // The SIMD Newton runs a fixed 10-iteration / 1e-6 schedule and SIMD-OK
  // lanes are only guarded by the 1 px round-trip check, so caller-supplied
  // solver options (tighter tolerances, more iterations) would be silently
  // degraded to that fixed accuracy. Honour non-default options by solving
  // through the exact scalar path instead.
  if (!isDefaultSolverOptionsForSimd(solver_options))
  {
    return pixelToRayBatchScalarOnly(
      model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, scalar_fn
    );
  }

  int valid_count = 0;
  const int simd_count = count & ~3;
  const bool fast_simd_no_status = (statuses_out == nullptr) &&
                                   isDefaultSolverOptionsForSimd(solver_options) &&
                                   !requiresSimdRoundTripVerification(model);

  for (int i = 0; i < simd_count; i += 4)
  {
    float *out = dirs_xyz + i * 3;
    const int vm = sse_fn(model, u_in + i, v_in + i, out);

    if (fast_simd_no_status)
    {
      // Fast path: trust SIMD, no scalar calls
      valid_count += __builtin_popcount(vm & 0xF);
    }
    else
    {
      // Status path: SIMD-OK lanes verified with a forward round-trip. When
      // the model has a 4-wide SIMD forward kernel (fwd_fn) one SIMD pass
      // covers the whole group; otherwise each lane round-trips through the
      // scalar projection as before. Only lanes that fail the round-trip
      // (or that SIMD marked invalid) pay for a scalar solve.
      alignas(16) float fwd_u[4];
      alignas(16) float fwd_v[4];
      int fwd_vm = 0;
      if (fwd_fn != nullptr && (vm & 0xF) != 0)
      {
        fwd_vm = fwd_fn(model, out, fwd_u, fwd_v);
      }
      for (int j = 0; j < 4; ++j)
      {
        const int idx = i + j;
        if ((vm >> j) & 1)
        {
          const int off = idx * 3;
          bool verified;
          if (fwd_fn != nullptr)
          {
            const float du = fwd_u[j] - u_in[idx];
            const float dv = fwd_v[j] - v_in[idx];
            verified = ((fwd_vm >> j) & 1) != 0 && (du * du + dv * dv) < kStatusVerifyThresholdSq;
          }
          else
          {
            // No SIMD forward kernel for this model config → scalar verify.
            const Eigen::Vector3f dir(dirs_xyz[off], dirs_xyz[off + 1], dirs_xyz[off + 2]);
            const PixelResult fwd = rayToPixel(model, dir);
            const float du = fwd.pixel.u - u_in[idx];
            const float dv = fwd.pixel.v - v_in[idx];
            verified =
              fwd.status == StatusCode::OK && (du * du + dv * dv) < kStatusVerifyThresholdSq;
          }
          if (verified)
          {
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = StatusCode::OK;
            }
            ++valid_count;
          }
          else
          {
            // Round-trip failed → fall back to scalar inverse
            const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = sr.status;
            }
            if (sr.status == StatusCode::OK)
            {
              dirs_xyz[off] = sr.ray.direction.x();
              dirs_xyz[off + 1] = sr.ray.direction.y();
              dirs_xyz[off + 2] = sr.ray.direction.z();
              ++valid_count;
            }
            else
            {
              dirs_xyz[off] = 0.0f;
              dirs_xyz[off + 1] = 0.0f;
              dirs_xyz[off + 2] = 0.0f;
            }
          }
        }
        else
        {
          // SIMD says invalid → scalar for exact status
          const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
          if (statuses_out != nullptr)
          {
            statuses_out[idx] = sr.status;
          }
          if (sr.status == StatusCode::OK)
          {
            const int off = idx * 3;
            dirs_xyz[off] = sr.ray.direction.x();
            dirs_xyz[off + 1] = sr.ray.direction.y();
            dirs_xyz[off + 2] = sr.ray.direction.z();
            ++valid_count;
          }
          else
          {
            const int off = idx * 3;
            dirs_xyz[off] = 0.0f;
            dirs_xyz[off + 1] = 0.0f;
            dirs_xyz[off + 2] = 0.0f;
          }
        }
      }
    }
  }

  // scalar tail
  for (int i = simd_count; i < count; ++i)
  {
    const RayResult sr = scalar_fn(model, Pixel2{u_in[i], v_in[i]}, solver_options);
    const int off = i * 3;
    storeScalarRayResult(sr, off, dirs_xyz);
    if (statuses_out != nullptr) statuses_out[i] = sr.status;
    if (sr.status == StatusCode::OK) ++valid_count;
  }

  return valid_count;
}

}  // namespace

// ---------------------------------------------------------------------------
// SSE batch drivers
// ---------------------------------------------------------------------------

int pixelToRayBatchPinholeSse(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    // tilt: full scalar fallback
    int vc = 0;
    for (int i = 0; i < count; ++i)
    {
      const RayResult r = pinhole::pixelToRay(model, Pixel2{u_in[i], v_in[i]}, solver_options);
      const int off = i * 3;
      storeScalarRayResult(r, off, dirs_xyz);
      if (statuses_out) statuses_out[i] = r.status;
      if (r.status == StatusCode::OK) ++vc;
    }
    return vc;
  }
  return pixelToRayBatchSseGeneric(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayPinholeSse4,
    &rayToPixelPinholeSse4, &pinhole::pixelToRay
  );
}

int pixelToRayBatchDsphSse(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    int vc = 0;
    for (int i = 0; i < count; ++i)
    {
      const RayResult r =
        double_sphere::pixelToRay(model, Pixel2{u_in[i], v_in[i]}, solver_options);
      const int off = i * 3;
      storeScalarRayResult(r, off, dirs_xyz);
      if (statuses_out) statuses_out[i] = r.status;
      if (r.status == StatusCode::OK) ++vc;
    }
    return vc;
  }
  return pixelToRayBatchSseGeneric(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayDsphSse4,
    &rayToPixelDsphSse4, &double_sphere::pixelToRay
  );
}

int pixelToRayBatchEucmSse(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    int vc = 0;
    for (int i = 0; i < count; ++i)
    {
      const RayResult r = eucm::pixelToRay(model, Pixel2{u_in[i], v_in[i]}, solver_options);
      const int off = i * 3;
      storeScalarRayResult(r, off, dirs_xyz);
      if (statuses_out) statuses_out[i] = r.status;
      if (r.status == StatusCode::OK) ++vc;
    }
    return vc;
  }
  return pixelToRayBatchSseGeneric(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayEucmSse4,
    &rayToPixelEucmSse4, &eucm::pixelToRay
  );
}

int pixelToRayBatchOmniSse(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    int vc = 0;
    for (int i = 0; i < count; ++i)
    {
      const RayResult r =
        omnidirectional::pixelToRay(model, Pixel2{u_in[i], v_in[i]}, solver_options);
      const int off = i * 3;
      storeScalarRayResult(r, off, dirs_xyz);
      if (statuses_out) statuses_out[i] = r.status;
      if (r.status == StatusCode::OK) ++vc;
    }
    return vc;
  }
  return pixelToRayBatchSseGeneric(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayOmniSse4,
    &rayToPixelOmniSse4, &omnidirectional::pixelToRay
  );
}

int pixelToRayBatchFisheyeSse(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.type == DistortionModelType::KB8)
  {
    return pixelToRayBatchScalarOnly(
      model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &fisheye::pixelToRay
    );
  }
  return pixelToRayBatchSseGeneric(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayFisheyeSse4,
    fisheyeSseForwardSupports(model.distortion.type) ? &rayToPixelFisheyeSse4 : nullptr,
    &fisheye::pixelToRay
  );
}

#endif  // CAMXIOM_HAS_SSE2

// ===========================================================================
// AVX2 (8-point) inverse kernels
// ===========================================================================

#ifdef CAMXIOM_HAS_AVX2

namespace
{

inline void storeRaysAvx8(
  const __m256 dx, const __m256 dy, const __m256 dz, const int valid_mask, float *dirs_xyz
)
{
  alignas(32) float xs[8], ys[8], zs[8];
  _mm256_store_ps(xs, dx);
  _mm256_store_ps(ys, dy);
  _mm256_store_ps(zs, dz);
  for (int i = 0; i < 8; ++i)
  {
    const int off = i * 3;
    if (valid_mask & (1 << i))
    {
      dirs_xyz[off] = xs[i];
      dirs_xyz[off + 1] = ys[i];
      dirs_xyz[off + 2] = zs[i];
    }
    else
    {
      dirs_xyz[off] = 0.0f;
      dirs_xyz[off + 1] = 0.0f;
      dirs_xyz[off + 2] = 0.0f;
    }
  }
}

inline void normalizeAvx8(__m256 &dx, __m256 &dy, __m256 &dz, __m256 &valid)
{
  const __m256 norm_sq = _mm256_add_ps(
    _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)), _mm256_mul_ps(dz, dz)
  );
  const __m256 norm = _mm256_sqrt_ps(norm_sq);
  const __m256 eps = _mm256_set1_ps(1e-8f);
  const __m256 norm_ok = _mm256_cmp_ps(norm, eps, _CMP_GT_OQ);
  valid = _mm256_and_ps(valid, _mm256_and_ps(norm_ok, finiteMaskAvxInv(norm)));
  const __m256 safe_norm = selectAvxInv(_mm256_set1_ps(1.0f), norm, norm_ok);
  const __m256 inv_norm = _mm256_div_ps(_mm256_set1_ps(1.0f), safe_norm);
  dx = _mm256_mul_ps(dx, inv_norm);
  dy = _mm256_mul_ps(dy, inv_norm);
  dz = _mm256_mul_ps(dz, inv_norm);
}

}  // namespace

// ---------------------------------------------------------------------------
// Pinhole AVX8
// ---------------------------------------------------------------------------

int pixelToRayPinholeAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m256 x_d, y_d, valid;
  removeIntrinsicsAvx8(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m256 ux, uy;
  if (isDistortionNone(model.distortion))
  {
    ux = x_d;
    uy = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonAvx8(model.distortion, x_d, y_d, ux, uy, valid);
  }
  else
  {
    for (int i = 0; i < 24; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  __m256 dz = _mm256_set1_ps(1.0f);
  normalizeAvx8(ux, uy, dz, valid);

  const int vm = _mm256_movemask_ps(valid);
  storeRaysAvx8(ux, uy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// DoubleSphere AVX8
// ---------------------------------------------------------------------------

int pixelToRayDsphAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m256 x_d, y_d, valid;
  removeIntrinsicsAvx8(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m256 mx, my;
  if (isDistortionNone(model.distortion))
  {
    mx = x_d;
    my = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonAvx8(model.distortion, x_d, y_d, mx, my, valid);
  }
  else
  {
    for (int i = 0; i < 24; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 eps = _mm256_set1_ps(1e-8f);
  const __m256 alpha_v = _mm256_set1_ps(model.projection.alpha);
  const __m256 xi_v = _mm256_set1_ps(model.projection.xi);

  const __m256 r2 = _mm256_add_ps(_mm256_mul_ps(mx, mx), _mm256_mul_ps(my, my));

  const __m256 inner =
    _mm256_sub_ps(one, _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(two, alpha_v), one), r2));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(inner, _mm256_setzero_ps(), _CMP_GE_OQ));
  const __m256 sqrt_inner = _mm256_sqrt_ps(_mm256_max_ps(inner, _mm256_setzero_ps()));

  const __m256 alpha_sq = _mm256_mul_ps(alpha_v, alpha_v);
  const __m256 mz_num = _mm256_sub_ps(one, _mm256_mul_ps(alpha_sq, r2));

  const __m256 mz_den =
    _mm256_add_ps(_mm256_mul_ps(alpha_v, sqrt_inner), _mm256_sub_ps(one, alpha_v));
  const __m256 den_ok = _mm256_cmp_ps(absAvxInv(mz_den), eps, _CMP_GT_OQ);
  valid = _mm256_and_ps(valid, den_ok);
  const __m256 safe_den = selectAvxInv(one, mz_den, den_ok);
  const __m256 mz = _mm256_div_ps(mz_num, safe_den);

  const __m256 norm_sq = _mm256_add_ps(r2, _mm256_mul_ps(mz, mz));
  const __m256 xi_sq = _mm256_mul_ps(xi_v, xi_v);
  const __m256 inner_xi =
    _mm256_add_ps(_mm256_mul_ps(mz, mz), _mm256_mul_ps(_mm256_sub_ps(one, xi_sq), r2));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(inner_xi, _mm256_setzero_ps(), _CMP_GE_OQ));

  const __m256 sqrt_inner_xi = _mm256_sqrt_ps(_mm256_max_ps(inner_xi, _mm256_setzero_ps()));
  const __m256 scalar = _mm256_div_ps(
    _mm256_add_ps(_mm256_mul_ps(mz, xi_v), sqrt_inner_xi), _mm256_max_ps(norm_sq, eps)
  );

  __m256 dx = _mm256_mul_ps(scalar, mx);
  __m256 dy = _mm256_mul_ps(scalar, my);
  __m256 dz = _mm256_sub_ps(_mm256_mul_ps(scalar, mz), xi_v);

  normalizeAvx8(dx, dy, dz, valid);

  // Scalar parity: forward theta_max cap (see the SSE4 DoubleSphere kernel).
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m256 cos_tmax = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(dz, cos_tmax, _CMP_GE_OQ));
  }

  const int vm = _mm256_movemask_ps(valid);
  storeRaysAvx8(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// EUCM AVX8
// ---------------------------------------------------------------------------

int pixelToRayEucmAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m256 x_d, y_d, valid;
  removeIntrinsicsAvx8(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m256 mx, my;
  if (isDistortionNone(model.distortion))
  {
    mx = x_d;
    my = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonAvx8(model.distortion, x_d, y_d, mx, my, valid);
  }
  else
  {
    for (int i = 0; i < 24; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 two = _mm256_set1_ps(2.0f);
  const __m256 eps = _mm256_set1_ps(1e-8f);
  const __m256 alpha_v = _mm256_set1_ps(model.projection.alpha);
  const __m256 beta_v = _mm256_set1_ps(model.projection.beta);

  const __m256 r2 = _mm256_add_ps(_mm256_mul_ps(mx, mx), _mm256_mul_ps(my, my));

  const __m256 inner = _mm256_sub_ps(
    one, _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(two, alpha_v), one), beta_v), r2)
  );
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(inner, _mm256_setzero_ps(), _CMP_GE_OQ));
  const __m256 sqrt_inner = _mm256_sqrt_ps(_mm256_max_ps(inner, _mm256_setzero_ps()));

  const __m256 alpha_sq = _mm256_mul_ps(alpha_v, alpha_v);
  const __m256 mz_num = _mm256_sub_ps(one, _mm256_mul_ps(_mm256_mul_ps(alpha_sq, beta_v), r2));

  const __m256 mz_den =
    _mm256_add_ps(_mm256_mul_ps(alpha_v, sqrt_inner), _mm256_sub_ps(one, alpha_v));
  const __m256 den_ok = _mm256_cmp_ps(absAvxInv(mz_den), eps, _CMP_GT_OQ);
  valid = _mm256_and_ps(valid, den_ok);
  const __m256 safe_den = selectAvxInv(one, mz_den, den_ok);
  const __m256 mz = _mm256_div_ps(mz_num, safe_den);

  __m256 dx = mx;
  __m256 dy = my;
  __m256 dz = mz;

  normalizeAvx8(dx, dy, dz, valid);

  // Scalar parity: forward theta_max cap (see the SSE4 DoubleSphere kernel).
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m256 cos_tmax = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(dz, cos_tmax, _CMP_GE_OQ));
  }

  const int vm = _mm256_movemask_ps(valid);
  storeRaysAvx8(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// Omni AVX8
// ---------------------------------------------------------------------------

int pixelToRayOmniAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m256 x_d, y_d, valid;
  removeIntrinsicsAvx8(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  __m256 ux, uy;
  if (isDistortionNone(model.distortion))
  {
    ux = x_d;
    uy = y_d;
  }
  else if (!model.distortion.has_tilt)
  {
    undistortPlaneNewtonAvx8(model.distortion, x_d, y_d, ux, uy, valid);
  }
  else
  {
    for (int i = 0; i < 24; ++i) dirs_xyz[i] = 0.0f;
    return 0;
  }

  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 eps = _mm256_set1_ps(1e-8f);
  const __m256 xi_v = _mm256_set1_ps(model.projection.xi);
  const __m256 xi_sq = _mm256_mul_ps(xi_v, xi_v);

  const __m256 r2 = _mm256_add_ps(_mm256_mul_ps(ux, ux), _mm256_mul_ps(uy, uy));

  const __m256 inside = _mm256_add_ps(one, _mm256_mul_ps(_mm256_sub_ps(one, xi_sq), r2));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(inside, _mm256_setzero_ps(), _CMP_GE_OQ));
  const __m256 sqrt_term = _mm256_sqrt_ps(_mm256_max_ps(inside, _mm256_setzero_ps()));

  const __m256 denom = _mm256_add_ps(one, r2);
  const __m256 denom_ok = _mm256_cmp_ps(denom, eps, _CMP_GT_OQ);
  valid = _mm256_and_ps(valid, denom_ok);
  const __m256 safe_denom = selectAvxInv(one, denom, denom_ok);

  const __m256 lambda = _mm256_div_ps(_mm256_add_ps(xi_v, sqrt_term), safe_denom);
  valid = _mm256_and_ps(valid, finiteMaskAvxInv(lambda));

  __m256 dx = _mm256_mul_ps(lambda, ux);
  __m256 dy = _mm256_mul_ps(lambda, uy);
  __m256 dz = _mm256_sub_ps(lambda, xi_v);

  normalizeAvx8(dx, dy, dz, valid);

  // Scalar parity: forward theta_max cap (see the SSE4 DoubleSphere kernel).
  if (detail_impl::hasThetaMaxCap(model.projection.theta_max))
  {
    const __m256 cos_tmax = _mm256_set1_ps(std::cos(model.projection.theta_max));
    valid = _mm256_and_ps(valid, _mm256_cmp_ps(dz, cos_tmax, _CMP_GE_OQ));
  }

  const int vm = _mm256_movemask_ps(valid);
  storeRaysAvx8(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// Fisheye AVX8
// ---------------------------------------------------------------------------

int pixelToRayFisheyeAvx8(
  const CameraModel &model, const float *u_in, const float *v_in, float *dirs_xyz
)
{
  __m256 x_d, y_d, valid;
  removeIntrinsicsAvx8(model.intrinsics, u_in, v_in, x_d, y_d, valid);

  const __m256 r_d =
    _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(x_d, x_d), _mm256_mul_ps(y_d, y_d)));
  const __m256 eps = _mm256_set1_ps(1e-8f);
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 zero = _mm256_setzero_ps();

  __m256 theta;
  const auto dtype = model.distortion.type;
  if (dtype == DistortionModelType::EQUIDISTANT || dtype == DistortionModelType::NONE)
  {
    theta = r_d;
  }
  else if (dtype == DistortionModelType::EQUISOLID)
  {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 arg = _mm256_mul_ps(r_d, half);
    const __m256 clamped = _mm256_min_ps(_mm256_max_ps(arg, _mm256_set1_ps(-1.0f)), one);
    alignas(32) float arg_arr[8], theta_arr[8];
    _mm256_store_ps(arg_arr, clamped);
    for (int i = 0; i < 8; ++i) theta_arr[i] = 2.0f * std::asin(arg_arr[i]);
    theta = _mm256_load_ps(theta_arr);
  }
  else if (dtype == DistortionModelType::STEREOGRAPHIC)
  {
    alignas(32) float rd_arr[8], theta_arr[8];
    _mm256_store_ps(rd_arr, r_d);
    for (int i = 0; i < 8; ++i) theta_arr[i] = 2.0f * std::atan(rd_arr[i] * 0.5f);
    theta = _mm256_load_ps(theta_arr);
  }
  else if (dtype == DistortionModelType::ORTHOGRAPHIC)
  {
    const __m256 clamped = _mm256_min_ps(_mm256_max_ps(r_d, _mm256_set1_ps(-1.0f)), one);
    alignas(32) float rd_arr[8], theta_arr[8];
    _mm256_store_ps(rd_arr, clamped);
    for (int i = 0; i < 8; ++i) theta_arr[i] = std::asin(rd_arr[i]);
    theta = _mm256_load_ps(theta_arr);
  }
  else
  {
    theta = r_d;
    const auto &c = model.distortion.coeffs;
    const __m256 k1 = _mm256_set1_ps(c[0]);
    const __m256 k2 = _mm256_set1_ps(c[1]);
    const __m256 k3 = _mm256_set1_ps(c[2]);
    const __m256 k4 = _mm256_set1_ps(c[3]);

    for (int iter = 0; iter < 15; ++iter)
    {
      const __m256 t2 = _mm256_mul_ps(theta, theta);
      const __m256 t4 = _mm256_mul_ps(t2, t2);
      const __m256 t6 = _mm256_mul_ps(t4, t2);
      const __m256 t8 = _mm256_mul_ps(t4, t4);

      __m256 poly = _mm256_add_ps(one, _mm256_mul_ps(k1, t2));
      poly = _mm256_add_ps(poly, _mm256_mul_ps(k2, t4));
      poly = _mm256_add_ps(poly, _mm256_mul_ps(k3, t6));
      poly = _mm256_add_ps(poly, _mm256_mul_ps(k4, t8));
      const __m256 f_val = _mm256_sub_ps(_mm256_mul_ps(theta, poly), r_d);

      __m256 dpoly = _mm256_add_ps(one, _mm256_mul_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(k1, t2)));
      dpoly = _mm256_add_ps(dpoly, _mm256_mul_ps(_mm256_set1_ps(5.0f), _mm256_mul_ps(k2, t4)));
      dpoly = _mm256_add_ps(dpoly, _mm256_mul_ps(_mm256_set1_ps(7.0f), _mm256_mul_ps(k3, t6)));
      dpoly = _mm256_add_ps(dpoly, _mm256_mul_ps(_mm256_set1_ps(9.0f), _mm256_mul_ps(k4, t8)));

      const __m256 dpoly_abs_safe = _mm256_max_ps(absAvxInv(dpoly), eps);
      const __m256 dpoly_neg = _mm256_cmp_ps(dpoly, zero, _CMP_LT_OQ);
      const __m256 dpoly_safe =
        selectAvxInv(dpoly_abs_safe, _mm256_sub_ps(zero, dpoly_abs_safe), dpoly_neg);
      const __m256 delta = _mm256_div_ps(f_val, dpoly_safe);
      theta = _mm256_sub_ps(theta, delta);
      theta = _mm256_max_ps(theta, zero);
    }
    valid = _mm256_and_ps(valid, finiteMaskAvxInv(theta));
  }

  const __m256 t_max = _mm256_set1_ps(model.projection.theta_max - 1e-5f);
  valid = _mm256_and_ps(valid, finiteMaskAvxInv(theta));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(theta, t_max, _CMP_LE_OQ));
  valid = _mm256_and_ps(valid, _mm256_cmp_ps(theta, zero, _CMP_GE_OQ));

  const __m256 on_axis = _mm256_cmp_ps(r_d, eps, _CMP_LE_OQ);

  const __m256 safe_rd = _mm256_max_ps(r_d, eps);
  const __m256 inv_rd = _mm256_div_ps(one, safe_rd);
  const __m256 cos_phi = _mm256_mul_ps(x_d, inv_rd);
  const __m256 sin_phi = _mm256_mul_ps(y_d, inv_rd);

  alignas(32) float theta_arr[8], sin_arr[8], cos_arr[8];
  _mm256_store_ps(theta_arr, theta);
  for (int i = 0; i < 8; ++i)
  {
    sin_arr[i] = std::sin(theta_arr[i]);
    cos_arr[i] = std::cos(theta_arr[i]);
  }
  const __m256 sin_t = _mm256_load_ps(sin_arr);
  const __m256 cos_t = _mm256_load_ps(cos_arr);

  __m256 dx = _mm256_mul_ps(sin_t, cos_phi);
  __m256 dy = _mm256_mul_ps(sin_t, sin_phi);
  __m256 dz = cos_t;

  dx = selectAvxInv(dx, zero, on_axis);
  dy = selectAvxInv(dy, zero, on_axis);
  dz = selectAvxInv(dz, one, on_axis);

  normalizeAvx8(dx, dy, dz, valid);

  const int vm = _mm256_movemask_ps(valid);
  storeRaysAvx8(dx, dy, dz, vm, dirs_xyz);
  return vm;
}

// ---------------------------------------------------------------------------
// AVX2 batch drivers (AVX8 → SSE4 → scalar)
// ---------------------------------------------------------------------------

namespace
{

using Avx8InvFn = int (*)(const CameraModel &, const float *, const float *, float *);

int pixelToRayBatchAvx2Generic(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options, Avx8InvFn avx_fn, Sse4InvFn sse_fn,
  ScalarInvFn scalar_fn
)
{
  // See pixelToRayBatchSseGeneric: non-default solver options must not be
  // silently degraded to the fixed SIMD schedule.
  if (!isDefaultSolverOptionsForSimd(solver_options))
  {
    return pixelToRayBatchScalarOnly(
      model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, scalar_fn
    );
  }

  int valid_count = 0;
  int i = 0;
  const bool fast_simd_no_status = (statuses_out == nullptr) &&
                                   isDefaultSolverOptionsForSimd(solver_options) &&
                                   !requiresSimdRoundTripVerification(model);

  // AVX8 chunk: 8 pixels at a time
  const int avx_count = count & ~7;
  for (; i < avx_count; i += 8)
  {
    float *out = dirs_xyz + i * 3;
    const int vm = avx_fn(model, u_in + i, v_in + i, out);

    if (fast_simd_no_status)
    {
      valid_count += __builtin_popcount(vm & 0xFF);
    }
    else
    {
      for (int j = 0; j < 8; ++j)
      {
        const int idx = i + j;
        if ((vm >> j) & 1)
        {
          const int off = idx * 3;
          const Eigen::Vector3f dir(dirs_xyz[off], dirs_xyz[off + 1], dirs_xyz[off + 2]);
          const PixelResult fwd = rayToPixel(model, dir);
          const float du = fwd.pixel.u - u_in[idx];
          const float dv = fwd.pixel.v - v_in[idx];
          if (fwd.status == StatusCode::OK && (du * du + dv * dv) < kStatusVerifyThresholdSq)
          {
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = StatusCode::OK;
            }
            ++valid_count;
          }
          else
          {
            const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = sr.status;
            }
            if (sr.status == StatusCode::OK)
            {
              dirs_xyz[off] = sr.ray.direction.x();
              dirs_xyz[off + 1] = sr.ray.direction.y();
              dirs_xyz[off + 2] = sr.ray.direction.z();
              ++valid_count;
            }
            else
            {
              dirs_xyz[off] = 0.0f;
              dirs_xyz[off + 1] = 0.0f;
              dirs_xyz[off + 2] = 0.0f;
            }
          }
        }
        else
        {
          const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
          if (statuses_out != nullptr)
          {
            statuses_out[idx] = sr.status;
          }
          if (sr.status == StatusCode::OK)
          {
            const int off = idx * 3;
            dirs_xyz[off] = sr.ray.direction.x();
            dirs_xyz[off + 1] = sr.ray.direction.y();
            dirs_xyz[off + 2] = sr.ray.direction.z();
            ++valid_count;
          }
          else
          {
            const int off = idx * 3;
            dirs_xyz[off] = 0.0f;
            dirs_xyz[off + 1] = 0.0f;
            dirs_xyz[off + 2] = 0.0f;
          }
        }
      }
    }
  }

  // SSE4 chunk: 4 pixels
  const int sse_end = count & ~3;
  for (; i < sse_end; i += 4)
  {
    float *out = dirs_xyz + i * 3;
    const int vm = sse_fn(model, u_in + i, v_in + i, out);

    if (fast_simd_no_status)
    {
      valid_count += __builtin_popcount(vm & 0xF);
    }
    else
    {
      for (int j = 0; j < 4; ++j)
      {
        const int idx = i + j;
        if ((vm >> j) & 1)
        {
          const int off = idx * 3;
          const Eigen::Vector3f dir(dirs_xyz[off], dirs_xyz[off + 1], dirs_xyz[off + 2]);
          const PixelResult fwd = rayToPixel(model, dir);
          const float du = fwd.pixel.u - u_in[idx];
          const float dv = fwd.pixel.v - v_in[idx];
          if (fwd.status == StatusCode::OK && (du * du + dv * dv) < kStatusVerifyThresholdSq)
          {
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = StatusCode::OK;
            }
            ++valid_count;
          }
          else
          {
            const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
            if (statuses_out != nullptr)
            {
              statuses_out[idx] = sr.status;
            }
            if (sr.status == StatusCode::OK)
            {
              dirs_xyz[off] = sr.ray.direction.x();
              dirs_xyz[off + 1] = sr.ray.direction.y();
              dirs_xyz[off + 2] = sr.ray.direction.z();
              ++valid_count;
            }
            else
            {
              dirs_xyz[off] = 0.0f;
              dirs_xyz[off + 1] = 0.0f;
              dirs_xyz[off + 2] = 0.0f;
            }
          }
        }
        else
        {
          const RayResult sr = scalar_fn(model, Pixel2{u_in[idx], v_in[idx]}, solver_options);
          if (statuses_out != nullptr)
          {
            statuses_out[idx] = sr.status;
          }
          if (sr.status == StatusCode::OK)
          {
            const int off = idx * 3;
            dirs_xyz[off] = sr.ray.direction.x();
            dirs_xyz[off + 1] = sr.ray.direction.y();
            dirs_xyz[off + 2] = sr.ray.direction.z();
            ++valid_count;
          }
          else
          {
            const int off = idx * 3;
            dirs_xyz[off] = 0.0f;
            dirs_xyz[off + 1] = 0.0f;
            dirs_xyz[off + 2] = 0.0f;
          }
        }
      }
    }
  }

  // scalar tail
  for (; i < count; ++i)
  {
    const RayResult sr = scalar_fn(model, Pixel2{u_in[i], v_in[i]}, solver_options);
    const int off = i * 3;
    storeScalarRayResult(sr, off, dirs_xyz);
    if (statuses_out != nullptr) statuses_out[i] = sr.status;
    if (sr.status == StatusCode::OK) ++valid_count;
  }

  return valid_count;
}

}  // namespace

int pixelToRayBatchPinholeAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    return pixelToRayBatchPinholeSse(
      model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options
    );
  }
  return pixelToRayBatchAvx2Generic(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayPinholeAvx8,
    &pixelToRayPinholeSse4, &pinhole::pixelToRay
  );
}

int pixelToRayBatchDsphAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    return pixelToRayBatchDsphSse(model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options);
  }
  return pixelToRayBatchAvx2Generic(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayDsphAvx8,
    &pixelToRayDsphSse4, &double_sphere::pixelToRay
  );
}

int pixelToRayBatchEucmAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    return pixelToRayBatchEucmSse(model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options);
  }
  return pixelToRayBatchAvx2Generic(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayEucmAvx8,
    &pixelToRayEucmSse4, &eucm::pixelToRay
  );
}

int pixelToRayBatchOmniAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.has_tilt)
  {
    return pixelToRayBatchOmniSse(model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options);
  }
  return pixelToRayBatchAvx2Generic(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayOmniAvx8,
    &pixelToRayOmniSse4, &omnidirectional::pixelToRay
  );
}

int pixelToRayBatchFisheyeAvx2(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (model.distortion.type == DistortionModelType::KB8)
  {
    return pixelToRayBatchScalarOnly(
      model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &fisheye::pixelToRay
    );
  }
  return pixelToRayBatchAvx2Generic(
    model, u_in, v_in, count, dirs_xyz, statuses_out, solver_options, &pixelToRayFisheyeAvx8,
    &pixelToRayFisheyeSse4, &fisheye::pixelToRay
  );
}

#endif  // CAMXIOM_HAS_AVX2

}  // namespace camxiom::detail
