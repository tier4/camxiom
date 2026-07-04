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

#include "camxiom/batch64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "detail/projection_models.hpp"
#include "projection64/internal.hpp"  // validateCameraModelQuery64

#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include "batch/batch_parallel.hpp"

#include <arm_neon.h>
#endif

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

namespace camxiom
{
namespace
{

using PixelFn64 = PixelResult64 (*)(const CameraModel64 &, const Eigen::Vector3d &);
using RayFn64 = RayResult64 (*)(const CameraModel64 &, const Pixel2d &, const SolverOptions64 &);

#if defined(__AVX2__)
bool isPinholeUndistorted64(const CameraModel64 &model)
{
  return model.projection.type == ProjectionModelType::PINHOLE &&
         model.distortion.type == DistortionModelType::NONE &&
         model.distortion.space == DistortionSpace::NONE && model.distortion.count == 0U &&
         !model.distortion.is_rational && !model.distortion.has_thin_prism &&
         !model.distortion.has_tilt;
}

inline __m256d absAvxD(const __m256d values)
{
  const __m256i sign_mask = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
  return _mm256_and_pd(values, _mm256_castsi256_pd(sign_mask));
}

inline __m256d finiteMaskAvxD(const __m256d values)
{
  const __m256d max_value = _mm256_set1_pd((std::numeric_limits<double>::max)());
  return _mm256_cmp_pd(absAvxD(values), max_value, _CMP_LE_OQ);
}

inline int rayToPixelPinholeUndistortedAvx4(
  const CameraModel64 &model, const double *rays_xyz, double *u_out, double *v_out
)
{
  const __m256d xs = _mm256_set_pd(rays_xyz[9], rays_xyz[6], rays_xyz[3], rays_xyz[0]);
  const __m256d ys = _mm256_set_pd(rays_xyz[10], rays_xyz[7], rays_xyz[4], rays_xyz[1]);
  const __m256d zs = _mm256_set_pd(rays_xyz[11], rays_xyz[8], rays_xyz[5], rays_xyz[2]);

  const __m256d zero = _mm256_setzero_pd();
  const __m256d z_positive = _mm256_cmp_pd(zs, zero, _CMP_GT_OQ);
  const __m256d finite_xyz =
    _mm256_and_pd(finiteMaskAvxD(xs), _mm256_and_pd(finiteMaskAvxD(ys), finiteMaskAvxD(zs)));
  __m256d valid_mask = _mm256_and_pd(z_positive, finite_xyz);
  if (_mm256_movemask_pd(valid_mask) == 0)
  {
    _mm256_storeu_pd(u_out, zero);
    _mm256_storeu_pd(v_out, zero);
    return 0;
  }

  const __m256d inv_z = _mm256_div_pd(_mm256_set1_pd(1.0), zs);
  const __m256d x_n = _mm256_mul_pd(xs, inv_z);
  const __m256d y_n = _mm256_mul_pd(ys, inv_z);

  const __m256d fx = _mm256_set1_pd(model.intrinsics.fx);
  const __m256d fy = _mm256_set1_pd(model.intrinsics.fy);
  const __m256d cx = _mm256_set1_pd(model.intrinsics.cx);
  const __m256d cy = _mm256_set1_pd(model.intrinsics.cy);
  const __m256d sk = _mm256_set1_pd(model.intrinsics.skew);

  __m256d u = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(fx, x_n), _mm256_mul_pd(sk, y_n)), cx);
  __m256d v = _mm256_add_pd(_mm256_mul_pd(fy, y_n), cy);
  valid_mask = _mm256_and_pd(valid_mask, _mm256_and_pd(finiteMaskAvxD(u), finiteMaskAvxD(v)));

  u = _mm256_and_pd(u, valid_mask);
  v = _mm256_and_pd(v, valid_mask);
  _mm256_storeu_pd(u_out, u);
  _mm256_storeu_pd(v_out, v);
  return _mm256_movemask_pd(valid_mask);
}

inline int pixelToRayPinholeUndistortedAvx4(
  const CameraModel64 &model, const double *u_in, const double *v_in, double *dx_out,
  double *dy_out, double *dz_out
)
{
  const __m256d u = _mm256_loadu_pd(u_in);
  const __m256d v = _mm256_loadu_pd(v_in);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d eps = _mm256_set1_pd(1e-15);

  const __m256d finite_uv = _mm256_and_pd(finiteMaskAvxD(u), finiteMaskAvxD(v));

  const __m256d fy_inv = _mm256_set1_pd(1.0 / model.intrinsics.fy);
  const __m256d fx_inv = _mm256_set1_pd(1.0 / model.intrinsics.fx);
  const __m256d cx = _mm256_set1_pd(model.intrinsics.cx);
  const __m256d cy = _mm256_set1_pd(model.intrinsics.cy);
  const __m256d sk = _mm256_set1_pd(model.intrinsics.skew);

  const __m256d y = _mm256_mul_pd(_mm256_sub_pd(v, cy), fy_inv);
  const __m256d x =
    _mm256_mul_pd(_mm256_sub_pd(_mm256_sub_pd(u, cx), _mm256_mul_pd(sk, y)), fx_inv);
  const __m256d finite_xy = _mm256_and_pd(finiteMaskAvxD(x), finiteMaskAvxD(y));

  const __m256d norm2 = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)), one);
  const __m256d norm = _mm256_sqrt_pd(norm2);
  const __m256d norm_valid =
    _mm256_and_pd(finiteMaskAvxD(norm), _mm256_cmp_pd(norm, eps, _CMP_GT_OQ));
  __m256d valid_mask = _mm256_and_pd(finite_uv, _mm256_and_pd(finite_xy, norm_valid));

  const __m256d inv_norm = _mm256_div_pd(one, norm);
  __m256d dx = _mm256_mul_pd(x, inv_norm);
  __m256d dy = _mm256_mul_pd(y, inv_norm);
  __m256d dz = inv_norm;
  valid_mask = _mm256_and_pd(
    valid_mask,
    _mm256_and_pd(finiteMaskAvxD(dx), _mm256_and_pd(finiteMaskAvxD(dy), finiteMaskAvxD(dz)))
  );

  dx = _mm256_and_pd(dx, valid_mask);
  dy = _mm256_and_pd(dy, valid_mask);
  dz = _mm256_and_pd(dz, valid_mask);
  _mm256_storeu_pd(dx_out, dx);
  _mm256_storeu_pd(dy_out, dy);
  _mm256_storeu_pd(dz_out, dz);
  return _mm256_movemask_pd(valid_mask);
}
#endif

#if defined(__aarch64__)
inline uint64x2_t finiteMaskNeonD(const float64x2_t values)
{
  const float64x2_t max_value = vdupq_n_f64((std::numeric_limits<double>::max)());
  return vcleq_f64(vabsq_f64(values), max_value);
}

/// 2-wide NEON pinhole forward projection: plane RadTan distortion (radial +
/// rational + tangential + thin prism — the same families and masking as the
/// float rayToPixelPinholeSse4) followed by the intrinsics. Lanes the SIMD
/// marks invalid are recomputed through the scalar projection by the caller,
/// so statuses stay exact. Tilt models must not reach this kernel.
///
/// Returns bitmask of valid points (bits 0-1).
inline int rayToPixelPinholeNeon2(
  const CameraModel64 &model, const double *rays_xyz, double *u_out, double *v_out
)
{
  const float64x2_t xs = {rays_xyz[0], rays_xyz[3]};
  const float64x2_t ys = {rays_xyz[1], rays_xyz[4]};
  const float64x2_t zs = {rays_xyz[2], rays_xyz[5]};

  const float64x2_t zero = vdupq_n_f64(0.0);
  const float64x2_t one = vdupq_n_f64(1.0);
  const float64x2_t two = vdupq_n_f64(2.0);

  uint64x2_t valid = vcgtq_f64(zs, zero);
  valid = vandq_u64(valid, finiteMaskNeonD(xs));
  valid = vandq_u64(valid, finiteMaskNeonD(ys));
  valid = vandq_u64(valid, finiteMaskNeonD(zs));

  const float64x2_t inv_z = vdivq_f64(one, zs);
  const float64x2_t x_n = vmulq_f64(xs, inv_z);
  const float64x2_t y_n = vmulq_f64(ys, inv_z);
  valid = vandq_u64(valid, finiteMaskNeonD(x_n));
  valid = vandq_u64(valid, finiteMaskNeonD(y_n));

  const float64x2_t xx = vmulq_f64(x_n, x_n);
  const float64x2_t yy = vmulq_f64(y_n, y_n);
  const float64x2_t xy = vmulq_f64(x_n, y_n);
  const float64x2_t r2 = vaddq_f64(xx, yy);
  const float64x2_t r4 = vmulq_f64(r2, r2);
  const float64x2_t r6 = vmulq_f64(r4, r2);

  const auto &c = model.distortion.coeffs;
  const float64x2_t k1 = vdupq_n_f64(c[0]);
  const float64x2_t k2 = vdupq_n_f64(c[1]);
  const float64x2_t p1 = vdupq_n_f64(c[2]);
  const float64x2_t p2 = vdupq_n_f64(c[3]);
  const float64x2_t k3 = vdupq_n_f64(c[4]);

  // radial_num = 1 + k1*r² + k2*r⁴ + k3*r⁶
  float64x2_t rad_num = vfmaq_f64(one, k1, r2);
  rad_num = vfmaq_f64(rad_num, k2, r4);
  rad_num = vfmaq_f64(rad_num, k3, r6);

  float64x2_t radial = rad_num;
  if (model.distortion.is_rational)
  {
    const float64x2_t k4 = vdupq_n_f64(c[5]);
    const float64x2_t k5 = vdupq_n_f64(c[6]);
    const float64x2_t k6 = vdupq_n_f64(c[7]);
    float64x2_t rad_den = vfmaq_f64(one, k4, r2);
    rad_den = vfmaq_f64(rad_den, k5, r4);
    rad_den = vfmaq_f64(rad_den, k6, r6);
    const float64x2_t eps = vdupq_n_f64(1e-8);
    const uint64x2_t den_valid =
      vandq_u64(finiteMaskNeonD(rad_den), vcgtq_f64(vabsq_f64(rad_den), eps));
    valid = vandq_u64(valid, den_valid);
    const float64x2_t safe_den = vbslq_f64(den_valid, rad_den, one);
    radial = vdivq_f64(rad_num, safe_den);
  }

  // x_tan = 2*p1*x*y + p2*(r² + 2*x²), y_tan = p1*(r² + 2*y²) + 2*p2*x*y
  const float64x2_t x_tan =
    vfmaq_f64(vmulq_f64(vmulq_f64(two, p1), xy), p2, vfmaq_f64(r2, two, xx));
  const float64x2_t y_tan =
    vfmaq_f64(vmulq_f64(vmulq_f64(two, p2), xy), p1, vfmaq_f64(r2, two, yy));

  float64x2_t x_d = vfmaq_f64(x_tan, x_n, radial);
  float64x2_t y_d = vfmaq_f64(y_tan, y_n, radial);

  if (model.distortion.has_thin_prism)
  {
    const float64x2_t s1 = vdupq_n_f64(c[8]);
    const float64x2_t s2 = vdupq_n_f64(c[9]);
    const float64x2_t s3 = vdupq_n_f64(c[10]);
    const float64x2_t s4 = vdupq_n_f64(c[11]);
    x_d = vaddq_f64(x_d, vfmaq_f64(vmulq_f64(s1, r2), s2, r4));
    y_d = vaddq_f64(y_d, vfmaq_f64(vmulq_f64(s3, r2), s4, r4));
  }

  // u = fx*x_d + skew*y_d + cx, v = fy*y_d + cy
  const float64x2_t fx = vdupq_n_f64(model.intrinsics.fx);
  const float64x2_t fy = vdupq_n_f64(model.intrinsics.fy);
  const float64x2_t cx = vdupq_n_f64(model.intrinsics.cx);
  const float64x2_t cy = vdupq_n_f64(model.intrinsics.cy);
  const float64x2_t sk = vdupq_n_f64(model.intrinsics.skew);

  float64x2_t u = vfmaq_f64(vfmaq_f64(cx, sk, y_d), fx, x_d);
  float64x2_t v = vfmaq_f64(cy, fy, y_d);
  valid = vandq_u64(valid, vandq_u64(finiteMaskNeonD(u), finiteMaskNeonD(v)));

  u = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(u), valid));
  v = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v), valid));
  vst1q_f64(u_out, u);
  vst1q_f64(v_out, v);

  return (vgetq_lane_u64(valid, 0) != 0U ? 1 : 0) | (vgetq_lane_u64(valid, 1) != 0U ? 2 : 0);
}
#endif

PixelFn64 resolveForwardFn64(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::rayToPixel64;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::rayToPixel64;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::rayToPixel64;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::rayToPixel64;
    case ProjectionModelType::EUCM:
      return &eucm::rayToPixel64;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

RayFn64 resolveInverseFn64(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::pixelToRay64;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::pixelToRay64;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::pixelToRay64;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::pixelToRay64;
    case ProjectionModelType::EUCM:
      return &eucm::pixelToRay64;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

}  // namespace

int rayToPixelBatch64(
  const CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix3Xd> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xd> pixels_out, StatusCode *statuses_out
)
{
  if (ray_directions.rows() != 3 || pixels_out.rows() != 2)
  {
    pixels_out.setZero();
    const int count = static_cast<int>(ray_directions.cols());
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const int count = static_cast<int>(ray_directions.cols());
  if (count <= 0) return 0;
  if (pixels_out.cols() != count)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail64::validateCameraModelQuery64(model);
  if (validation != StatusCode::OK)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = validation;
    }
    return -1;
  }

  const PixelFn64 fn = resolveForwardFn64(model.projection.type);
  if (fn == nullptr)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_MODEL;
    }
    return -1;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const PixelResult64 result = fn(model, ray_directions.col(i));
    pixels_out(0, i) = result.pixel.u;
    pixels_out(1, i) = result.pixel.v;
    if (statuses_out != nullptr) statuses_out[i] = result.status;
    if (result.status == StatusCode::OK) ++valid_count;
  }
  return valid_count;
}

int rayToPixelBatch64(
  const CameraModel64 &model, const double *rays_xyz, const int count, double *u_out, double *v_out,
  StatusCode *statuses_out
)
{
  if (count < 0) return -1;
  if (count == 0) return 0;
  if (rays_xyz == nullptr || u_out == nullptr || v_out == nullptr)
  {
    if (u_out != nullptr)
    {
      for (int i = 0; i < count; ++i) u_out[i] = 0.0;
    }
    if (v_out != nullptr)
    {
      for (int i = 0; i < count; ++i) v_out[i] = 0.0;
    }
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail64::validateCameraModelQuery64(model);
  if (validation != StatusCode::OK)
  {
    for (int i = 0; i < count; ++i)
    {
      u_out[i] = 0.0;
      v_out[i] = 0.0;
      if (statuses_out != nullptr) statuses_out[i] = validation;
    }
    return 0;
  }

  const PixelFn64 fn = resolveForwardFn64(model.projection.type);
  if (fn == nullptr)
  {
    for (int i = 0; i < count; ++i)
    {
      u_out[i] = 0.0;
      v_out[i] = 0.0;
      if (statuses_out != nullptr) statuses_out[i] = StatusCode::INVALID_MODEL;
    }
    return 0;
  }

#if defined(__AVX2__)
  if (isPinholeUndistorted64(model))
  {
    const int simd_count = count & ~3;
    int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
    for (int i = 0; i < simd_count; i += 4)
    {
      double u4[4];
      double v4[4];
      const int mask = rayToPixelPinholeUndistortedAvx4(model, rays_xyz + i * 3, u4, v4);
      for (int j = 0; j < 4; ++j)
      {
        const int index = i + j;
        const bool simd_ok = ((mask >> j) & 1) != 0;
        if (simd_ok)
        {
          u_out[index] = u4[j];
          v_out[index] = v4[j];
          if (statuses_out != nullptr)
          {
            statuses_out[index] = StatusCode::OK;
          }
          ++valid_count;
        }
        else
        {
          const int off = index * 3;
          const Eigen::Vector3d ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
          const PixelResult64 scalar_result = pinhole::rayToPixel64(model, ray);
          u_out[index] = scalar_result.pixel.u;
          v_out[index] = scalar_result.pixel.v;
          if (statuses_out != nullptr)
          {
            statuses_out[index] = scalar_result.status;
          }
          if (scalar_result.status == StatusCode::OK)
          {
            ++valid_count;
          }
        }
      }
    }
    for (int i = simd_count; i < count; ++i)
    {
      const int off = i * 3;
      const Eigen::Vector3d ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
      const PixelResult64 result = pinhole::rayToPixel64(model, ray);
      u_out[i] = result.pixel.u;
      v_out[i] = result.pixel.v;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = result.status;
      }
      if (result.status == StatusCode::OK)
      {
        ++valid_count;
      }
    }
    return valid_count;
  }
#endif

#if defined(__aarch64__)
  // 2-wide NEON pinhole forward (whole RadTan plane family; tilt falls back
  // to the scalar loop below). Measured 1.9x single-threaded / 1.6x at 12
  // threads over the scalar+OpenMP path on a Jetson Orin; SIMD-invalid lanes
  // are recomputed through the scalar projection, so statuses stay exact.
  if (model.projection.type == ProjectionModelType::PINHOLE && !model.distortion.has_tilt)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      const double *rays = rays_xyz + 3 * begin;
      double *u_o = u_out + begin;
      double *v_o = v_out + begin;
      StatusCode *st = detail::offsetStatuses(statuses_out, begin);
      const int simd_count = len & ~1;
      int vc = 0;
      for (int i = 0; i < simd_count; i += 2)
      {
        double u2[2];
        double v2[2];
        const int mask = rayToPixelPinholeNeon2(model, rays + i * 3, u2, v2);
        for (int j = 0; j < 2; ++j)
        {
          const int index = i + j;
          if ((mask >> j) & 1)
          {
            u_o[index] = u2[j];
            v_o[index] = v2[j];
            if (st != nullptr)
            {
              st[index] = StatusCode::OK;
            }
            ++vc;
          }
          else
          {
            const int off = index * 3;
            const Eigen::Vector3d ray(rays[off], rays[off + 1], rays[off + 2]);
            const PixelResult64 sr = pinhole::rayToPixel64(model, ray);
            u_o[index] = sr.pixel.u;
            v_o[index] = sr.pixel.v;
            if (st != nullptr)
            {
              st[index] = sr.status;
            }
            if (sr.status == StatusCode::OK)
            {
              ++vc;
            }
          }
        }
      }
      for (int i = simd_count; i < len; ++i)
      {
        const int off = i * 3;
        const Eigen::Vector3d ray(rays[off], rays[off + 1], rays[off + 2]);
        const PixelResult64 sr = pinhole::rayToPixel64(model, ray);
        u_o[i] = sr.pixel.u;
        v_o[i] = sr.pixel.v;
        if (st != nullptr)
        {
          st[i] = sr.status;
        }
        if (sr.status == StatusCode::OK)
        {
          ++vc;
        }
      }
      return vc;
    });
  }
#endif

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const int off = i * 3;
    const Eigen::Vector3d ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
    const PixelResult64 result = fn(model, ray);
    u_out[i] = result.pixel.u;
    v_out[i] = result.pixel.v;
    if (statuses_out != nullptr) statuses_out[i] = result.status;
    if (result.status == StatusCode::OK) ++valid_count;
  }
  return valid_count;
}

int pixelToRayBatch64(
  const CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix2Xd> &pixels,
  Eigen::Ref<Eigen::Matrix3Xd> directions_out, StatusCode *statuses_out,
  const SolverOptions64 &solver_options
)
{
  if (pixels.rows() != 2 || directions_out.rows() != 3)
  {
    directions_out.setZero();
    const int count = static_cast<int>(pixels.cols());
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const int count = static_cast<int>(pixels.cols());
  if (count <= 0) return 0;
  if (directions_out.cols() != count)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail64::validateCameraModelQuery64(model);
  if (validation != StatusCode::OK)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = validation;
    }
    return -1;
  }

  const RayFn64 fn = resolveInverseFn64(model.projection.type);
  if (fn == nullptr)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_MODEL;
    }
    return -1;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const RayResult64 result = fn(model, Pixel2d{pixels(0, i), pixels(1, i)}, solver_options);
    if (result.status == StatusCode::OK)
    {
      directions_out.col(i) = result.ray.direction;
    }
    else
    {
      directions_out.col(i).setZero();
    }
    if (statuses_out != nullptr) statuses_out[i] = result.status;
    if (result.status == StatusCode::OK) ++valid_count;
  }
  return valid_count;
}

int pixelToRayBatch64(
  const CameraModel64 &model, const double *u_in, const double *v_in, const int count,
  double *dirs_xyz, StatusCode *statuses_out, const SolverOptions64 &solver_options
)
{
  if (count < 0) return -1;
  if (count == 0) return 0;
  if (u_in == nullptr || v_in == nullptr || dirs_xyz == nullptr)
  {
    if (dirs_xyz != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        const int off = i * 3;
        dirs_xyz[off] = 0.0;
        dirs_xyz[off + 1] = 0.0;
        dirs_xyz[off + 2] = 0.0;
      }
    }
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail64::validateCameraModelQuery64(model);
  if (validation != StatusCode::OK)
  {
    for (int i = 0; i < count; ++i)
    {
      const int off = i * 3;
      dirs_xyz[off] = 0.0;
      dirs_xyz[off + 1] = 0.0;
      dirs_xyz[off + 2] = 0.0;
      if (statuses_out != nullptr) statuses_out[i] = validation;
    }
    return 0;
  }

  const RayFn64 fn = resolveInverseFn64(model.projection.type);
  if (fn == nullptr)
  {
    for (int i = 0; i < count; ++i)
    {
      const int off = i * 3;
      dirs_xyz[off] = 0.0;
      dirs_xyz[off + 1] = 0.0;
      dirs_xyz[off + 2] = 0.0;
      if (statuses_out != nullptr) statuses_out[i] = StatusCode::INVALID_MODEL;
    }
    return 0;
  }

#if defined(__AVX2__)
  if (isPinholeUndistorted64(model))
  {
    const int simd_count = count & ~3;
    int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
    for (int i = 0; i < simd_count; i += 4)
    {
      double dx4[4];
      double dy4[4];
      double dz4[4];
      const int mask = pixelToRayPinholeUndistortedAvx4(model, u_in + i, v_in + i, dx4, dy4, dz4);
      for (int j = 0; j < 4; ++j)
      {
        const int index = i + j;
        const int off = index * 3;
        const bool simd_ok = ((mask >> j) & 1) != 0;
        if (simd_ok)
        {
          dirs_xyz[off] = dx4[j];
          dirs_xyz[off + 1] = dy4[j];
          dirs_xyz[off + 2] = dz4[j];
          if (statuses_out != nullptr)
          {
            statuses_out[index] = StatusCode::OK;
          }
          ++valid_count;
        }
        else
        {
          const RayResult64 scalar_result =
            pinhole::pixelToRay64(model, Pixel2d{u_in[index], v_in[index]}, solver_options);
          if (scalar_result.status == StatusCode::OK)
          {
            dirs_xyz[off] = scalar_result.ray.direction.x();
            dirs_xyz[off + 1] = scalar_result.ray.direction.y();
            dirs_xyz[off + 2] = scalar_result.ray.direction.z();
          }
          else
          {
            dirs_xyz[off] = 0.0;
            dirs_xyz[off + 1] = 0.0;
            dirs_xyz[off + 2] = 0.0;
          }
          if (statuses_out != nullptr)
          {
            statuses_out[index] = scalar_result.status;
          }
          if (scalar_result.status == StatusCode::OK)
          {
            ++valid_count;
          }
        }
      }
    }
    for (int i = simd_count; i < count; ++i)
    {
      const RayResult64 result =
        pinhole::pixelToRay64(model, Pixel2d{u_in[i], v_in[i]}, solver_options);
      const int off = i * 3;
      if (result.status == StatusCode::OK)
      {
        dirs_xyz[off] = result.ray.direction.x();
        dirs_xyz[off + 1] = result.ray.direction.y();
        dirs_xyz[off + 2] = result.ray.direction.z();
      }
      else
      {
        dirs_xyz[off] = 0.0;
        dirs_xyz[off + 1] = 0.0;
        dirs_xyz[off + 2] = 0.0;
      }
      if (statuses_out != nullptr)
      {
        statuses_out[i] = result.status;
      }
      if (result.status == StatusCode::OK)
      {
        ++valid_count;
      }
    }
    return valid_count;
  }
#endif

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const RayResult64 result = fn(model, Pixel2d{u_in[i], v_in[i]}, solver_options);
    const int off = i * 3;
    if (result.status == StatusCode::OK)
    {
      dirs_xyz[off] = result.ray.direction.x();
      dirs_xyz[off + 1] = result.ray.direction.y();
      dirs_xyz[off + 2] = result.ray.direction.z();
    }
    else
    {
      dirs_xyz[off] = 0.0;
      dirs_xyz[off + 1] = 0.0;
      dirs_xyz[off + 2] = 0.0;
    }
    if (statuses_out != nullptr) statuses_out[i] = result.status;
    if (result.status == StatusCode::OK) ++valid_count;
  }
  return valid_count;
}

}  // namespace camxiom
