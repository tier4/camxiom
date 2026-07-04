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

#include "camxiom/batch.hpp"
#include "camxiom/projection.hpp"
#include "detail/internal.hpp"  // validateCameraModelQuery
#include "detail/projection_models.hpp"
#include "detail/simd_avx2.hpp"
#include "detail/simd_ds_eucm.hpp"
#include "detail/simd_fisheye.hpp"
#include "detail/simd_inverse.hpp"
#include "detail/simd_pinhole.hpp"

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

#include "batch/batch_parallel.hpp"

namespace camxiom
{
namespace
{

using PixelFn = PixelResult (*)(const CameraModel &, const Eigen::Vector3f &);
using RayFn = RayResult (*)(const CameraModel &, const Pixel2 &, const SolverOptions &);

bool supportsFisheyeSseDistortion(const DistortionModelType type)
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

PixelFn resolveForwardFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::rayToPixel;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::rayToPixel;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::rayToPixel;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::rayToPixel;
    case ProjectionModelType::EUCM:
      return &eucm::rayToPixel;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

RayFn resolveInverseFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::pixelToRay;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::pixelToRay;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::pixelToRay;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::pixelToRay;
    case ProjectionModelType::EUCM:
      return &eucm::pixelToRay;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

}  // namespace

int rayToPixelBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xf> pixels_out, StatusCode *statuses_out
)
{
  if (ray_directions.rows() != 3 || pixels_out.rows() != 2)
  {
    pixels_out.setZero();
    const int count = static_cast<int>(ray_directions.cols());
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_INPUT;
      }
    }
    return -1;
  }

  const int count = static_cast<int>(ray_directions.cols());
  if (count <= 0)
  {
    return 0;
  }
  if (pixels_out.cols() != count)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_INPUT;
      }
    }
    return -1;
  }

  const StatusCode validation = detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = validation;
      }
    }
    return -1;
  }

  const PixelFn fn = resolveForwardFn(model.projection.type);
  if (fn == nullptr)
  {
    pixels_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_MODEL;
      }
    }
    return -1;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const PixelResult result = fn(model, ray_directions.col(i));
    pixels_out(0, i) = result.pixel.u;
    pixels_out(1, i) = result.pixel.v;
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

int rayToPixelBatch(
  const CameraModel &model, const float *rays_xyz, const int count, float *u_out, float *v_out,
  StatusCode *statuses_out
)
{
  if (count < 0)
  {
    return -1;
  }
  if (count == 0)
  {
    return 0;
  }
  if (rays_xyz == nullptr || u_out == nullptr || v_out == nullptr)
  {
    if (u_out != nullptr)
    {
      for (int i = 0; i < count; ++i) u_out[i] = 0.0f;
    }
    if (v_out != nullptr)
    {
      for (int i = 0; i < count; ++i) v_out[i] = 0.0f;
    }
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    for (int i = 0; i < count; ++i)
    {
      u_out[i] = 0.0f;
      v_out[i] = 0.0f;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = validation;
      }
    }
    return 0;
  }

#ifdef CAMXIOM_HAS_AVX2
  if (model.projection.type == ProjectionModelType::PINHOLE)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchPinholeAvx2(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::FISHEYE_THETA &&
      supportsFisheyeSseDistortion(model.distortion.type))
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchFisheyeAvx2(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::OMNIDIRECTIONAL)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchOmniAvx2(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::DOUBLE_SPHERE)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchDsphAvx2(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::EUCM)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchEucmAvx2(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
#elif defined(CAMXIOM_HAS_SSE2)
  if (model.projection.type == ProjectionModelType::PINHOLE)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchPinholeSse(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
#endif

#ifdef CAMXIOM_HAS_SSE2
  if (model.projection.type == ProjectionModelType::FISHEYE_THETA &&
      supportsFisheyeSseDistortion(model.distortion.type))
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchFisheyeSse(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::OMNIDIRECTIONAL)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchOmniSse(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::DOUBLE_SPHERE)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchDsphSse(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
  if (model.projection.type == ProjectionModelType::EUCM)
  {
    return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
      return detail::rayToPixelBatchEucmSse(
        model, rays_xyz + 3 * begin, len, u_out + begin, v_out + begin,
        detail::offsetStatuses(statuses_out, begin)
      );
    });
  }
#endif

  const PixelFn fn = resolveForwardFn(model.projection.type);
  if (fn == nullptr)
  {
    for (int i = 0; i < count; ++i)
    {
      u_out[i] = 0.0f;
      v_out[i] = 0.0f;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = StatusCode::INVALID_MODEL;
      }
    }
    return 0;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const int offset = i * 3;
    const Eigen::Vector3f ray(rays_xyz[offset], rays_xyz[offset + 1], rays_xyz[offset + 2]);
    const PixelResult result = fn(model, ray);
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

int pixelToRayBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix2Xf> &pixels,
  Eigen::Ref<Eigen::Matrix3Xf> directions_out, StatusCode *statuses_out,
  const SolverOptions &solver_options
)
{
  if (pixels.rows() != 2 || directions_out.rows() != 3)
  {
    directions_out.setZero();
    const int count = static_cast<int>(pixels.cols());
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_INPUT;
      }
    }
    return -1;
  }

  const int count = static_cast<int>(pixels.cols());
  if (count <= 0)
  {
    return 0;
  }
  if (directions_out.cols() != count)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_INPUT;
      }
    }
    return -1;
  }

  const StatusCode validation = detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = validation;
      }
    }
    return -1;
  }

  const RayFn fn = resolveInverseFn(model.projection.type);
  if (fn == nullptr)
  {
    directions_out.setZero();
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        statuses_out[i] = StatusCode::INVALID_MODEL;
      }
    }
    return -1;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const RayResult result = fn(model, Pixel2{pixels(0, i), pixels(1, i)}, solver_options);
    if (result.status == StatusCode::OK)
    {
      directions_out.col(i) = result.ray.direction;
    }
    else
    {
      directions_out.col(i).setZero();
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

int pixelToRayBatch(
  const CameraModel &model, const float *u_in, const float *v_in, const int count, float *dirs_xyz,
  StatusCode *statuses_out, const SolverOptions &solver_options
)
{
  if (count < 0)
  {
    return -1;
  }
  if (count == 0)
  {
    return 0;
  }
  if (u_in == nullptr || v_in == nullptr || dirs_xyz == nullptr)
  {
    if (dirs_xyz != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        const int offset = i * 3;
        dirs_xyz[offset] = 0.0f;
        dirs_xyz[offset + 1] = 0.0f;
        dirs_xyz[offset + 2] = 0.0f;
      }
    }
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = detail::validateCameraModelQuery(model);
  if (validation != StatusCode::OK)
  {
    for (int i = 0; i < count; ++i)
    {
      const int offset = i * 3;
      dirs_xyz[offset] = 0.0f;
      dirs_xyz[offset + 1] = 0.0f;
      dirs_xyz[offset + 2] = 0.0f;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = validation;
      }
    }
    return 0;
  }

// On aarch64 the SIMD inverse runs through the NEON compat shim. With the
// Newton early-exit and the 4-wide status-path round-trip verification
// (simd_inverse.{hpp,cpp}) it beats the scalar OpenMP path for every model
// EXCEPT the KB4-family fisheyes: their mandatory round-trip guard plus the
// per-lane scalar trig in the theta path still lose to the scalar solver
// (measured on a 12-core Jetson Orin), so those stay pinned to scalar there.
#if defined(__aarch64__)
  const bool simd_inverse_profitable =
    !(model.projection.type == ProjectionModelType::FISHEYE_THETA &&
      (model.distortion.type == DistortionModelType::OPENCV_FISHEYE4 ||
       model.distortion.type == DistortionModelType::KB4 ||
       model.distortion.type == DistortionModelType::KB8));
#else
  const bool simd_inverse_profitable = true;
#endif
  if (simd_inverse_profitable)
  {
#ifdef CAMXIOM_HAS_AVX2
    if (model.projection.type == ProjectionModelType::PINHOLE)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchPinholeAvx2(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::FISHEYE_THETA)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchFisheyeAvx2(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::OMNIDIRECTIONAL)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchOmniAvx2(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::DOUBLE_SPHERE)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchDsphAvx2(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::EUCM)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchEucmAvx2(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
#elif defined(CAMXIOM_HAS_SSE2)
    if (model.projection.type == ProjectionModelType::PINHOLE)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchPinholeSse(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::FISHEYE_THETA)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchFisheyeSse(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::OMNIDIRECTIONAL)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchOmniSse(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::DOUBLE_SPHERE)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchDsphSse(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
    if (model.projection.type == ProjectionModelType::EUCM)
    {
      return detail::runBatchKernelParallel(count, [&](const int begin, const int len) {
        return detail::pixelToRayBatchEucmSse(
          model, u_in + begin, v_in + begin, len, dirs_xyz + 3 * begin,
          detail::offsetStatuses(statuses_out, begin), solver_options
        );
      });
    }
#endif
  }

  const RayFn fn = resolveInverseFn(model.projection.type);
  if (fn == nullptr)
  {
    for (int i = 0; i < count; ++i)
    {
      const int offset = i * 3;
      dirs_xyz[offset] = 0.0f;
      dirs_xyz[offset + 1] = 0.0f;
      dirs_xyz[offset + 2] = 0.0f;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = StatusCode::INVALID_MODEL;
      }
    }
    return 0;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const RayResult result = fn(model, Pixel2{u_in[i], v_in[i]}, solver_options);
    const int offset = i * 3;
    if (result.status == StatusCode::OK)
    {
      dirs_xyz[offset] = result.ray.direction.x();
      dirs_xyz[offset + 1] = result.ray.direction.y();
      dirs_xyz[offset + 2] = result.ray.direction.z();
    }
    else
    {
      dirs_xyz[offset] = 0.0f;
      dirs_xyz[offset + 1] = 0.0f;
      dirs_xyz[offset + 2] = 0.0f;
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

}  // namespace camxiom
