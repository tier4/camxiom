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

#include "camxiom/jacobian_batch.hpp"
#include "camxiom/jacobian_batch64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "jacobian/jacobian_impl.hpp"

#include <type_traits>

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

// Scalar-templated batch Jacobian implementation (#1 single-source): one
// template drives both rayToPixelWithJacobianBatch (float) and
// rayToPixelWithJacobianBatch64 (double); the two files were previously
// verbatim type-substituted twins.
//
// The batch entry validates the model ONCE, then resolves the per-model
// Jacobian implementation to a function pointer and calls it directly in the
// hot loop -- the generic rayToPixelWithJacobian(64) would re-validate the
// model on every point (same pattern as batch/float.cpp). The per-model impl
// functions keep their own per-ray guards (finite input, behind-camera), so
// per-point statuses are unchanged.

namespace camxiom
{
namespace
{

template <typename T>
using PerModelJacFn =
  ProjectionJacobianT<T> (*)(const CameraModelT<T> &, const Eigen::Matrix<T, 3, 1> &);

template <typename T>
PerModelJacFn<T> resolveJacobianFn(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::PINHOLE:
      return &pinhole::impl::rayToPixelWithJacobian<T>;
    case ProjectionModelType::FISHEYE_THETA:
      return &fisheye::impl::rayToPixelWithJacobian<T>;
    case ProjectionModelType::OMNIDIRECTIONAL:
      return &omnidirectional::impl::rayToPixelWithJacobian<T>;
    case ProjectionModelType::DOUBLE_SPHERE:
      return &double_sphere::impl::rayToPixelWithJacobian<T>;
    case ProjectionModelType::EUCM:
      return &eucm::impl::rayToPixelWithJacobian<T>;
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return nullptr;
}

template <typename T>
StatusCode validateModelT(const CameraModelT<T> &model)
{
  if constexpr (std::is_same_v<T, float>)
  {
    return validateCameraModel(model);
  }
  else
  {
    return validateCameraModel64(model);
  }
}

template <typename T>
int batchEigenImpl(
  const CameraModelT<T> &model,
  const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>> &ray_directions,
  Eigen::Ref<Eigen::Matrix<T, 2, Eigen::Dynamic>> pixels_out, Eigen::Matrix<T, 2, 3> *jacobians_out,
  StatusCode *statuses_out
)
{
  if (ray_directions.rows() != 3 || pixels_out.rows() != 2)
  {
    pixels_out.setZero();
    const int count = static_cast<int>(ray_directions.cols());
    for (int i = 0; i < count; ++i)
    {
      if (jacobians_out != nullptr)
      {
        jacobians_out[i].setZero();
      }
      if (statuses_out != nullptr)
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
    for (int i = 0; i < count; ++i)
    {
      if (jacobians_out != nullptr)
      {
        jacobians_out[i].setZero();
      }
      if (statuses_out != nullptr)
      {
        statuses_out[i] = StatusCode::INVALID_INPUT;
      }
    }
    return -1;
  }

  const StatusCode validation = validateModelT<T>(model);
  const PerModelJacFn<T> jac_fn =
    (validation == StatusCode::OK) ? resolveJacobianFn<T>(model.projection.type) : nullptr;
  if (validation != StatusCode::OK || jac_fn == nullptr)
  {
    const StatusCode failure =
      (validation != StatusCode::OK) ? validation : StatusCode::INVALID_MODEL;
    pixels_out.setZero();
    for (int i = 0; i < count; ++i)
    {
      if (jacobians_out != nullptr)
      {
        jacobians_out[i].setZero();
      }
      if (statuses_out != nullptr)
      {
        statuses_out[i] = failure;
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
    const ProjectionJacobianT<T> result = jac_fn(model, ray_directions.col(i));
    pixels_out(0, i) = result.pixel.u;
    pixels_out(1, i) = result.pixel.v;
    if (jacobians_out != nullptr)
    {
      jacobians_out[i] = result.J;
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

template <typename T>
int batchPointerImpl(
  const CameraModelT<T> &model, const T *rays_xyz, const int count, T *u_out, T *v_out,
  T *jacobians_out, StatusCode *statuses_out
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
      for (int i = 0; i < count; ++i) u_out[i] = T(0);
    }
    if (v_out != nullptr)
    {
      for (int i = 0; i < count; ++i) v_out[i] = T(0);
    }
    if (jacobians_out != nullptr)
    {
      for (int i = 0; i < count; ++i)
      {
        const int joff = i * 6;
        for (int k = 0; k < 6; ++k) jacobians_out[joff + k] = T(0);
      }
    }
    if (statuses_out != nullptr)
    {
      for (int i = 0; i < count; ++i) statuses_out[i] = StatusCode::INVALID_INPUT;
    }
    return -1;
  }

  const StatusCode validation = validateModelT<T>(model);
  const PerModelJacFn<T> jac_fn =
    (validation == StatusCode::OK) ? resolveJacobianFn<T>(model.projection.type) : nullptr;
  if (validation != StatusCode::OK || jac_fn == nullptr)
  {
    const StatusCode failure =
      (validation != StatusCode::OK) ? validation : StatusCode::INVALID_MODEL;
    for (int i = 0; i < count; ++i)
    {
      u_out[i] = T(0);
      v_out[i] = T(0);
      if (jacobians_out != nullptr)
      {
        const int joff = i * 6;
        for (int k = 0; k < 6; ++k) jacobians_out[joff + k] = T(0);
      }
      if (statuses_out != nullptr)
      {
        statuses_out[i] = failure;
      }
    }
    // Historical asymmetry, kept intact: the pointer variant reports a
    // validation failure as 0 while the Eigen variant reports -1 (documented
    // in jacobian_batch.hpp).
    return 0;
  }

  int valid_count = 0;
#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int i = 0; i < count; ++i)
  {
    const int off = i * 3;
    const Eigen::Matrix<T, 3, 1> ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
    const ProjectionJacobianT<T> result = jac_fn(model, ray);
    u_out[i] = result.pixel.u;
    v_out[i] = result.pixel.v;
    if (jacobians_out != nullptr)
    {
      const int joff = i * 6;
      jacobians_out[joff + 0] = result.J(0, 0);
      jacobians_out[joff + 1] = result.J(0, 1);
      jacobians_out[joff + 2] = result.J(0, 2);
      jacobians_out[joff + 3] = result.J(1, 0);
      jacobians_out[joff + 4] = result.J(1, 1);
      jacobians_out[joff + 5] = result.J(1, 2);
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

}  // namespace

int rayToPixelWithJacobianBatch(
  const CameraModel &model, const Eigen::Ref<const Eigen::Matrix3Xf> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xf> pixels_out, Eigen::Matrix<float, 2, 3> *jacobians_out,
  StatusCode *statuses_out
)
{
  return batchEigenImpl<float>(model, ray_directions, pixels_out, jacobians_out, statuses_out);
}

int rayToPixelWithJacobianBatch(
  const CameraModel &model, const float *rays_xyz, const int count, float *u_out, float *v_out,
  float *jacobians_out, StatusCode *statuses_out
)
{
  return batchPointerImpl<float>(model, rays_xyz, count, u_out, v_out, jacobians_out, statuses_out);
}

int rayToPixelWithJacobianBatch64(
  const CameraModel64 &model, const Eigen::Ref<const Eigen::Matrix3Xd> &ray_directions,
  Eigen::Ref<Eigen::Matrix2Xd> pixels_out, Eigen::Matrix<double, 2, 3> *jacobians_out,
  StatusCode *statuses_out
)
{
  return batchEigenImpl<double>(model, ray_directions, pixels_out, jacobians_out, statuses_out);
}

int rayToPixelWithJacobianBatch64(
  const CameraModel64 &model, const double *rays_xyz, const int count, double *u_out, double *v_out,
  double *jacobians_out, StatusCode *statuses_out
)
{
  return batchPointerImpl<double>(
    model, rays_xyz, count, u_out, v_out, jacobians_out, statuses_out
  );
}

}  // namespace camxiom
