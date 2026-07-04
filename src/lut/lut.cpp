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

#include "camxiom/lut.hpp"

#include "camxiom/lut64.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "distortion/plane_impl.hpp"  // detail_impl::PlaneTraits<T>::kEpsilon

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#ifdef CAMXIOM_HAS_OPENMP
#include <omp.h>
#endif

// Scalar-templated InverseLutT<T> implementation (#1 step 5). Single source for
// the build/query logic previously hand-duplicated between src/lut/float.cpp
// (InverseLut) and src/lut/double.cpp (InverseLut64). The precision-specific
// dispatch (validateCameraModel / pixelToRay vs the *64 variants) is selected
// with `if constexpr`, and the query degeneracy threshold uses
// PlaneTraits<T>::kEpsilon (float 1e-8f / double 1e-15), matching the old code.

namespace camxiom
{

template <typename T>
int InverseLutT<T>::build(
  const CameraModelT<T> &model, const int width, const int height,
  const typename detail::LutSolverOptions<T>::type &solver_options, const int step
)
{
  clear();

  if (width <= 0 || height <= 0 || step <= 0)
  {
    return 0;
  }

  StatusCode validation;
  if constexpr (std::is_same_v<T, float>)
  {
    validation = validateCameraModel(model);
  }
  else
  {
    validation = validateCameraModel64(model);
  }
  if (validation != StatusCode::OK)
  {
    return 0;
  }

  // Grid sizing in 64-bit: width + step - 1 overflows int for pathological
  // near-INT_MAX inputs (same guard family as the remap pixel-count checks).
  const long long grid_w_ll = (static_cast<long long>(width) + step - 1) / step + 1;
  const long long grid_h_ll = (static_cast<long long>(height) + step - 1) / step + 1;
  if (grid_w_ll > static_cast<long long>(std::numeric_limits<int>::max()) || grid_h_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    return 0;
  }
  // The per-axis checks above do not bound the PRODUCT: e.g. a 200000 x
  // 200000 image at step 1 passes both yet asks the resize()s below for
  // ~160 GB, which throws uncaught instead of following this function's
  // reject-with-0 contract. Cap the total like the axes.
  if (grid_w_ll * grid_h_ll > static_cast<long long>(std::numeric_limits<int>::max()))
  {
    return 0;
  }

  step_ = step;
  image_w_ = width;
  image_h_ = height;
  grid_w_ = static_cast<int>(grid_w_ll);
  grid_h_ = static_cast<int>(grid_h_ll);

  const std::size_t total = static_cast<std::size_t>(grid_w_) * static_cast<std::size_t>(grid_h_);
  dirs_x_.resize(total, T(0));
  dirs_y_.resize(total, T(0));
  dirs_z_.resize(total, T(0));
  valid_.resize(total, 0U);

  int valid_count = 0;

#ifdef CAMXIOM_HAS_OPENMP
#pragma omp parallel for reduction(+ : valid_count) schedule(static)
#endif
  for (int gy = 0; gy < grid_h_; ++gy)
  {
    const T v = static_cast<T>(gy * step);
    const std::size_t row_offset = static_cast<std::size_t>(gy) * static_cast<std::size_t>(grid_w_);

    for (int gx = 0; gx < grid_w_; ++gx)
    {
      const T u = static_cast<T>(gx * step);
      const std::size_t idx = row_offset + static_cast<std::size_t>(gx);

      RayResultT<T> result;
      if constexpr (std::is_same_v<T, float>)
      {
        result = pixelToRay(model, Pixel2T<T>{u, v}, solver_options);
      }
      else
      {
        result = pixelToRay64(model, Pixel2T<T>{u, v}, solver_options);
      }
      if (result.status == StatusCode::OK)
      {
        dirs_x_[idx] = result.ray.direction.x();
        dirs_y_[idx] = result.ray.direction.y();
        dirs_z_[idx] = result.ray.direction.z();
        valid_[idx] = 1U;
        ++valid_count;
      }
    }
  }

  built_ = (valid_count > 0);
  return valid_count;
}

template <typename T>
RayResultT<T> InverseLutT<T>::query(const T u, const T v) const
{
  RayResultT<T> result;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = Eigen::Matrix<T, 3, 1>::Zero();

  if (!built_)
  {
    result.status = StatusCode::INVALID_MODEL;
    return result;
  }
  if (!std::isfinite(u) || !std::isfinite(v))
  {
    result.status = StatusCode::INVALID_INPUT;
    return result;
  }

  const T gx_f = u / static_cast<T>(step_);
  const T gy_f = v / static_cast<T>(step_);

  // Range-check in floating point BEFORE the narrowing cast: a finite but
  // extreme coordinate (garbage from an upstream bug) would make the
  // float->int cast itself UB, not merely fail the grid bounds test below.
  if (gx_f < T(0) || gy_f < T(0) || gx_f >= static_cast<T>(grid_w_) || gy_f >= static_cast<T>(grid_h_))
  {
    result.status = StatusCode::OUT_OF_FOV;
    return result;
  }

  const int gx0 = static_cast<int>(std::floor(gx_f));
  const int gy0 = static_cast<int>(std::floor(gy_f));
  const int gx1 = gx0 + 1;
  const int gy1 = gy0 + 1;

  if (gx0 < 0 || gy0 < 0 || gx1 >= grid_w_ || gy1 >= grid_h_)
  {
    result.status = StatusCode::OUT_OF_FOV;
    return result;
  }

  const T tx = gx_f - static_cast<T>(gx0);
  const T ty = gy_f - static_cast<T>(gy0);

  const std::size_t gw = static_cast<std::size_t>(grid_w_);
  const std::size_t i00 = static_cast<std::size_t>(gy0) * gw + static_cast<std::size_t>(gx0);
  const std::size_t i10 = i00 + 1;
  const std::size_t i01 = i00 + gw;
  const std::size_t i11 = i01 + 1;

  if (valid_[i00] == 0U || valid_[i10] == 0U || valid_[i01] == 0U || valid_[i11] == 0U)
  {
    result.status = StatusCode::OUT_OF_FOV;
    return result;
  }

  const T w00 = (T(1) - tx) * (T(1) - ty);
  const T w10 = tx * (T(1) - ty);
  const T w01 = (T(1) - tx) * ty;
  const T w11 = tx * ty;

  const T dx = w00 * dirs_x_[i00] + w10 * dirs_x_[i10] + w01 * dirs_x_[i01] + w11 * dirs_x_[i11];
  const T dy = w00 * dirs_y_[i00] + w10 * dirs_y_[i10] + w01 * dirs_y_[i01] + w11 * dirs_y_[i11];
  const T dz = w00 * dirs_z_[i00] + w10 * dirs_z_[i10] + w01 * dirs_z_[i01] + w11 * dirs_z_[i11];

  const T norm = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (!std::isfinite(norm) || norm <= detail_impl::PlaneTraits<T>::kEpsilon)
  {
    result.status = StatusCode::NUMERIC_ERROR;
    return result;
  }

  const T inv_norm = T(1) / norm;
  result.status = StatusCode::OK;
  result.ray.origin = Eigen::Matrix<T, 3, 1>::Zero();
  result.ray.direction = Eigen::Matrix<T, 3, 1>(dx * inv_norm, dy * inv_norm, dz * inv_norm);
  return result;
}

template <typename T>
bool InverseLutT<T>::isValid() const
{
  return built_;
}

template <typename T>
int InverseLutT<T>::gridWidth() const
{
  return grid_w_;
}

template <typename T>
int InverseLutT<T>::gridHeight() const
{
  return grid_h_;
}

template <typename T>
int InverseLutT<T>::step() const
{
  return step_;
}

template <typename T>
int InverseLutT<T>::imageWidth() const
{
  return image_w_;
}

template <typename T>
int InverseLutT<T>::imageHeight() const
{
  return image_h_;
}

template <typename T>
void InverseLutT<T>::clear()
{
  dirs_x_.clear();
  dirs_y_.clear();
  dirs_z_.clear();
  valid_.clear();
  grid_w_ = 0;
  grid_h_ = 0;
  image_w_ = 0;
  image_h_ = 0;
  step_ = 1;
  built_ = false;
}

template class InverseLutT<float>;
template class InverseLutT<double>;

}  // namespace camxiom
