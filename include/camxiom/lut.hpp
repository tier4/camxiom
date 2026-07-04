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

#ifndef CAMXIOM__LUT_HPP
#define CAMXIOM__LUT_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <vector>

namespace camxiom
{

namespace detail
{
// Maps the LUT scalar type to its matching (defaulted) solver-options type.
// InverseLutT<float> uses SolverOptions (types.hpp); InverseLutT<double> uses
// SolverOptions64, specialised in camxiom/lut64.hpp where that type is visible.
// This keeps each precision's distinct default solver behaviour (float 10 iters
// / 1e-6, double 15 iters / 1e-12) while sharing one class template.
template <typename T>
struct LutSolverOptions;

template <>
struct LutSolverOptions<float>
{
  using type = SolverOptions;
};
}  // namespace detail

/// Pre-computed lookup table for fast approximate inverse projection (pixelToRay).
///
/// The LUT stores unit direction vectors on a regular grid covering the image.
/// Queries at arbitrary (u, v) use bilinear interpolation for O(1) lookup.
///
/// This is a single scalar-templated class (#1 step 5). The float32 runtime type
/// is `InverseLut` (this header); the double type is `InverseLut64`
/// (camxiom/lut64.hpp). Both are aliases of `InverseLutT<T>`, so the public API
/// and object layout are unchanged from the previous hand-duplicated classes.
///
/// Typical usage:
///   InverseLut lut;
///   lut.build(model, 1280, 720);            // step=1 → full resolution
///   lut.build(model, 1280, 720, {}, 4);     // step=4 → 320x180 grid, ~16x less memory
///   RayResult r = lut.query(500.3f, 300.7f); // O(1) bilinear interpolation
///
/// Thread safety: one writer, then many readers. build() and clear() mutate
/// the table and must not run concurrently with anything else on the same
/// instance; once build() has returned, concurrent query() calls (const)
/// from any number of threads are safe. (Details:
/// docs/design/thread-safety.md.)
template <typename T>
class InverseLutT
{
public:
  InverseLutT() = default;

  /// Build the LUT by evaluating pixelToRay on a regular grid.
  ///
  /// @param model   Camera model.
  /// @param width   Image width in pixels.
  /// @param height  Image height in pixels.
  /// @param solver_options  Solver options for pixelToRay during build.
  /// @param step    Grid step size in pixels (1 = full resolution, 4 = quarter).
  /// @return Number of valid grid points.
  int build(
    const CameraModelT<T> &model, int width, int height,
    const typename detail::LutSolverOptions<T>::type &solver_options = {}, int step = 1
  );

  /// Query the LUT at sub-pixel coordinates using bilinear interpolation.
  /// Returns an approximate ray direction. O(1) time.
  RayResultT<T> query(T u, T v) const;

  /// Check if the LUT has been built.
  bool isValid() const;

  /// Get the grid dimensions.
  int gridWidth() const;
  int gridHeight() const;
  int step() const;

  /// Get the image dimensions the LUT was built for.
  int imageWidth() const;
  int imageHeight() const;

  /// Clear the LUT and free memory.
  void clear();

private:
  std::vector<T> dirs_x_{};
  std::vector<T> dirs_y_{};
  std::vector<T> dirs_z_{};
  std::vector<std::uint8_t> valid_{};
  int grid_w_{0};
  int grid_h_{0};
  int image_w_{0};
  int image_h_{0};
  int step_{1};
  bool built_{false};
};

/// float32 runtime alias (primary real-time API).
using InverseLut = InverseLutT<float>;

}  // namespace camxiom

#endif  // CAMXIOM__LUT_HPP
