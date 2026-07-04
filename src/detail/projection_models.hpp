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

#ifndef CAMXIOM__DETAIL__PROJECTION_MODELS_HPP
#define CAMXIOM__DETAIL__PROJECTION_MODELS_HPP

// INTERNAL per-model projection entry points (NOT part of the public API).
//
// Each projection model exposes concrete (non-template) float and double
// rayToPixel/pixelToRay functions in its own namespace. These are the stable
// per-model symbols that the generic dispatch (src/dispatch.cpp,
// src/projection64/dispatch.cpp), the batch layer (src/batch/*), and the SIMD
// scalar fallbacks / function-pointer tables (src/detail/simd_*) call into.
// The math itself lives once in the scalar-templated cores
// (src/<model>/projection_impl.hpp, #1); these functions are the thin <float>
// / <double> instantiation boundary, defined in src/<model>/projection{,64}.cpp.
//
// #3: these declarations used to be copy-pasted as five near-identical
// namespace blocks in the PUBLIC headers (projection.hpp / projection64.hpp).
// The public API is the generic rayToPixel/pixelToRay (+ *64) and the batch
// API; the per-model entry points are an implementation detail, so they are
// declared here (out of the installed headers) and generated from a single
// macro to remove the symmetric copy-paste.

#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

namespace camxiom
{

// Declare the float + double per-model projection entry points for one model
// namespace. Defined per model in src/<model>/projection.cpp (float) and
// src/<model>/projection64.cpp (double).
#define CAMXIOM_DECLARE_MODEL_PROJECTION(model_ns)                                              \
  namespace model_ns                                                                            \
  {                                                                                             \
  PixelResult rayToPixel(const CameraModel &model, const Eigen::Vector3f &ray_direction);       \
  RayResult pixelToRay(                                                                         \
    const CameraModel &model, const Pixel2 &pixel,                                              \
    const SolverOptions &solver_options = SolverOptions{}                                       \
  );                                                                                            \
  PixelResult64 rayToPixel64(const CameraModel64 &model, const Eigen::Vector3d &ray_direction); \
  RayResult64 pixelToRay64(                                                                     \
    const CameraModel64 &model, const Pixel2d &pixel,                                           \
    const SolverOptions64 &solver_options = SolverOptions64{}                                   \
  );                                                                                            \
  }  // namespace model_ns

CAMXIOM_DECLARE_MODEL_PROJECTION(pinhole)
CAMXIOM_DECLARE_MODEL_PROJECTION(fisheye)
CAMXIOM_DECLARE_MODEL_PROJECTION(omnidirectional)
CAMXIOM_DECLARE_MODEL_PROJECTION(double_sphere)
CAMXIOM_DECLARE_MODEL_PROJECTION(eucm)

#undef CAMXIOM_DECLARE_MODEL_PROJECTION

}  // namespace camxiom

#endif  // CAMXIOM__DETAIL__PROJECTION_MODELS_HPP
