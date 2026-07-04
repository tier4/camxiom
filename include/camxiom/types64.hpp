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

#ifndef CAMXIOM__TYPES64_HPP
#define CAMXIOM__TYPES64_HPP

#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <array>
#include <cstdint>

namespace camxiom
{

// ---------------------------------------------------------------------------
// Double-precision (64-bit) counterparts of the core types.
// The float32 types in types.hpp remain the primary API for real-time use.
// These 64-bit types are for calibration / optimization where precision matters.
// ---------------------------------------------------------------------------

// --- float64 aliases of the scalar-templated types defined in types.hpp ----
// (#1 unification: same struct layout as the float runtime types, instantiated
// for double. Member names are identical, so all consumers and the conversion
// helpers below are unchanged.)
using Pixel2d = Pixel2T<double>;
using Ray3d = Ray3T<double>;
using IntrinsicsModel64 = IntrinsicsModelT<double>;
using ProjectionModel64 = ProjectionModelT<double>;
using DistortionModel64 = DistortionModelT<double>;
using CameraModel64 = CameraModelT<double>;
using PixelResult64 = PixelResultT<double>;
using RayResult64 = RayResultT<double>;

// SolverOptions64 keeps double-specific defaults (tighter tolerances / more
// iterations than the float SolverOptions) on purpose, so it stays a distinct
// type rather than an alias.
struct SolverOptions64
{
  int max_iterations{15};
  double residual_tolerance{1e-12};
  double step_tolerance{1e-14};
  bool skip_verify{false};
};

// ---------------------------------------------------------------------------
// Conversion between float32 and float64 types
// ---------------------------------------------------------------------------

namespace detail
{

// Single source of truth for the CameraModel precision conversions exposed as
// toCameraModel64 / fromCameraModel64 below. Numeric fields go through
// static_cast<To>; enum / flag / count fields are copied verbatim. Because the
// float and double models are the same CameraModelT<T> template (#1), this is
// written once for both directions — adding a field to the CameraModelT family
// only needs one update here, instead of two hand-mirrored copies.
template <typename To, typename From>
CameraModelT<To> cameraModelCast(const CameraModelT<From> &m)
{
  CameraModelT<To> out;

  out.intrinsics.fx = static_cast<To>(m.intrinsics.fx);
  out.intrinsics.fy = static_cast<To>(m.intrinsics.fy);
  out.intrinsics.cx = static_cast<To>(m.intrinsics.cx);
  out.intrinsics.cy = static_cast<To>(m.intrinsics.cy);
  out.intrinsics.skew = static_cast<To>(m.intrinsics.skew);

  out.projection.type = m.projection.type;
  out.projection.theta_max = static_cast<To>(m.projection.theta_max);
  out.projection.xi = static_cast<To>(m.projection.xi);
  out.projection.alpha = static_cast<To>(m.projection.alpha);
  out.projection.beta = static_cast<To>(m.projection.beta);

  out.distortion.type = m.distortion.type;
  out.distortion.space = m.distortion.space;
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    out.distortion.coeffs[i] = static_cast<To>(m.distortion.coeffs[i]);
  }
  for (std::size_t i = 0; i < m.distortion.tilt_matrix.size(); ++i)
  {
    out.distortion.tilt_matrix[i] = static_cast<To>(m.distortion.tilt_matrix[i]);
    out.distortion.inv_tilt_matrix[i] = static_cast<To>(m.distortion.inv_tilt_matrix[i]);
  }
  out.distortion.count = m.distortion.count;
  out.distortion.is_rational = m.distortion.is_rational;
  out.distortion.has_thin_prism = m.distortion.has_thin_prism;
  out.distortion.has_tilt = m.distortion.has_tilt;

  return out;
}

}  // namespace detail

inline CameraModel64 toCameraModel64(const CameraModel &m)
{
  return detail::cameraModelCast<double>(m);
}

inline CameraModel fromCameraModel64(const CameraModel64 &m64)
{
  return detail::cameraModelCast<float>(m64);
}

inline SolverOptions64 toSolverOptions64(const SolverOptions &opts)
{
  SolverOptions64 opts64;
  opts64.max_iterations = opts.max_iterations;
  opts64.residual_tolerance = static_cast<double>(opts.residual_tolerance);
  opts64.step_tolerance = static_cast<double>(opts.step_tolerance);
  opts64.skip_verify = opts.skip_verify;
  return opts64;
}

}  // namespace camxiom

#endif  // CAMXIOM__TYPES64_HPP
