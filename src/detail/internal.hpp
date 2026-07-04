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

#ifndef CAMXIOM__DETAIL__INTERNAL_HPP
#define CAMXIOM__DETAIL__INTERNAL_HPP

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "distortion/plane_impl.hpp"

#include <cmath>

namespace camxiom::detail
{

inline constexpr float kEpsilon = 1e-8f;
inline constexpr float kPi = camxiom::constants::kPiF;
inline constexpr float kHalfPi = camxiom::constants::kHalfPiF;

// NormPoint is the float instantiation of the shared, scalar-templated
// normalised-image-point type (#1 step 2a). detail64::NormPoint64 aliases the
// double instantiation; both name the same NormPointT<T> aggregate.
using NormPoint = detail_impl::NormPointT<float>;

inline bool isFiniteScalar(const float value) { return std::isfinite(value); }

inline bool isFinite2(const float x, const float y) { return std::isfinite(x) && std::isfinite(y); }

inline bool isFinite3(const float x, const float y, const float z)
{
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

inline PixelResult invalidPixelResult(const StatusCode status)
{
  PixelResult result;
  result.status = status;
  result.pixel = Pixel2{};
  return result;
}

inline RayResult invalidRayResult(const StatusCode status)
{
  RayResult result;
  result.status = status;
  result.ray.origin = Eigen::Vector3f::Zero();
  result.ray.direction = Eigen::Vector3f::Zero();
  return result;
}

inline bool removeIntrinsics(
  const IntrinsicsModel &intrinsics, const Pixel2 &pixel, float &x_distorted, float &y_distorted
)
{
  y_distorted = (pixel.v - intrinsics.cy) / intrinsics.fy;
  x_distorted = (pixel.u - intrinsics.cx - intrinsics.skew * y_distorted) / intrinsics.fx;
  return isFinite2(x_distorted, y_distorted);
}

// ---------------------------------------------------------------------------
// Distortion model consistency helpers (type-only, shared by float and double)
// ---------------------------------------------------------------------------

inline DistortionSpace expectedSpaceFromType(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
      return DistortionSpace::NONE;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
      return DistortionSpace::PLANE;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return DistortionSpace::ANGLE;
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return DistortionSpace::NONE;
}

inline bool typeRequiresRational(const DistortionModelType type)
{
  return type == DistortionModelType::RATIONAL8 || type == DistortionModelType::THIN_PRISM12 ||
         type == DistortionModelType::TILTED14;
}

// The polynomial theta-distortion families: theta_d is a fitted odd polynomial
// of theta, so its monotone range depends on the coefficients (unlike the
// ideal trig mappings, whose monotone range is fixed per type). These are the
// types whose theta_max cap must sit inside the polynomial's positive monotone
// range for the inverse bracket to be usable.
inline bool isPolynomialFisheyeDistortion(const DistortionModelType type)
{
  return type == DistortionModelType::OPENCV_FISHEYE4 || type == DistortionModelType::KB4 ||
         type == DistortionModelType::KB8;
}

inline bool typeAllowsThinPrism(const DistortionModelType type)
{
  return type == DistortionModelType::THIN_PRISM12 || type == DistortionModelType::TILTED14;
}

inline bool typeAllowsTilt(const DistortionModelType type)
{
  return type == DistortionModelType::TILTED14;
}

inline std::uint8_t minimumCoefficientCountForType(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
      return 0U;
    case DistortionModelType::RADTAN4:
      return 4U;
    case DistortionModelType::RADTAN5:
      return 5U;
    case DistortionModelType::RATIONAL8:
      return 8U;
    case DistortionModelType::THIN_PRISM12:
      return 12U;
    case DistortionModelType::TILTED14:
      return 14U;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
      return 4U;
    case DistortionModelType::KB8:
      return 8U;
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return 0U;
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return 0U;
}

inline std::uint8_t maximumCoefficientCountForType(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
      return 0U;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
      return 5U;
    case DistortionModelType::RATIONAL8:
      return 8U;
    case DistortionModelType::THIN_PRISM12:
      return 12U;
    case DistortionModelType::TILTED14:
      return 14U;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
      return 4U;
    case DistortionModelType::KB8:
      return 8U;
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return 14U;
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return 0U;
}

// Number of leading distortion coefficients that must be finite for `type`,
// independent of how many the caller declared. Type-only (precision-agnostic)
// single source for both validateCameraModel (float) and validateCameraModel64
// (double); previously hand-duplicated as detail::requiredFiniteCoefficientCount
// (model/internal.hpp) and detail64::requiredFiniteCoefficientCount64
// (projection64/internal.hpp).
inline std::size_t requiredFiniteCoefficientCount(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return 0U;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
      return 5U;
    case DistortionModelType::RATIONAL8:
      return 8U;
    case DistortionModelType::THIN_PRISM12:
      return 12U;
    case DistortionModelType::TILTED14:
      return 14U;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
      return 4U;
    case DistortionModelType::KB8:
      return 8U;
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return 0U;
}

inline StatusCode validateDistortionConsistency(
  const DistortionModelType type, const DistortionSpace space, const std::uint8_t count,
  const bool is_rational, const bool has_thin_prism, const bool has_tilt
)
{
  if (type == DistortionModelType::UNKNOWN || type == DistortionModelType::OMNIDIRECTIONAL)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (type == DistortionModelType::NONE)
  {
    if (space != DistortionSpace::NONE || count != 0U || is_rational || has_thin_prism || has_tilt)
    {
      return StatusCode::INVALID_MODEL;
    }
    return StatusCode::OK;
  }
  const DistortionSpace expected_space = expectedSpaceFromType(type);
  if (expected_space != DistortionSpace::NONE && space != expected_space)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (count < minimumCoefficientCountForType(type))
  {
    return StatusCode::INVALID_MODEL;
  }
  if (count > maximumCoefficientCountForType(type))
  {
    return StatusCode::INVALID_MODEL;
  }
  if (is_rational != typeRequiresRational(type))
  {
    return StatusCode::INVALID_MODEL;
  }
  if (has_thin_prism && !typeAllowsThinPrism(type))
  {
    return StatusCode::INVALID_MODEL;
  }
  if (has_tilt && !typeAllowsTilt(type))
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename CoeffsArray>
inline StatusCode validateCoeffsFlagsConsistency(
  const DistortionModelType type, const CoeffsArray &coeffs, const std::uint8_t count,
  const bool has_thin_prism, const bool has_tilt
)
{
  if (typeAllowsThinPrism(type) && !has_thin_prism)
  {
    const std::size_t end = std::min(static_cast<std::size_t>(12), static_cast<std::size_t>(count));
    for (std::size_t i = 8; i < end; ++i)
    {
      if (coeffs[i] != 0)
      {
        return StatusCode::INVALID_MODEL;
      }
    }
  }
  if (typeAllowsTilt(type) && !has_tilt)
  {
    const std::size_t end = std::min(static_cast<std::size_t>(14), static_cast<std::size_t>(count));
    for (std::size_t i = 12; i < end; ++i)
    {
      if (coeffs[i] != 0)
      {
        return StatusCode::INVALID_MODEL;
      }
    }
  }
  return StatusCode::OK;
}

// ---------------------------------------------------------------------------
// Plane distortion (forward / inverse)
// ---------------------------------------------------------------------------

StatusCode distortPlaneModel(
  const DistortionModel &model, const NormPoint &in_xy, NormPoint &out_xy
);

struct DistortionJacobian2x2
{
  float j00{1.0f};
  float j01{0.0f};
  float j10{0.0f};
  float j11{1.0f};
};

StatusCode distortPlaneModelWithJacobian(
  const DistortionModel &model, const NormPoint &in_xy, NormPoint &out_xy,
  DistortionJacobian2x2 &jacobian
);

StatusCode undistortPlaneModelNewton(
  const DistortionModel &model, const NormPoint &observed_xy, const SolverOptions &solver_options,
  NormPoint &undistorted_xy
);

// ---------------------------------------------------------------------------
// Angle distortion (forward / inverse)
// ---------------------------------------------------------------------------

StatusCode distortTheta(const DistortionModel &model, float theta, float &theta_d);

StatusCode distortThetaDerivative(const DistortionModel &model, float theta, float &derivative_out);

StatusCode undistortThetaHybrid(
  const DistortionModel &model, const ProjectionModel &projection, float radius_d,
  const SolverOptions &solver_options, float &theta_out
);

// ---------------------------------------------------------------------------
// Query-tier model validation (defined in src/model/validate.cpp)
// ---------------------------------------------------------------------------

/// The per-call guard used by the per-point / bulk QUERY paths (rayToPixel,
/// pixelToRay, Jacobians, batch): every structural model invariant, at a cost
/// bounded for per-point use. The public validateCameraModel() is this check
/// PLUS the scan-based fisheye monotone-cap certification, which is too
/// expensive to run per projected point. Every query path shares this exact
/// predicate so a given model is accepted or rejected identically by the
/// scalar, batch, SIMD, and Jacobian paths.
StatusCode validateCameraModelQuery(const CameraModel &model);

}  // namespace camxiom::detail

#endif  // CAMXIOM__DETAIL__INTERNAL_HPP
