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

#ifndef CAMXIOM__MODEL__VALIDATE_IMPL_HPP
#define CAMXIOM__MODEL__VALIDATE_IMPL_HPP

// Scalar-templated CameraModel validation core (#1 step 4). Single source of
// truth for the model-consistency checks previously hand-duplicated between the
// float validator (src/model/validate.cpp, validateCameraModel) and the double
// validator (src/projection64/validate.cpp, validateCameraModel64). The two
// were ~140 near-identical lines kept in sync only by review and comments --
// exactly the two-sided maintenance hazard #1 exists to remove.
//
// Precision-specific pieces fold into the template cleanly:
//   - the focal-length floor uses PlaneTraits<T>::kEpsilon (float 1e-8f /
//     double 1e-15), matching the old detail::kEpsilon / detail64::kEpsilon;
//   - the FISHEYE_THETA theta_max upper bound compares against
//     static_cast<T>(constants::kPiF), i.e. the *float* pi at width T. This is
//     the structural form of the D47 fix: theta_max originates as a float, so
//     using the float pi (widened to double for the double path) makes the
//     float and double reject edges definitionally identical for every
//     float-origin value. Comparing the double model against true double pi
//     would wrongly reject a float pi (~8.74e-8 above true double pi).
//
// The distortion consistency / coeff-flags checks and the required-finite-coeff
// count are already precision-agnostic (camxiom::detail, detail/internal.hpp).

#include "camxiom/internal/constants.hpp"
#include "camxiom/types.hpp"
#include "detail/internal.hpp"
#include "distortion/angle_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace camxiom::detail_impl
{

// QUERY-tier validation: every structural invariant, at a cost bounded for
// per-projected-point use. This is the guard the per-point and bulk query
// paths (rayToPixel / pixelToRay / Jacobians / batch, both precisions) run on
// every call, so a given model is accepted or rejected identically by the
// scalar, batch, SIMD, and Jacobian paths. The public validateCameraModel /
// validateCameraModel64 (validateCameraModelImpl below) add the scan-based
// fisheye monotone-cap certification on top — too expensive per point, but
// the right cost for the one-shot oracle / builder / factory entry points.
template <typename T>
inline StatusCode validateCameraModelQueryImpl(const CameraModelT<T> &model)
{
  const IntrinsicsModelT<T> &intrinsics = model.intrinsics;
  if (!std::isfinite(intrinsics.fx) ||
      !std::isfinite(intrinsics.fy) ||
      !std::isfinite(intrinsics.cx) ||
      !std::isfinite(intrinsics.cy) ||
      !std::isfinite(intrinsics.skew))
  {
    return StatusCode::INVALID_INPUT;
  }
  if (std::abs(intrinsics.fx) <= PlaneTraits<T>::kEpsilon || std::abs(intrinsics.fy) <= PlaneTraits<T>::kEpsilon)
  {
    return StatusCode::INVALID_MODEL;
  }

  const ProjectionModelT<T> &projection = model.projection;
  if (projection.type == ProjectionModelType::UNKNOWN)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (projection.type == ProjectionModelType::FISHEYE_THETA)
  {
    // theta_max compared against the float pi at width T (see header note: D47).
    if (!std::isfinite(projection.theta_max) || projection.theta_max <= T(0) || projection.theta_max > static_cast<T>(camxiom::constants::kPiF))
    {
      return StatusCode::INVALID_MODEL;
    }
  }
  if (projection.type == ProjectionModelType::OMNIDIRECTIONAL)
  {
    if (!std::isfinite(projection.xi))
    {
      return StatusCode::INVALID_MODEL;
    }
  }
  if (projection.type == ProjectionModelType::DOUBLE_SPHERE)
  {
    // Usenko 2018 "The Double Sphere Camera Model": xi in the open interval
    // (-1, 1), alpha in [0, 1]. Boundaries xi = +/-1 are excluded because the
    // forward equation degenerates there; per-point singularities at denom <=
    // eps are handled inside the DS forward kernel.
    if (!std::isfinite(projection.xi) ||
        !std::isfinite(projection.alpha) ||
        projection.xi <= T(-1) ||
        projection.xi >= T(1) ||
        projection.alpha < T(0) ||
        projection.alpha > T(1))
    {
      return StatusCode::INVALID_MODEL;
    }
  }
  if (projection.type == ProjectionModelType::EUCM)
  {
    if (!std::isfinite(projection.alpha) || !std::isfinite(projection.beta) ||
        projection.beta <= T(0) ||
        projection.alpha < T(0) ||
        projection.alpha > T(1))
    {
      return StatusCode::INVALID_MODEL;
    }
  }

  const DistortionModelT<T> &distortion = model.distortion;

  const StatusCode consistency_status = detail::validateDistortionConsistency(
    distortion.type, distortion.space, distortion.count, distortion.is_rational,
    distortion.has_thin_prism, distortion.has_tilt
  );
  if (consistency_status != StatusCode::OK)
  {
    return consistency_status;
  }

  const StatusCode coeffs_flags_status = detail::validateCoeffsFlagsConsistency(
    distortion.type, distortion.coeffs, distortion.count, distortion.has_thin_prism,
    distortion.has_tilt
  );
  if (coeffs_flags_status != StatusCode::OK)
  {
    return coeffs_flags_status;
  }

  if (distortion.count > distortion.coeffs.size())
  {
    return StatusCode::INVALID_MODEL;
  }
  const std::size_t declared_count = static_cast<std::size_t>(distortion.count);
  const std::size_t required_count = detail::requiredFiniteCoefficientCount(distortion.type);
  const std::size_t finite_check_count = std::max(declared_count, required_count);
  if (finite_check_count > distortion.coeffs.size())
  {
    return StatusCode::INVALID_MODEL;
  }
  for (std::size_t index = 0; index < finite_check_count; ++index)
  {
    if (!std::isfinite(distortion.coeffs[index]))
    {
      return StatusCode::INVALID_MODEL;
    }
  }
  // Entries between the declared count and the count the runtime kernels
  // actually read must be zero. RADTAN4 (count 4) shares the RADTAN5 kernels,
  // which unconditionally read coeffs[4] as k3, while the calibration
  // template zero-fills beyond `count`: a non-zero value there would make the
  // same model project differently per path. Rejecting it here makes the
  // types.hpp contract ("count says how many leading entries are meaningful")
  // structural instead of a silent divergence.
  for (std::size_t index = declared_count; index < required_count; ++index)
  {
    if (distortion.coeffs[index] != T(0))
    {
      return StatusCode::INVALID_MODEL;
    }
  }
  // Coefficient-free types (NONE and the ideal trig fisheye mappings) define
  // ZERO meaningful entries: the scalar kernels ignore coeffs[] entirely,
  // while the fixed-width bulk kernels read leading entries unconditionally
  // (the SIMD fisheye forward applies coeffs[0..3] as a KB polynomial; the
  // plane-space SIMD kernels apply the radtan tail). Requiring every entry to
  // be zero makes the two readings coincide exactly — a zero polynomial IS
  // the identity / ideal mapping — so a model carrying stale non-zero
  // coefficients under one of these types is rejected instead of projecting
  // differently per code path. Same contract-hardening rationale as the
  // RADTAN4 rule above.
  if (required_count == 0U)
  {
    for (const T value : distortion.coeffs)
    {
      if (value != T(0))
      {
        return StatusCode::INVALID_MODEL;
      }
    }
  }

  if (projection.type == ProjectionModelType::PINHOLE && distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (projection.type == ProjectionModelType::FISHEYE_THETA && distortion.space == DistortionSpace::PLANE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (projection.type == ProjectionModelType::FISHEYE_THETA &&
      distortion.type == DistortionModelType::ORTHOGRAPHIC &&
      projection.theta_max > static_cast<T>(camxiom::constants::kHalfPiF))
  {
    // theta_d = sin(theta) folds over past pi/2; the inverse bracket assumes
    // monotonicity on [0, theta_max], so a wider cap silently turns solvable
    // pixels into OUT_OF_FOV and breaks forward/inverse round-trips. Matches
    // detail::defaultFisheyeThetaMax's monotone-range rule (float half-pi at
    // width T, same reasoning as the D47 pi bound above).
    return StatusCode::INVALID_MODEL;
  }
  if (projection.type == ProjectionModelType::FISHEYE_THETA)
  {
    // The inverse bracket (undistortThetaHybrid) seeds its upper endpoint with
    // distortTheta(theta_max) and rejects EVERY query when that endpoint is
    // not a usable positive bound (f_max < 0 => unconditional OUT_OF_FOV). A
    // forward mapping that diverges or goes negative at theta_max —
    // STEREOGRAPHIC capped at the pi bound above (kPiF at width T sits just
    // past the true tan pole at double pi), or a polynomial model whose
    // negative coefficients fold below zero before a hand-set theta_max —
    // passes every check up to here yet zeroes out its whole FOV at query
    // time. Reject the endpoint here instead. Runs after the distortion
    // consistency/finite-coeff checks so distortTheta only sees vetted
    // coefficients; models built via detail::defaultFisheyeThetaMax /
    // updateThetaMax always pass (their caps stop short of the pole and
    // inside the polynomial's positive monotone range).
    T theta_d_at_cap = T(0);
    if (distortTheta<T>(distortion, projection.theta_max, theta_d_at_cap) != StatusCode::OK || theta_d_at_cap <= T(0))
    {
      return StatusCode::INVALID_MODEL;
    }
    // For the polynomial families additionally require the polynomial to be
    // non-decreasing AT the cap. A single-fold polynomial (dominant negative
    // k1) whose cap sits past the fold has a positive endpoint value but a
    // negative endpoint slope; such a model maps distinct thetas to one
    // radius inside its declared FOV, so the forward emits pixels the
    // bracketed inverse then rejects (OUT_OF_FOV) or resolves to the wrong
    // branch. One derivative evaluation keeps this affordable per point.
    // ">= 0" not "> 0": caps produced by updateThetaMax may sit within one
    // scan step of the true fold, where the slope is a small positive value —
    // only a definitively negative slope is proof of a cap past the fold.
    // Interior folds that recover by the cap (dip-and-recover polynomials)
    // are invisible to any endpoint test; the full-tier scan below catches
    // those.
    if (detail::isPolynomialFisheyeDistortion(distortion.type))
    {
      T derivative_at_cap = T(0);
      if (distortThetaDerivative<T>(distortion, projection.theta_max, derivative_at_cap) != StatusCode::OK || derivative_at_cap < T(0))
      {
        return StatusCode::INVALID_MODEL;
      }
    }
  }
  if (projection.type == ProjectionModelType::OMNIDIRECTIONAL && distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if ((projection.type == ProjectionModelType::DOUBLE_SPHERE ||
       projection.type == ProjectionModelType::EUCM) &&
      distortion.space == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }

  if (distortion.has_tilt)
  {
    for (const T value : distortion.tilt_matrix)
    {
      if (!std::isfinite(value))
      {
        return StatusCode::INVALID_MODEL;
      }
    }
    for (const T value : distortion.inv_tilt_matrix)
    {
      if (!std::isfinite(value))
      {
        return StatusCode::INVALID_MODEL;
      }
    }
  }

  return StatusCode::OK;
}

// FULL-tier validation: what the public validateCameraModel /
// validateCameraModel64 execute. Query tier plus the scan-based certification
// that a polynomial-fisheye theta_max sits inside the polynomial's positive
// monotone range, so "validate OK" implies the forward map is injective on
// [0, theta_max] and the bracketed inverse can reach every pixel the forward
// emits (to scan resolution). Endpoint tests cannot see an interior fold that
// recovers before the cap; only the scan can, and at ~512 polynomial
// evaluations it belongs in the one-shot entry points (this oracle,
// ValidatedCameraModel, the LUT / remap builders, factories, calibration),
// not in the per-point guard above.
//
// The acceptance slack is two scan steps: "safe" is the last grid sample
// whose slope was still positive, so a legitimate cap (e.g. produced by
// updateThetaMax at one precision and re-validated at the other, where the
// grid rounds differently) may exceed it by up to one step; a second step
// absorbs the seed caps pinned at pi (getDefaultSeed) sitting just past the
// scan cap of pi - 1e-4. A cap beyond the fold by more than that is a real
// contract violation, not grid noise.
template <typename T>
inline StatusCode validateCameraModelImpl(const CameraModelT<T> &model)
{
  const StatusCode query_status = validateCameraModelQueryImpl<T>(model);
  if (query_status != StatusCode::OK)
  {
    return query_status;
  }

  if (model.projection.type == ProjectionModelType::FISHEYE_THETA &&
      detail::isPolynomialFisheyeDistortion(model.distortion.type))
  {
    const T safe = estimateSafeThetaMaxForPolynomialFisheye<T>(model.distortion);
    const T slack = T(2) * fisheyeThetaScanStep<T>();
    if (model.projection.theta_max > safe + slack)
    {
      return StatusCode::INVALID_MODEL;
    }
  }

  return StatusCode::OK;
}

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__MODEL__VALIDATE_IMPL_HPP
