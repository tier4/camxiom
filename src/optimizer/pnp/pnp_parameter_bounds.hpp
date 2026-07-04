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

#ifndef CAMXIOM__OPTIMIZER__PNP__PNP_PARAMETER_BOUNDS_HPP
#define CAMXIOM__OPTIMIZER__PNP__PNP_PARAMETER_BOUNDS_HPP

// Internal (non-installed) header — single source of truth for the box bounds
// applied to the FREE intrinsic / projection scalar parameters of a calibration
// solve. It lives under src/ (a PRIVATE include of camxiom_calib), so it is NOT
// part of the public API surface scanned by the Rel1a snapshot test.
//
// Two consumers share this descriptor so they can never drift apart (C5 ⑤):
//   * PnpSolver::setupProblem() turns the descriptor into Ceres
//     SetParameterLower/UpperBound calls (a behaviour-invariant refactor of the
//     bounds it previously hard-coded inline).
//   * The C5 near-bound diagnostic in calib/intrinsics.cpp queries the SAME
//     descriptor to decide which free parameters sit at (or near) a bound.
//
// rot / trans extrinsic bounds are intentionally NOT modelled here: they remain
// inline in setupProblem() (they are not part of the intrinsics near-bound
// diagnostic and depend only on the caller's PnpBound rot/trans scalars).

#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/types.hpp"

#include <algorithm>

namespace camxiom::optimizer::detail
{

/// Widen the default PnpBound upper limits for a real imager. The stock
/// 5000 px caps (PnpBound::createUpperBound) are too small for large sensors
/// and long lenses: principal point <= 2x the image extent, focal length <=
/// 10x the image height — or 10x the seed focal length when the caller's warm
/// start is already longer, so the seed itself can never start outside the
/// feasible box. Lower bounds keep their permissive defaults.
///
/// Single source of truth for this widening: calib::calibrate,
/// init::seededPlanarInit and calib::convertCameraModel all fit with free
/// intrinsics against the same default PnpBound, and hand-copied versions of
/// this fix had already drifted apart (convertCameraModel shipped without it,
/// silently clamping telephoto fits at 5000 px).
inline void widenDefaultPnpUpperBounds(
  PnpSolverOptions &options, const int image_width, const int image_height, const double seed_fx,
  const double seed_fy
)
{
  const double pp_upper =
    2.0 * std::max(static_cast<double>(image_width), static_cast<double>(image_height));
  options.upper_bound.principal_points = Eigen::Vector2d(pp_upper, pp_upper);

  const double default_f_upper = 10.0 * static_cast<double>(image_height);
  options.upper_bound.focal_lengths = Eigen::Vector2d(
    std::max(default_f_upper, 10.0 * seed_fx), std::max(default_f_upper, 10.0 * seed_fy)
  );
}

/// Number of free scalar projection parameters per model family, matching
/// the solver's Ceres parameter-block layout ([xi, alpha, beta]). Shared so
/// callers that reason about the fit's degrees of freedom (e.g. the
/// convertCameraModel overfit margin) cannot drift from the solver.
inline int projectionParamCount(const camxiom::ProjectionModelType type)
{
  switch (type)
  {
    case camxiom::ProjectionModelType::OMNIDIRECTIONAL:
      return 1;
    case camxiom::ProjectionModelType::DOUBLE_SPHERE:
      return 2;
    case camxiom::ProjectionModelType::EUCM:
      return 2;
    case camxiom::ProjectionModelType::PINHOLE:
    case camxiom::ProjectionModelType::FISHEYE_THETA:
    case camxiom::ProjectionModelType::UNKNOWN:
      break;
  }
  return 0;
}

/// A one-dimensional box bound on a single scalar parameter. `has_lower` /
/// `has_upper` distinguish a genuinely one-sided bound (e.g. EUCM beta, which
/// is only lower-bounded) from a two-sided one, so a near-bound check never
/// tests against a non-existent +/-inf side.
struct ScalarBound
{
  bool has_lower{false};
  bool has_upper{false};
  double lower{0.0};
  double upper{0.0};
};

/// Bounds for the scalar intrinsic / projection parameters, laid out to match
/// the solver's Ceres parameter blocks exactly:
///   focal_lengths[0..1]    = fx, fy
///   principal_points[0..1] = cx, cy
///   projection[0..2]       = xi, alpha, beta
/// A locked (fixed) or unbounded parameter keeps a default ScalarBound
/// (has_lower == has_upper == false).
struct CalibrationParameterBounds
{
  ScalarBound focal_lengths[2];
  ScalarBound principal_points[2];
  ScalarBound projection[3];
};

/// Interior margin keeping Double Sphere xi strictly inside its open (-1, 1)
/// interval where the forward map is well-conditioned (Usenko 2018).
inline constexpr double kDoubleSphereXiMargin = 1e-6;
/// EUCM beta is only physically meaningful strictly positive; keep a small
/// positive floor. beta has NO upper bound.
inline constexpr double kEucmBetaLower = 1e-6;

/// Build the box-bound descriptor for the FREE scalar parameters, mirroring
/// exactly what setupProblem() would otherwise apply inline. Pure function of
/// (model, lock flags, caller-supplied focal/pp PnpBound):
///   * focal / principal-point bounds come from `lower` / `upper` with the same
///     std::max(1.0, .) / std::max(0.0, .) lower clamps setupProblem() used;
///   * xi / alpha / beta bounds are the model-validity hard bounds (Double
///     Sphere / EUCM), independent of the caller's PnpBound.
/// Parameters held by `flags` (or without a bound, e.g. OMNIDIRECTIONAL xi and
/// all distortion coefficients) keep a default (unbounded) ScalarBound.
inline CalibrationParameterBounds computeCalibrationParameterBounds(
  const camxiom::CameraModel &model, PnpFlag flags, const PnpBound &lower, const PnpBound &upper
)
{
  CalibrationParameterBounds bounds;

  if (!hasFlag(flags, PnpFlag::FIX_FOCAL_LENGTHS))
  {
    for (int i = 0; i < 2; ++i)
    {
      bounds.focal_lengths[i].has_lower = true;
      bounds.focal_lengths[i].has_upper = true;
      bounds.focal_lengths[i].lower = std::max(1.0, lower.focal_lengths[i]);
      bounds.focal_lengths[i].upper = upper.focal_lengths[i];
    }
  }

  if (!hasFlag(flags, PnpFlag::FIX_PRINCIPAL_POINTS))
  {
    for (int i = 0; i < 2; ++i)
    {
      bounds.principal_points[i].has_lower = true;
      bounds.principal_points[i].has_upper = true;
      bounds.principal_points[i].lower = std::max(0.0, lower.principal_points[i]);
      bounds.principal_points[i].upper = upper.principal_points[i];
    }
  }

  // Projection hard bounds only exist for Double Sphere / EUCM, and only when
  // the projection block is free. OMNIDIRECTIONAL xi is deliberately unbounded;
  // PINHOLE / FISHEYE carry no projection parameters.
  if (!hasFlag(flags, PnpFlag::FIX_PROJECTION_PARAMS))
  {
    switch (model.projection.type)
    {
      case camxiom::ProjectionModelType::DOUBLE_SPHERE:
        bounds.projection[0] =
          ScalarBound{true, true, -1.0 + kDoubleSphereXiMargin, 1.0 - kDoubleSphereXiMargin};
        bounds.projection[1] = ScalarBound{true, true, 0.0, 1.0};
        break;
      case camxiom::ProjectionModelType::EUCM:
        bounds.projection[1] = ScalarBound{true, true, 0.0, 1.0};
        bounds.projection[2] = ScalarBound{true, false, kEucmBetaLower, 0.0};
        break;
      case camxiom::ProjectionModelType::PINHOLE:
      case camxiom::ProjectionModelType::FISHEYE_THETA:
      case camxiom::ProjectionModelType::OMNIDIRECTIONAL:
      case camxiom::ProjectionModelType::UNKNOWN:
        break;
    }
  }

  return bounds;
}

}  // namespace camxiom::optimizer::detail

#endif  // CAMXIOM__OPTIMIZER__PNP__PNP_PARAMETER_BOUNDS_HPP
