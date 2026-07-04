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

#ifndef CAMXIOM__INIT__SEEDED_PLANAR_INIT_HPP
#define CAMXIOM__INIT__SEEDED_PLANAR_INIT_HPP

// Shared Phase A/B skeleton for the heuristic-seeded planar init estimators
// (MEI omnidirectional / double sphere / EUCM). Internal header, not
// installed.
//
// These three estimators share the identical flow -- only the seed model
// construction and the projection-parameter finiteness check differ:
//   1. Input validation (>= 3 views, >= 4 finite correspondences per view);
//      no out-param mutation on failure.
//   2. Phase A heuristic seed: fx = fy = image_height/2, cx = image_width/2,
//      cy = image_height/2, skew = 0, model-specific projection seeds baked
//      into the caller's seed-model factory. (A Zhang-style IAC bootstrap is
//      mathematically invalid for these models: the board-to-lifted-plane
//      mapping is not homographic. See IMPLEMENTATION_NOTES.md MS1-4/5/6.)
//   3. Phase A per-view poses via estimatePoseDLT with the seeded model;
//      any failure aborts with DEGENERATE_CONFIG.
//   4. Phase B best-effort joint refinement via PnpSolver: K + extrinsics
//      free, projection parameters LOCKED at the seeds
//      (FIX_PROJECTION_PARAMS, per D29/D30). Bounds: principal point
//      <= 2*max(w,h), focal <= 10*h. Failures are silently absorbed and
//      Phase A's result is committed (D5/D28).
//   5. Final finite sweep -> NUMERIC_ERROR; only then fill the outcome.
//
// Previously this flow existed as three ~300-line verbatim copies; the
// per-model estimators now only build the seed model, check their own
// projection parameters, and unpack the outcome.

#include "camxiom/init/planar_observation.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <vector>

namespace camxiom::init::detail
{

/// Builds the model-specific CameraModel64 seed for the given intrinsics
/// (projection type + seed parameters baked in by the estimator).
using MakeSeedModelFn = camxiom::CameraModel64 (*)(double fx, double fy, double cx, double cy);

/// Model-specific part of the Phase B usability check: are the refined
/// model's ACTIVE projection parameters finite?
using ProjectionParamsFiniteFn = bool (*)(const camxiom::CameraModel &refined);

struct SeededPlanarInitOutcome
{
  Eigen::Matrix3d K{Eigen::Matrix3d::Identity()};
  /// Final projection parameters (Phase B refined when accepted, otherwise
  /// the Phase A seeds). The estimator picks the ones its model uses.
  double xi{0.0};
  double alpha{0.0};
  double beta{1.0};
  std::vector<Eigen::Matrix3d> R_per_view;
  std::vector<Eigen::Vector3d> t_per_view;
};

/// Run the shared Phase A/B flow. On any non-OK status `out` is untouched
/// (atomic mutation, same contract as the estimators themselves).
StatusCode runSeededPlanarInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  MakeSeedModelFn make_seed_model, ProjectionParamsFiniteFn projection_params_finite,
  SeededPlanarInitOutcome &out
);

}  // namespace camxiom::init::detail

#endif  // CAMXIOM__INIT__SEEDED_PLANAR_INIT_HPP
