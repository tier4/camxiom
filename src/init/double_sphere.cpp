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

#include "camxiom/init/double_sphere.hpp"

#include "camxiom/types64.hpp"
#include "init/seeded_planar_init.hpp"

#include <Eigen/Core>

#include <cmath>
#include <utility>

// Double-sphere init (MS1-5): heuristic Phase A seed + locked-(xi, alpha)
// Phase B polish. The whole Phase A/B flow lives in the shared skeleton
// (init/seeded_planar_init.cpp); this file only supplies the DS seed model
// and the xi/alpha finiteness check, and unpacks the outcome.

namespace camxiom::init
{

namespace
{

/// Heuristic seeds for the double-sphere projection parameters
/// (Usenko 2018 typical central values).
constexpr double kXiSeed = -0.2;
constexpr double kAlphaSeed = 0.5;

/// Build the camxiom CameraModel64 seed used during Phase A and Phase B.
///
/// DOUBLE_SPHERE projection with the supplied (fx, fy, cx, cy), xi, alpha.
/// Distortion is held at NONE for pure DS (plane-space distortion is
/// deferred to MS2). theta_max is left at the default (pi/2); DS's xi and
/// alpha together capture the wide field of view.
camxiom::CameraModel64 makeDSModel(
  const double fx, const double fy, const double cx, const double cy
)
{
  camxiom::CameraModel64 model{};
  model.intrinsics.fx = fx;
  model.intrinsics.fy = fy;
  model.intrinsics.cx = cx;
  model.intrinsics.cy = cy;
  model.intrinsics.skew = 0.0;

  model.projection.type = camxiom::ProjectionModelType::DOUBLE_SPHERE;
  model.projection.xi = kXiSeed;
  model.projection.alpha = kAlphaSeed;

  model.distortion.type = camxiom::DistortionModelType::NONE;
  model.distortion.space = camxiom::DistortionSpace::NONE;
  model.distortion.count = 0U;

  return model;
}

/// Mirrors MS1-5's phaseBLooksUsable model-specific part: checks
/// projection.xi and projection.alpha.
bool xiAlphaFinite(const camxiom::CameraModel &cam)
{
  return std::isfinite(cam.projection.xi) && std::isfinite(cam.projection.alpha);
}

}  // namespace

StatusCode estimateDSInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  DSInitResult &result_out
)
{
  detail::SeededPlanarInitOutcome outcome;
  const StatusCode status = detail::runSeededPlanarInit(
    views, image_width, image_height, &makeDSModel, &xiAlphaFinite, outcome
  );
  if (status != StatusCode::OK)
  {
    return status;
  }

  result_out.K = outcome.K;
  result_out.xi = outcome.xi;
  result_out.alpha = outcome.alpha;
  result_out.R_per_view = std::move(outcome.R_per_view);
  result_out.t_per_view = std::move(outcome.t_per_view);
  return StatusCode::OK;
}

}  // namespace camxiom::init
