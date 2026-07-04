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

#include "camxiom/init/mei_omni.hpp"

#include "camxiom/types64.hpp"
#include "init/seeded_planar_init.hpp"

#include <Eigen/Core>

#include <cmath>
#include <utility>

// MEI omnidirectional init (MS1-4): heuristic Phase A seed + locked-xi
// Phase B polish. The whole Phase A/B flow lives in the shared skeleton
// (init/seeded_planar_init.cpp); this file only supplies the MEI seed model
// and the xi finiteness check, and unpacks the outcome.

namespace camxiom::init
{

namespace
{

/// Build the camxiom CameraModel64 seed used during Phase A and Phase B.
///
/// OMNIDIRECTIONAL projection with the supplied (fx, fy, cx, cy) and
/// xi = 1 (Mei 2007 canonical parabolic value). Distortion is held at NONE
/// for pure MEI (plane-space distortion is deferred to MS2). theta_max is
/// left at the default (pi/2); MEI's xi is what captures the wide field of
/// view.
constexpr double kXiSeed = 1.0;

camxiom::CameraModel64 makeMEIModel(
  const double fx, const double fy, const double cx, const double cy
)
{
  camxiom::CameraModel64 model{};
  model.intrinsics.fx = fx;
  model.intrinsics.fy = fy;
  model.intrinsics.cx = cx;
  model.intrinsics.cy = cy;
  model.intrinsics.skew = 0.0;

  model.projection.type = camxiom::ProjectionModelType::OMNIDIRECTIONAL;
  model.projection.xi = kXiSeed;

  model.distortion.type = camxiom::DistortionModelType::NONE;
  model.distortion.space = camxiom::DistortionSpace::NONE;
  model.distortion.count = 0U;

  return model;
}

/// Mirrors MS1-4's phaseBLooksUsable model-specific part: checks
/// projection.xi rather than distortion coefficients.
bool xiFinite(const camxiom::CameraModel &cam) { return std::isfinite(cam.projection.xi); }

}  // namespace

StatusCode estimateMEIInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  MEIInitResult &result_out
)
{
  detail::SeededPlanarInitOutcome outcome;
  const StatusCode status = detail::runSeededPlanarInit(
    views, image_width, image_height, &makeMEIModel, &xiFinite, outcome
  );
  if (status != StatusCode::OK)
  {
    return status;
  }

  result_out.K = outcome.K;
  result_out.xi = outcome.xi;
  result_out.R_per_view = std::move(outcome.R_per_view);
  result_out.t_per_view = std::move(outcome.t_per_view);
  return StatusCode::OK;
}

}  // namespace camxiom::init
