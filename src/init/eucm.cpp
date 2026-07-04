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

#include "camxiom/init/eucm.hpp"

#include "camxiom/types64.hpp"
#include "init/seeded_planar_init.hpp"

#include <Eigen/Core>

#include <cmath>
#include <utility>

// EUCM init (MS1-6): heuristic Phase A seed + locked-(alpha, beta) Phase B
// polish. The whole Phase A/B flow lives in the shared skeleton
// (init/seeded_planar_init.cpp); this file only supplies the EUCM seed model
// and the alpha/beta finiteness check, and unpacks the outcome.

namespace camxiom::init
{

namespace
{

/// Heuristic seed for the EUCM blending factor alpha. Khomutenko 2015
/// reports real-camera calibrated values around 0.5 for typical fisheye
/// lenses; this is also the symmetric midpoint of the (0, 1) parameter range
/// and is used as the locked Phase B value.
constexpr double kAlphaSeed = 0.5;

/// Heuristic seed for the EUCM squish factor beta. Khomutenko 2015 reports
/// real-camera calibrated values close to 1.0 for typical lenses (1.0 = no
/// squish, i.e. pure unified camera model); used as the locked Phase B
/// value.
constexpr double kBetaSeed = 1.0;

/// Build the camxiom CameraModel64 seed used during Phase A and Phase B.
///
/// EUCM projection with the supplied (fx, fy, cx, cy), alpha, beta.
/// Distortion is held at NONE for pure EUCM (plane-space distortion is
/// deferred to MS2). theta_max is left at the default (pi/2); EUCM's alpha
/// and beta together capture the wide field of view.
camxiom::CameraModel64 makeEUCMModel(
  const double fx, const double fy, const double cx, const double cy
)
{
  camxiom::CameraModel64 model{};
  model.intrinsics.fx = fx;
  model.intrinsics.fy = fy;
  model.intrinsics.cx = cx;
  model.intrinsics.cy = cy;
  model.intrinsics.skew = 0.0;

  model.projection.type = camxiom::ProjectionModelType::EUCM;
  model.projection.alpha = kAlphaSeed;
  model.projection.beta = kBetaSeed;

  model.distortion.type = camxiom::DistortionModelType::NONE;
  model.distortion.space = camxiom::DistortionSpace::NONE;
  model.distortion.count = 0U;

  return model;
}

/// Mirrors MS1-6's phaseBLooksUsable model-specific part: checks
/// projection.alpha and projection.beta.
bool alphaBetaFinite(const camxiom::CameraModel &cam)
{
  return std::isfinite(cam.projection.alpha) && std::isfinite(cam.projection.beta);
}

}  // namespace

StatusCode estimateEUCMInit(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  EUCMInitResult &result_out
)
{
  detail::SeededPlanarInitOutcome outcome;
  const StatusCode status = detail::runSeededPlanarInit(
    views, image_width, image_height, &makeEUCMModel, &alphaBetaFinite, outcome
  );
  if (status != StatusCode::OK)
  {
    return status;
  }

  result_out.K = outcome.K;
  result_out.alpha = outcome.alpha;
  result_out.beta = outcome.beta;
  result_out.R_per_view = std::move(outcome.R_per_view);
  result_out.t_per_view = std::move(outcome.t_per_view);
  return StatusCode::OK;
}

}  // namespace camxiom::init
