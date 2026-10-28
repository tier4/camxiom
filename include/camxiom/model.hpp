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

#ifndef CAMXIOM__MODEL_HPP
#define CAMXIOM__MODEL_HPP

#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <string>

namespace camxiom
{

/// True when an interchange distortion_model name (e.g. ROS CameraInfo)
/// denotes a fisheye law: the KB names ("equidistant", "fisheye", "kb4",
/// "kb8", ...) and the plain law names ("ideal_equidistant", "equisolid",
/// "stereographic", "orthographic").
[[nodiscard]] bool isFisheyeDistortionModel(const std::string &model_name);
[[nodiscard]] bool isRationalDistortionModel(const std::string &model_name);
/// Distortion (lens correction) names: the canonical toString spellings plus
/// the interchange aliases ("plumb_bob", "rational_polynomial", and the ROS /
/// OpenCV names of KB4: "equidistant", "fisheye"). Projection-law names such
/// as "equisolid" are not distortions and parse to UNKNOWN.
[[nodiscard]] DistortionModelType parseDistortionModelType(const std::string &model_name);
/// Inverse of toString(ProjectionModelType); anything else parses to UNKNOWN.
[[nodiscard]] ProjectionModelType parseProjectionModelType(const std::string &model_name);

/// Full model-invariant check: structural consistency (types, spaces, counts,
/// flags, finiteness, per-model parameter ranges) PLUS certification that a
/// polynomial-fisheye theta_max sits inside the distortion polynomial's
/// positive monotone range. OK therefore implies the forward map is injective
/// on [0, theta_max] and pixelToRay can invert every pixel rayToPixel emits
/// (up to solver tolerances; the monotone range is certified at ~pi/512
/// sample resolution).
///
/// Cost note: the monotone-range certification evaluates the polynomial a few
/// hundred times, so this function is meant for one-shot use (validating an
/// incoming model, asserting a hand-built one). The per-point query APIs
/// (rayToPixel, pixelToRay, Jacobians, batch) guard themselves with the
/// structural subset only — identical across all of them — so a model that
/// skips this oracle still projects consistently on every path, but only OK
/// from THIS function certifies the forward/inverse round-trip. Models built
/// by the library itself (makeCameraModel, calibrate(), updateThetaMax) pass
/// by construction. For validate-once hot loops, use ValidatedCameraModel.
[[nodiscard]] StatusCode validateCameraModel(const CameraModel &model);

/// Double-precision counterpart of validateCameraModel (same two-tier
/// contract). Declared here beside the float validator (it used to live in
/// camxiom/projection64.hpp, an easy place to miss when looking for model
/// helpers).
[[nodiscard]] StatusCode validateCameraModel64(const CameraModel64 &model);

/// Create a copy of the model with its lens correction removed: distortion
/// type NONE, coefficients zeroed, and the projection law (including xi,
/// alpha, beta and theta_max) unchanged. A stereographic model stays
/// stereographic; an equidistant model with KB4 becomes the plain
/// equidistant law.
[[nodiscard]] CameraModel makeDistortionFree(const CameraModel &model);

/// What bounds the usable angular range, independently of a declared FOV.
enum class FisheyeDomainLimit : std::uint8_t {
  UNKNOWN = 0U,
  ANGULAR_LIMIT,       ///< Geometric endpoint (pi, or pi/2 for orthographic).
  MONOTONICITY_LIMIT,  ///< Sampled radius is negative or its slope is non-positive.
  NUMERIC_LIMIT        ///< Sampled radius or slope cannot be evaluated finitely.
};

struct FisheyeDomain64
{
  StatusCode status{StatusCode::INVALID_INPUT};
  double safe_theta_max{0.0};  ///< Usable cap in radians; zero on failure.
  double theta_limit{0.0};     ///< Geometric endpoint or first failing scan sample (radians).
  FisheyeDomainLimit limit{FisheyeDomainLimit::UNKNOWN};

  constexpr bool ok() const { return status == StatusCode::OK; }
};

/// Inspect a fisheye-law model (isFisheyeProjection) without changing it. The declared theta_max
/// is ignored, even if non-finite or outside the current monotone range.
/// All other model invariants must hold; errors leave both angles zero and
/// limit UNKNOWN. Non-fisheye models return INVALID_MODEL.
///
/// Polynomial families share updateThetaMax64's scan: 512 intervals from
/// 1e-3 to approximately pi - 1e-4 radians, including the start sample.
/// safe_theta_max is the last finite, non-negative, positive-slope sample;
/// theta_limit is the first failing sample, NOT an exact polynomial root.
/// Without a failing sample the limit is ANGULAR_LIMIT at pi. Coefficient-free
/// laws use their geometric endpoint with a 1e-4 rad inset for safe_theta_max.
/// This is a sampled estimate: folds narrower than a scan step or below
/// the start sample can be missed. No usable positive sample is an error.
/// Recompute after changing coefficients; observation gates and evaluation
/// margins are caller policy, not inputs to this query.
[[nodiscard]] FisheyeDomain64 getFisheyeDomain64(const CameraModel64 &model);

/// Recalculate projection.theta_max from current distortion coefficients.
/// This must be called after distortion coefficients are modified (e.g. by an
/// optimizer) so that the inverse projection (pixelToRay) uses an accurate
/// monotonic range for the theta solver. Overwrites any caller-selected cap.
/// Non-fisheye models reset to their default angular cap.
void updateThetaMax(CameraModel &model);

/// Recompute the derived field of view without narrowing coefficients to float.
/// For valid fisheye models, assigns getFisheyeDomain64(model).safe_theta_max.
void updateThetaMax64(CameraModel64 &model);

[[nodiscard]] CameraModel64 makeDistortionFree64(const CameraModel64 &model);

}  // namespace camxiom

#endif  // CAMXIOM__MODEL_HPP
