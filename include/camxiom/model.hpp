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

[[nodiscard]] bool isFisheyeDistortionModel(const std::string &model_name);
[[nodiscard]] bool isRationalDistortionModel(const std::string &model_name);
[[nodiscard]] DistortionModelType parseDistortionModelType(const std::string &model_name);
[[nodiscard]] ProjectionModelType chooseProjectionModelType(
  DistortionModelType distortion_model_type
);

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

/// Create a copy of the model with distortion removed (coeffs zeroed, type=NONE).
[[nodiscard]] CameraModel makeDistortionFree(const CameraModel &model);

/// Recalculate projection.theta_max from current distortion coefficients.
/// This must be called after distortion coefficients are modified (e.g. by an
/// optimizer) so that the inverse projection (pixelToRay) uses an accurate
/// monotonic range for the theta solver.  For non-fisheye models this is a
/// no-op.
void updateThetaMax(CameraModel &model);

}  // namespace camxiom

#endif  // CAMXIOM__MODEL_HPP
