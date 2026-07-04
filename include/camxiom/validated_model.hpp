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

#ifndef CAMXIOM__VALIDATED_MODEL_HPP
#define CAMXIOM__VALIDATED_MODEL_HPP

// ValidatedCameraModel — a *validated-once, immutable* camera model.
//
// The generic single-point API (rayToPixel / pixelToRay in projection.hpp)
// re-runs validateCameraModel() on every call. That is the right default for a
// free function that cannot trust its argument, but it re-pays the full
// consistency check on every point of a hot loop.
//
// ValidatedCameraModel makes "valid by construction" a property of the type:
//   * tryMake() runs validateCameraModel() exactly once and, on success,
//     resolves the per-model forward/inverse projection entry points once;
//   * the wrapped CameraModel is then immutable, so every subsequent
//     rayToPixel/pixelToRay skips both the re-validation and the dispatch
//     switch and calls straight into the resolved entry point.
//
// It is a NON-BREAKING ADDITION: the generic rayToPixel(CameraModel, ...) /
// pixelToRay(...) functions are unchanged and remain the trust-nothing default.
// Adopting ValidatedCameraModel is opt-in; results are bit-for-bit identical to
// the generic path for any model that validates.
//
// Precision: this covers the float32 runtime API (the real-time hot path). The
// double calibration path can gain an analogous ValidatedCameraModel64 later
// using the same shape if a double single-point hot loop appears.

#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <optional>

namespace camxiom
{

class ValidatedCameraModel
{
public:
  // Validate `model` once (identical criteria to validateCameraModel). Returns
  // an engaged optional holding an immutable, ready-to-project wrapper on
  // success, or std::nullopt if the model fails validation. Callers who need
  // the specific rejection reason can call validateCameraModel(model) directly.
  [[nodiscard]] static std::optional<ValidatedCameraModel> tryMake(const CameraModel &model);

  // The validated, immutable model. Never changes after construction.
  const CameraModel &get() const noexcept { return model_; }

  ProjectionModelType projectionType() const noexcept { return model_.projection.type; }

  // Hot-path projection: no per-call re-validation, no dispatch switch. For a
  // model that validated, these return exactly what the generic
  // rayToPixel/pixelToRay would return for the same input.
  PixelResult rayToPixel(const Eigen::Vector3f &ray_direction) const;
  PixelResult rayToPixel(float x_direction, float y_direction, float z_direction) const;
  /// Convenience overload for Ray3. Projection is central: only ray.direction
  /// is used and ray.origin is IGNORED (assumed to be the camera center).
  PixelResult rayToPixel(const Ray3 &ray) const;

  RayResult pixelToRay(const Pixel2 &pixel, const SolverOptions &solver_options = SolverOptions{})
    const;
  RayResult pixelToRay(float u, float v, const SolverOptions &solver_options = SolverOptions{})
    const;

private:
  // Resolved-once per-model entry points. The pointee functions are an internal
  // implementation detail (src/detail/projection_models.hpp); only their public
  // signature is named here, the concrete values are bound in tryMake().
  using ForwardFn = PixelResult (*)(const CameraModel &, const Eigen::Vector3f &);
  using InverseFn = RayResult (*)(const CameraModel &, const Pixel2 &, const SolverOptions &);

  ValidatedCameraModel(const CameraModel &model, ForwardFn forward, InverseFn inverse)
  : model_(model), forward_(forward), inverse_(inverse)
  {
  }

  CameraModel model_{};
  ForwardFn forward_{nullptr};
  InverseFn inverse_{nullptr};
};

}  // namespace camxiom

#endif  // CAMXIOM__VALIDATED_MODEL_HPP
