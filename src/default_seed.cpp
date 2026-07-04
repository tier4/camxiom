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

#include "camxiom/default_seed.hpp"

#include "camxiom/internal/constants.hpp"

namespace camxiom
{

namespace
{

/// Apply the common intrinsics shared by all five magic seeds:
/// principal point at the image centre, zero skew. Computed in double
/// then narrowed to the float intrinsics fields.
void setCommonIntrinsics(CameraModel &model, int image_width, int image_height)
{
  const double w = static_cast<double>(image_width);
  const double h = static_cast<double>(image_height);
  model.intrinsics.cx = static_cast<float>(w / 2.0);
  model.intrinsics.cy = static_cast<float>(h / 2.0);
  model.intrinsics.skew = 0.0f;
}

/// Set fx = fy from a double-precision focal heuristic.
void setIsotropicFocal(CameraModel &model, double focal)
{
  model.intrinsics.fx = static_cast<float>(focal);
  model.intrinsics.fy = static_cast<float>(focal);
}

}  // namespace

CameraModel getDefaultSeed(ProjectionModelType model_type, int image_width, int image_height)
{
  // Invalid input: return a default-constructed sentinel. Its
  // projection.type is UNKNOWN, which validateCameraModel rejects.
  if (image_width <= 0 || image_height <= 0)
  {
    return CameraModel{};
  }

  const double h = static_cast<double>(image_height);

  CameraModel model{};

  switch (model_type)
  {
    case ProjectionModelType::PINHOLE: {
      model.projection.type = ProjectionModelType::PINHOLE;
      setCommonIntrinsics(model, image_width, image_height);
      setIsotropicFocal(model, h / 2.0);
      // Distortion: NONE / NONE / count 0 / coeffs all zero (struct defaults).
      break;
    }

    case ProjectionModelType::FISHEYE_THETA: {
      model.projection.type = ProjectionModelType::FISHEYE_THETA;
      // theta_max = pi. constants::kPiF is bit-identical to detail::kPi
      // used by validateCameraModel, so theta_max <= detail::kPi holds
      // (the bound is inclusive).
      model.projection.theta_max = constants::kPiF;
      setCommonIntrinsics(model, image_width, image_height);
      setIsotropicFocal(model, h / constants::kPi);
      // KB4 angle distortion with k1..k4 = 0. Mirrors kb4_fisheye.cpp's
      // makeSeedModel use_kb4=true branch. Only fields differing from the
      // struct defaults are set; tilt matrices are already identity.
      model.distortion.type = DistortionModelType::KB4;
      model.distortion.space = DistortionSpace::ANGLE;
      model.distortion.coeffs.fill(0.0f);
      model.distortion.count = 4U;
      model.distortion.is_rational = false;
      model.distortion.has_thin_prism = false;
      model.distortion.has_tilt = false;
      break;
    }

    case ProjectionModelType::OMNIDIRECTIONAL: {
      model.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
      model.projection.xi = 1.0f;
      setCommonIntrinsics(model, image_width, image_height);
      setIsotropicFocal(model, h / 2.0);
      // Distortion: NONE / NONE / count 0 / coeffs all zero (struct defaults).
      break;
    }

    case ProjectionModelType::DOUBLE_SPHERE: {
      model.projection.type = ProjectionModelType::DOUBLE_SPHERE;
      model.projection.xi = -0.2f;
      model.projection.alpha = 0.5f;
      setCommonIntrinsics(model, image_width, image_height);
      setIsotropicFocal(model, h / 2.0);
      // Distortion: NONE / NONE / count 0 / coeffs all zero (struct defaults).
      break;
    }

    case ProjectionModelType::EUCM: {
      model.projection.type = ProjectionModelType::EUCM;
      model.projection.alpha = 0.5f;
      model.projection.beta = 1.0f;
      setCommonIntrinsics(model, image_width, image_height);
      setIsotropicFocal(model, h / 2.0);
      // Distortion: NONE / NONE / count 0 / coeffs all zero (struct defaults).
      break;
    }

    case ProjectionModelType::UNKNOWN: {
      // Unrecognised model: detectable sentinel (projection.type == UNKNOWN).
      return CameraModel{};
    }
  }

  return model;
}

}  // namespace camxiom
