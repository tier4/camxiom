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

#ifndef CAMXIOM__MODEL__INTERNAL_HPP
#define CAMXIOM__MODEL__INTERNAL_HPP

#include "camxiom/model.hpp"
#include "detail/internal.hpp"
#include "distortion/angle_impl.hpp"
#include "model/distortion_aux.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <string>

namespace camxiom::detail
{
inline std::string toLowerCopy(const std::string &input)
{
  std::string lower = input;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return lower;
}

// hasAnyNonZero / setIdentity3x3 / computeTiltMatrices moved to the
// precision-agnostic model/distortion_aux.hpp (single source shared with the
// PnP solver write-back and the analytical double cost), together with
// rebuildDistortionAuxState.

inline DistortionSpace distortionSpaceFromType(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
      return DistortionSpace::NONE;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
      return DistortionSpace::PLANE;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return DistortionSpace::ANGLE;
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return DistortionSpace::NONE;
}

inline DistortionModelType classifyPlaneDistortionByCount(const std::size_t count)
{
  if (count >= 14U)
  {
    return DistortionModelType::TILTED14;
  }
  if (count >= 12U)
  {
    return DistortionModelType::THIN_PRISM12;
  }
  if (count >= 8U)
  {
    return DistortionModelType::RATIONAL8;
  }
  if (count >= 5U)
  {
    return DistortionModelType::RADTAN5;
  }
  if (count >= 4U)
  {
    return DistortionModelType::RADTAN4;
  }
  return DistortionModelType::NONE;
}

// requiredFiniteCoefficientCount moved to the precision-agnostic, type-only
// detail::requiredFiniteCoefficientCount(DistortionModelType) in
// detail/internal.hpp (#1 step 4), shared by the float and double validators.

// The monotone-range scan itself lives in distortion/angle_impl.hpp
// (detail_impl::estimateSafeThetaMaxForPolynomialFisheye) so the full-tier
// validator, the fit write-back, and the optimizer's mid-solve rescue all
// share one grid and one definition of "safe". This alias keeps the
// historical call sites readable.
template <typename T>
T estimateSafeThetaMaxForPolynomialFisheyeT(const DistortionModelT<T> &model)
{
  return detail_impl::estimateSafeThetaMaxForPolynomialFisheye<T>(model);
}

inline float estimateSafeThetaMaxForPolynomialFisheye(const DistortionModel &model)
{
  return estimateSafeThetaMaxForPolynomialFisheyeT<float>(model);
}

/// For the polynomial fisheye families theta_max is a quantity DERIVED from
/// the coefficients (defaultFisheyeThetaMax below): once a fit moves the
/// coefficients, a cap chosen for the seed polynomial can end up past the
/// fitted polynomial's fold, where validateCameraModel's endpoint check
/// rightly rejects the model as self-inconsistent. Shrink the cap back to
/// the trial polynomial's positive monotone range. Returns true when the cap
/// changed. No-op for non-polynomial models, whose theta_max is an
/// independent FOV declaration rather than a derived quantity.
///
/// This is a MID-SOLVE helper for the analytical batch cost, where theta_max
/// stays frozen at the seed value while the coefficients move each Evaluate
/// (theta_max is not a Ceres parameter). It must not be mistaken for the
/// final-model derivation: PnpSolver::writeBack() unconditionally reassigns
/// theta_max from the fitted coefficients via the public updateThetaMax()
/// (an overwrite, not a shrink — a deliberately narrow caller cap does NOT
/// survive a coefficient-fitting solve), which is why a post-writeBack
/// validation never needs this rescue.
template <typename T>
bool shrinkThetaMaxToPolynomialMonotoneRange(CameraModelT<T> &model)
{
  if (model.projection.type != ProjectionModelType::FISHEYE_THETA)
  {
    return false;
  }
  if (model.distortion.type != DistortionModelType::KB4 &&
      model.distortion.type != DistortionModelType::KB8 &&
      model.distortion.type != DistortionModelType::OPENCV_FISHEYE4)
  {
    return false;
  }
  const T safe = estimateSafeThetaMaxForPolynomialFisheyeT<T>(model.distortion);
  if (safe < model.projection.theta_max)
  {
    model.projection.theta_max = safe;
    return true;
  }
  return false;
}

inline float defaultFisheyeThetaMax(const DistortionModel &model)
{
  switch (model.type)
  {
    case DistortionModelType::ORTHOGRAPHIC:
      // theta->sin(theta) is monotonic only on [0, pi/2].
      return kHalfPi - 1e-4f;
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
      return estimateSafeThetaMaxForPolynomialFisheye(model);
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
      return kPi - 1e-4f;
    case DistortionModelType::NONE:
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::OMNIDIRECTIONAL:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return kHalfPi - 1e-4f;
}

}  // namespace camxiom::detail

#endif  // CAMXIOM__MODEL__INTERNAL_HPP
