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

#ifndef CAMXIOM__INTERNAL__ANGLE_LAW_HPP
#define CAMXIOM__INTERNAL__ANGLE_LAW_HPP

// The radial function theta -> theta_d of a fisheye camera model.
//
// A fisheye model names its projection law in ProjectionModelT::type
// (equidistant, equisolid, stereographic, orthographic) and an optional angle
// correction in DistortionModelT::type (KB4 / KB8, on the equidistant law
// only). The kernels evaluate the two together as one function of theta;
// AngleLaw names that function so a kernel is chosen once per model instead
// of reading both enums at every point.
//
// STABILITY NOTE: like internal/constants.hpp, this header lives under
// internal/ and is EXCLUDED from the public API snapshot, but it is installed
// because a public header (projection_template.hpp) uses it. Treat it as an
// implementation detail with no compatibility guarantee.

#include "camxiom/types.hpp"

#include <cstdint>

namespace camxiom::detail_impl
{

enum class AngleLaw : std::uint8_t {
  INVALID = 0U,   ///< Not a fisheye law, or a correction the law does not take.
  EQUIDISTANT,    ///< theta_d = theta
  EQUISOLID,      ///< theta_d = 2 sin(theta / 2)
  STEREOGRAPHIC,  ///< theta_d = 2 tan(theta / 2)
  ORTHOGRAPHIC,   ///< theta_d = sin(theta)
  KB4,            ///< Equidistant law with the 4-coefficient KB correction.
  KB8             ///< Equidistant law with the 8-coefficient KB correction.
};

constexpr AngleLaw angleLawOf(
  const ProjectionModelType projection, const DistortionModelType distortion
) noexcept
{
  switch (projection)
  {
    case ProjectionModelType::EQUIDISTANT:
      switch (distortion)
      {
        case DistortionModelType::NONE:
          return AngleLaw::EQUIDISTANT;
        case DistortionModelType::KB4:
          return AngleLaw::KB4;
        case DistortionModelType::KB8:
          return AngleLaw::KB8;
        default:
          return AngleLaw::INVALID;
      }
    case ProjectionModelType::EQUISOLID:
      return distortion == DistortionModelType::NONE ? AngleLaw::EQUISOLID : AngleLaw::INVALID;
    case ProjectionModelType::STEREOGRAPHIC:
      return distortion == DistortionModelType::NONE ? AngleLaw::STEREOGRAPHIC : AngleLaw::INVALID;
    case ProjectionModelType::ORTHOGRAPHIC:
      return distortion == DistortionModelType::NONE ? AngleLaw::ORTHOGRAPHIC : AngleLaw::INVALID;
    default:
      return AngleLaw::INVALID;
  }
}

/// The KB corrections have coefficients and a sampled monotone range; the
/// plain laws have neither.
constexpr bool isPolynomialAngleLaw(const AngleLaw law) noexcept
{
  return law == AngleLaw::KB4 || law == AngleLaw::KB8;
}

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__INTERNAL__ANGLE_LAW_HPP
