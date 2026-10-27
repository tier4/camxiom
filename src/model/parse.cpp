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

#include "camxiom/model.hpp"
#include "model/internal.hpp"

namespace camxiom
{

bool isFisheyeDistortionModel(const std::string &model_name)
{
  return isFisheyeProjection(detail::readInterchangeModelName(model_name).projection);
}

bool isRationalDistortionModel(const std::string &model_name)
{
  const DistortionModelType model_type = parseDistortionModelType(model_name);
  return model_type == DistortionModelType::RATIONAL8 ||
         model_type == DistortionModelType::THIN_PRISM12 ||
         model_type == DistortionModelType::TILTED14;
}

DistortionModelType parseDistortionModelType(const std::string &model_name)
{
  const std::string model = detail::toLowerCopy(model_name);

  // Canonical camxiom names first: every toString(DistortionModelType)
  // output must parse back to the same enum so persisted model strings
  // survive a round trip. (The remaining canonical names — radtan4/5, kb4,
  // kb8 — are already covered by the compatibility aliases below.)
  if (model == "none")
  {
    return DistortionModelType::NONE;
  }
  if (model == "rational8")
  {
    return DistortionModelType::RATIONAL8;
  }
  if (model == "thin_prism12")
  {
    return DistortionModelType::THIN_PRISM12;
  }
  if (model == "tilted14")
  {
    return DistortionModelType::TILTED14;
  }

  if (model == "plumb_bob")
  {
    return DistortionModelType::RADTAN5;
  }
  if (model == "rational_polynomial")
  {
    return DistortionModelType::RATIONAL8;
  }
  if (model == "brown_conrady" || model == "brown-conrady")
  {
    return DistortionModelType::RADTAN5;
  }
  if (model == "radtan5")
  {
    return DistortionModelType::RADTAN5;
  }
  if (model == "radtan4")
  {
    return DistortionModelType::RADTAN4;
  }
  if (model == "thin_prism_fisheye")
  {
    return DistortionModelType::THIN_PRISM12;
  }
  if (model == "tilted")
  {
    return DistortionModelType::TILTED14;
  }
  // OpenCV's fisheye model, θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸), is the
  // same function with the same coefficient order as KB4, so its names
  // (ROS "equidistant", OpenCV "fisheye") are read as KB4.
  if (model == "kannala_brandt" || model == "kannala-brandt" || model == "kb4" ||
      model == "equidistant" || model == "fisheye" || model == "opencv_fisheye4")
  {
    return DistortionModelType::KB4;
  }
  if (model == "kannala_brandt8" || model == "kannala-brandt8" || model == "kb8")
  {
    return DistortionModelType::KB8;
  }
  return DistortionModelType::UNKNOWN;
}

ProjectionModelType parseProjectionModelType(const std::string &model_name)
{
  const std::string model = detail::toLowerCopy(model_name);
  for (const ProjectionModelType type :
       {ProjectionModelType::PINHOLE, ProjectionModelType::EQUIDISTANT,
        ProjectionModelType::EQUISOLID, ProjectionModelType::STEREOGRAPHIC,
        ProjectionModelType::ORTHOGRAPHIC, ProjectionModelType::OMNIDIRECTIONAL,
        ProjectionModelType::DOUBLE_SPHERE, ProjectionModelType::EUCM})
  {
    if (model == toString(type))
    {
      return type;
    }
  }
  return ProjectionModelType::UNKNOWN;
}

namespace detail
{

InterchangeModel readInterchangeModelName(const std::string &model_name)
{
  const std::string model = toLowerCopy(model_name);

  // The coefficient-free fisheye laws. ROS has no name for the plain
  // equidistant law ("equidistant" is its KB4 polynomial), so camxiom uses
  // "ideal_equidistant" for it.
  if (model == "ideal_equidistant")
  {
    return {ProjectionModelType::EQUIDISTANT, DistortionModelType::NONE};
  }
  if (model == "equisolid" || model == "equisolid_angle")
  {
    return {ProjectionModelType::EQUISOLID, DistortionModelType::NONE};
  }
  if (model == "stereographic" || model == "stereographic_angle")
  {
    return {ProjectionModelType::STEREOGRAPHIC, DistortionModelType::NONE};
  }
  if (model == "orthographic" || model == "orthographic_angle")
  {
    return {ProjectionModelType::ORTHOGRAPHIC, DistortionModelType::NONE};
  }

  // Otherwise the name is a correction, and the correction implies the law
  // it is defined on: the KB polynomials on the equidistant law, the plane
  // corrections on the pinhole law.
  const DistortionModelType distortion = parseDistortionModelType(model);
  switch (distortion)
  {
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
      return {ProjectionModelType::EQUIDISTANT, distortion};
    case DistortionModelType::NONE:
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
      return {ProjectionModelType::PINHOLE, distortion};
    case DistortionModelType::UNKNOWN:
      break;
  }
  return {ProjectionModelType::UNKNOWN, DistortionModelType::UNKNOWN};
}

}  // namespace detail

}  // namespace camxiom
