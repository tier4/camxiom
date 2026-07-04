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
  const DistortionModelType model_type = parseDistortionModelType(model_name);
  return model_type == DistortionModelType::OPENCV_FISHEYE4 ||
         model_type == DistortionModelType::EQUIDISTANT || model_type == DistortionModelType::KB4 ||
         model_type == DistortionModelType::KB8 || model_type == DistortionModelType::EQUISOLID ||
         model_type == DistortionModelType::STEREOGRAPHIC ||
         model_type == DistortionModelType::ORTHOGRAPHIC;
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
  // kb8, equisolid, stereographic, orthographic, omnidirectional — are
  // already covered by the compatibility aliases below.) Note the ROS name
  // "equidistant" denotes the 4-coefficient OpenCV fisheye model, so the
  // coefficient-free ideal mapping uses the distinct canonical name
  // "ideal_equidistant".
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
  if (model == "opencv_fisheye4")
  {
    return DistortionModelType::OPENCV_FISHEYE4;
  }
  if (model == "ideal_equidistant")
  {
    return DistortionModelType::EQUIDISTANT;
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
  if (model == "equidistant" || model == "fisheye")
  {
    return DistortionModelType::OPENCV_FISHEYE4;
  }
  if (model == "kannala_brandt" || model == "kannala-brandt" || model == "kb4")
  {
    return DistortionModelType::KB4;
  }
  if (model == "kannala_brandt8" || model == "kannala-brandt8" || model == "kb8")
  {
    return DistortionModelType::KB8;
  }
  if (model == "equisolid" || model == "equisolid_angle")
  {
    return DistortionModelType::EQUISOLID;
  }
  if (model == "stereographic" || model == "stereographic_angle")
  {
    return DistortionModelType::STEREOGRAPHIC;
  }
  if (model == "orthographic" || model == "orthographic_angle")
  {
    return DistortionModelType::ORTHOGRAPHIC;
  }
  if (model == "omnidirectional" || model == "omni")
  {
    return DistortionModelType::OMNIDIRECTIONAL;
  }
  return DistortionModelType::UNKNOWN;
}

ProjectionModelType chooseProjectionModelType(const DistortionModelType distortion_model_type)
{
  switch (distortion_model_type)
  {
    case DistortionModelType::OPENCV_FISHEYE4:
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
    case DistortionModelType::EQUIDISTANT:
    case DistortionModelType::EQUISOLID:
    case DistortionModelType::STEREOGRAPHIC:
    case DistortionModelType::ORTHOGRAPHIC:
      return ProjectionModelType::FISHEYE_THETA;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::THIN_PRISM12:
    case DistortionModelType::TILTED14:
    case DistortionModelType::NONE:
      return ProjectionModelType::PINHOLE;
    case DistortionModelType::OMNIDIRECTIONAL:
      return ProjectionModelType::OMNIDIRECTIONAL;
    case DistortionModelType::UNKNOWN:
      break;
  }
  return ProjectionModelType::UNKNOWN;
}

}  // namespace camxiom
