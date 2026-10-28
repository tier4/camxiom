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

bool isFisheyeDistortionModel(const std::string &model_name) { return model_name == "equidistant"; }

bool isRationalDistortionModel(const std::string &model_name)
{
  return model_name == "rational_polynomial";
}

DistortionModelType parseDistortionModelType(const std::string &model_name)
{
  const std::string model = detail::toLowerCopy(model_name);
  for (const DistortionModelType type :
       {DistortionModelType::NONE, DistortionModelType::RADTAN4, DistortionModelType::RADTAN5,
        DistortionModelType::RATIONAL8, DistortionModelType::THIN_PRISM12,
        DistortionModelType::TILTED14, DistortionModelType::KB4, DistortionModelType::KB8})
  {
    if (model == toString(type))
    {
      return type;
    }
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
        ProjectionModelType::DOUBLE_SPHERE, ProjectionModelType::EUCM,
        ProjectionModelType::EQUIRECTANGULAR, ProjectionModelType::CYLINDRICAL})
  {
    if (model == toString(type))
    {
      return type;
    }
  }
  return ProjectionModelType::UNKNOWN;
}

}  // namespace camxiom
