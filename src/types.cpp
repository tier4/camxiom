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

#include "camxiom/types.hpp"

namespace camxiom
{

const char *toString(const StatusCode status_code)
{
  switch (status_code)
  {
    case StatusCode::OK:
      return "ok";
    case StatusCode::INVALID_INPUT:
      return "invalid_input";
    case StatusCode::INVALID_MODEL:
      return "invalid_model";
    case StatusCode::BEHIND_CAMERA:
      return "behind_camera";
    case StatusCode::OUT_OF_FOV:
      return "out_of_fov";
    case StatusCode::DOMAIN_ERROR:
      return "domain_error";
    case StatusCode::NON_CONVERGED:
      return "non_converged";
    case StatusCode::NUMERIC_ERROR:
      return "numeric_error";
    case StatusCode::DEGENERATE_CONFIG:
      return "degenerate_config";
    default:
      return "unknown";
  }
}

const char *toString(const DistortionModelType model_type)
{
  switch (model_type)
  {
    case DistortionModelType::NONE:
      return "none";
    case DistortionModelType::RADTAN4:
      return "radtan4";
    case DistortionModelType::RADTAN5:
      return "radtan5";
    case DistortionModelType::RATIONAL8:
      return "rational8";
    case DistortionModelType::THIN_PRISM12:
      return "thin_prism12";
    case DistortionModelType::TILTED14:
      return "tilted14";
    case DistortionModelType::OPENCV_FISHEYE4:
      return "opencv_fisheye4";
    case DistortionModelType::KB4:
      return "kb4";
    case DistortionModelType::KB8:
      return "kb8";
    case DistortionModelType::EQUIDISTANT:
      // Deliberately NOT "equidistant": the ROS distortion_model string
      // "equidistant" denotes the 4-coefficient OpenCV fisheye model, and
      // parseDistortionModelType keeps that compatibility mapping. The
      // coefficient-free ideal mapping gets a distinct canonical name so
      // every toString output parses back to the same enum.
      return "ideal_equidistant";
    case DistortionModelType::EQUISOLID:
      return "equisolid";
    case DistortionModelType::STEREOGRAPHIC:
      return "stereographic";
    case DistortionModelType::ORTHOGRAPHIC:
      return "orthographic";
    case DistortionModelType::OMNIDIRECTIONAL:
      return "omnidirectional";
    case DistortionModelType::UNKNOWN:
      break;
  }
  return "unknown";
}

const char *toString(const ProjectionModelType model_type)
{
  switch (model_type)
  {
    case ProjectionModelType::PINHOLE:
      return "pinhole";
    case ProjectionModelType::FISHEYE_THETA:
      return "fisheye_theta";
    case ProjectionModelType::OMNIDIRECTIONAL:
      return "omnidirectional";
    case ProjectionModelType::DOUBLE_SPHERE:
      return "double_sphere";
    case ProjectionModelType::EUCM:
      return "eucm";
    case ProjectionModelType::UNKNOWN:
      break;
  }
  return "unknown";
}

const char *toString(const DistortionSpace space)
{
  switch (space)
  {
    case DistortionSpace::NONE:
      return "none";
    case DistortionSpace::PLANE:
      return "plane";
    case DistortionSpace::ANGLE:
      return "angle";
    default:
      return "unknown";
  }
}

}  // namespace camxiom
