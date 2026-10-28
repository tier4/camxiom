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

// CameraModel <-> CamxiomCameraInfo: camxiom's own description, which holds
// every valid model field by field.

#include "camxiom/compat.hpp"
#include "camxiom/model.hpp"
#include "compat/internal.hpp"
#include "model/internal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>

namespace camxiom
{
namespace
{

// The law's own parameters, in the order CamxiomCameraInfo lists them.
std::size_t projectionParameterCount(const ProjectionModelType type)
{
  switch (type)
  {
    case ProjectionModelType::OMNIDIRECTIONAL:
      return 1U;
    case ProjectionModelType::DOUBLE_SPHERE:
    case ProjectionModelType::EUCM:
      return 2U;
    default:
      return 0U;
  }
}

// The number of coefficients each correction stores.
std::size_t correctionCoefficientCount(const DistortionModelType type)
{
  switch (type)
  {
    case DistortionModelType::NONE:
      return 0U;
    case DistortionModelType::RADTAN4:
    case DistortionModelType::KB4:
      return 4U;
    case DistortionModelType::RADTAN5:
      return 5U;
    case DistortionModelType::RATIONAL8:
    case DistortionModelType::KB8:
      return 8U;
    case DistortionModelType::THIN_PRISM12:
      return 12U;
    case DistortionModelType::TILTED14:
      return 14U;
    case DistortionModelType::UNKNOWN:
      break;
  }
  return 0U;
}

// The shortest decimal that reads back as the same float, so a float model is
// written as 0.12 rather than as the double nearest to the float 0.12.
double shortestDecimal(const float value)
{
  if (!std::isfinite(value))
  {
    return static_cast<double>(value);
  }
  for (int digits = 1; digits <= std::numeric_limits<float>::max_digits10; ++digits)
  {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.precision(digits);
    stream << static_cast<double>(value);
    std::istringstream reader(stream.str());
    reader.imbue(std::locale::classic());
    double parsed = 0.0;
    reader >> parsed;
    if (static_cast<float>(parsed) == value)
    {
      return parsed;
    }
  }
  return static_cast<double>(value);
}

template <typename T>
double exported(const T value)
{
  if constexpr (std::is_same_v<T, float>)
    return shortestDecimal(value);
  else
    return value;
}

template <typename T>
CameraModelT<T> makeCamxiomModelImpl(const CamxiomCameraInfo &info)
{
  CameraModelT<T> model;
  model.intrinsics.fx = static_cast<T>(info.intrinsic_parameters[0]);
  model.intrinsics.fy = static_cast<T>(info.intrinsic_parameters[1]);
  model.intrinsics.cx = static_cast<T>(info.intrinsic_parameters[2]);
  model.intrinsics.cy = static_cast<T>(info.intrinsic_parameters[3]);
  model.intrinsics.skew = static_cast<T>(info.intrinsic_parameters[4]);

  const ProjectionModelType law = parseProjectionModelType(info.projection_model);
  const DistortionModelType correction = parseDistortionModelType(info.distortion_model);
  if (law == ProjectionModelType::UNKNOWN || correction == DistortionModelType::UNKNOWN ||
      info.projection_parameters.size() != projectionParameterCount(law) ||
      info.distortion_parameters.size() != correctionCoefficientCount(correction))
  {
    // projection.type stays UNKNOWN, which validation rejects.
    return model;
  }

  switch (law)
  {
    case ProjectionModelType::OMNIDIRECTIONAL:
      model.projection.xi = static_cast<T>(info.projection_parameters[0]);
      break;
    case ProjectionModelType::DOUBLE_SPHERE:
      model.projection.xi = static_cast<T>(info.projection_parameters[0]);
      model.projection.alpha = static_cast<T>(info.projection_parameters[1]);
      break;
    case ProjectionModelType::EUCM:
      model.projection.alpha = static_cast<T>(info.projection_parameters[0]);
      model.projection.beta = static_cast<T>(info.projection_parameters[1]);
      break;
    default:
      break;
  }
  model.projection.type = law;

  model.distortion.type = correction;
  model.distortion.count = static_cast<std::uint8_t>(info.distortion_parameters.size());
  for (std::size_t index = 0; index < info.distortion_parameters.size(); ++index)
  {
    model.distortion.coeffs[index] = static_cast<T>(info.distortion_parameters[index]);
  }
  detail::rebuildDistortionAuxState(model.distortion);

  if constexpr (std::is_same_v<T, float>)
    updateThetaMax(model);
  else
    updateThetaMax64(model);
  return model;
}

template <typename T>
StatusCode exportCamxiomCameraInfoImpl(const CameraModelT<T> &model, CamxiomCameraInfo &out)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }

  out.projection_model = toString(model.projection.type);
  switch (model.projection.type)
  {
    case ProjectionModelType::OMNIDIRECTIONAL:
      out.projection_parameters = {exported(model.projection.xi)};
      break;
    case ProjectionModelType::DOUBLE_SPHERE:
      out.projection_parameters = {exported(model.projection.xi), exported(model.projection.alpha)};
      break;
    case ProjectionModelType::EUCM:
      out.projection_parameters = {
        exported(model.projection.alpha), exported(model.projection.beta)};
      break;
    default:
      out.projection_parameters.clear();
      break;
  }

  // A RADTAN4 model may carry k3 in a fifth coefficient, which projection
  // evaluates: that lens is radtan5.
  const DistortionModelType correction =
    model.distortion.type == DistortionModelType::RADTAN4 && model.distortion.count == 5U
      ? DistortionModelType::RADTAN5
      : model.distortion.type;
  out.distortion_model = toString(correction);
  const std::size_t count = correctionCoefficientCount(correction);
  out.distortion_parameters.assign(count, 0.0);
  for (std::size_t index = 0; index < count; ++index)
  {
    out.distortion_parameters[index] = exported(model.distortion.coeffs[index]);
  }

  out.intrinsic_parameters = {
    exported(model.intrinsics.fx), exported(model.intrinsics.fy), exported(model.intrinsics.cx),
    exported(model.intrinsics.cy), exported(model.intrinsics.skew)};
  return StatusCode::OK;
}

}  // namespace

CameraModel makeCameraModel(const CamxiomCameraInfo &camera_info)
{
  return makeCamxiomModelImpl<float>(camera_info);
}

CameraModel64 makeCameraModel64(const CamxiomCameraInfo &camera_info)
{
  return makeCamxiomModelImpl<double>(camera_info);
}

StatusCode exportCameraInfo(const CameraModel &model, CamxiomCameraInfo &camera_info_out)
{
  return exportCamxiomCameraInfoImpl(model, camera_info_out);
}

StatusCode exportCameraInfo64(const CameraModel64 &model, CamxiomCameraInfo &camera_info_out)
{
  return exportCamxiomCameraInfoImpl(model, camera_info_out);
}

}  // namespace camxiom
