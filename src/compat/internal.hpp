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

#ifndef CAMXIOM__COMPAT__INTERNAL_HPP
#define CAMXIOM__COMPAT__INTERNAL_HPP

#include "camxiom/compat.hpp"
#include "camxiom/internal/angle_law.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace camxiom
{
namespace
{

inline bool isFiniteScalar(const double value) { return std::isfinite(value); }

inline bool isFiniteArray9(const std::array<double, 9> &array_9)
{
  for (const double value : array_9)
  {
    if (!isFiniteScalar(value))
    {
      return false;
    }
  }
  return true;
}

inline bool isFiniteVector(const std::vector<double> &values)
{
  for (const double value : values)
  {
    if (!isFiniteScalar(value))
    {
      return false;
    }
  }
  return true;
}

inline int expectedDistortionCount(const PinholeCompatProfile profile)
{
  switch (profile)
  {
    case PinholeCompatProfile::CANONICAL:
      return -1;
    case PinholeCompatProfile::OPENCV_CALIB3D_D4:
      return 4;
    case PinholeCompatProfile::OPENCV_CALIB3D_D5:
      return 5;
    case PinholeCompatProfile::OPENCV_CALIB3D_D8:
      return 8;
    case PinholeCompatProfile::OPENCV_CALIB3D_D12:
      return 12;
    case PinholeCompatProfile::OPENCV_CALIB3D_D14:
      return 14;
    case PinholeCompatProfile::ROS_CAMERA_INFO_RAW:
      return -1;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const FisheyeCompatProfile profile)
{
  switch (profile)
  {
    case FisheyeCompatProfile::CANONICAL:
      return -1;
    case FisheyeCompatProfile::OPENCV_FISHEYE_D4:
      return 4;
    case FisheyeCompatProfile::ROS_CAMERA_INFO_EQUIDISTANT:
      return -1;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const OmnidirectionalCompatProfile profile)
{
  switch (profile)
  {
    case OmnidirectionalCompatProfile::CANONICAL:
      return -1;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D4:
      return 4;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D5:
      return 5;
    case OmnidirectionalCompatProfile::OPENCV_OMNIDIR_D8:
      return 8;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const DoubleSphereCompatProfile profile)
{
  switch (profile)
  {
    case DoubleSphereCompatProfile::CANONICAL:
      return -1;
    case DoubleSphereCompatProfile::BASALT_D0:
      return 0;
    case DoubleSphereCompatProfile::BASALT_D4:
      return 4;
    default:
      return -2;
  }
}

inline int expectedDistortionCount(const EucmCompatProfile profile)
{
  switch (profile)
  {
    case EucmCompatProfile::CANONICAL:
      return -1;
    case EucmCompatProfile::KALIBR_D0:
      return 0;
    case EucmCompatProfile::KALIBR_D4:
      return 4;
    default:
      return -2;
  }
}

/// Binning and ROI of a ROS CameraInfo. sensor_msgs/CameraInfo defines K and
/// P in FULL-RESOLUTION coordinates, while the image that accompanies it is
/// the ROI crop decimated by the binning factors. Per the message spec,
/// binning 0 means 1 and an ROI with width/height 0 means the full image.
struct BinningAndRoi
{
  std::uint32_t binning_x{0U};
  std::uint32_t binning_y{0U};
  std::uint32_t x_offset{0U};
  std::uint32_t y_offset{0U};
  std::uint32_t width{0U};
  std::uint32_t height{0U};
};

/// Fold binning and ROI into K, P and the image size — the same adjustment
/// image_geometry::PinholeCameraModel applies — so the model read from the
/// result maps the published image's pixels. D is in normalised coordinates
/// and needs no adjustment.
inline void foldBinningAndRoi(const BinningAndRoi &crop, CameraInfo &info)
{
  const double bx = (crop.binning_x > 1U) ? static_cast<double>(crop.binning_x) : 1.0;
  const double by = (crop.binning_y > 1U) ? static_cast<double>(crop.binning_y) : 1.0;
  const bool cropped = (crop.x_offset != 0U || crop.y_offset != 0U) ||
                       (crop.width != 0U && crop.width != info.width) ||
                       (crop.height != 0U && crop.height != info.height);
  if (bx == 1.0 && by == 1.0 && !cropped)
  {
    return;
  }
  const double ox = static_cast<double>(crop.x_offset);
  const double oy = static_cast<double>(crop.y_offset);
  info.k[0] /= bx;                    // fx
  info.k[1] /= bx;                    // skew
  info.k[2] = (info.k[2] - ox) / bx;  // cx
  info.k[4] /= by;                    // fy
  info.k[5] = (info.k[5] - oy) / by;  // cy
  info.p[0] /= bx;                    // fx'
  info.p[1] /= bx;                    // skew'
  info.p[2] = (info.p[2] - ox) / bx;  // cx'
  info.p[3] /= bx;                    // Tx (= -fx' * baseline)
  info.p[5] /= by;                    // fy'
  info.p[6] = (info.p[6] - oy) / by;  // cy'
  info.p[7] /= by;                    // Ty
  const std::uint32_t roi_w = (crop.width != 0U) ? crop.width : info.width;
  const std::uint32_t roi_h = (crop.height != 0U) ? crop.height : info.height;
  info.width = static_cast<std::uint32_t>(static_cast<double>(roi_w) / bx);
  info.height = static_cast<std::uint32_t>(static_cast<double>(roi_h) / by);
}

template <typename T>
inline void fillProjectionFromIntrinsics(
  const IntrinsicsModelT<T> &intrinsics, std::array<double, 9> &camera_matrix
)
{
  camera_matrix = {
    static_cast<double>(intrinsics.fx),
    static_cast<double>(intrinsics.skew),
    static_cast<double>(intrinsics.cx),
    0.0,
    static_cast<double>(intrinsics.fy),
    static_cast<double>(intrinsics.cy),
    0.0,
    0.0,
    1.0};
}

template <typename T>
inline void fillProjectionMatrix3x4(
  const IntrinsicsModelT<T> &intrinsics, std::array<double, 12> &projection
)
{
  projection = {
    static_cast<double>(intrinsics.fx),
    static_cast<double>(intrinsics.skew),
    static_cast<double>(intrinsics.cx),
    0.0,
    0.0,
    static_cast<double>(intrinsics.fy),
    static_cast<double>(intrinsics.cy),
    0.0,
    0.0,
    0.0,
    1.0,
    0.0};
}

inline std::string pinholeDistortionModelByCount(const std::size_t count)
{
  return count >= 8U ? "rational_polynomial" : "plumb_bob";
}

inline std::string pinholeDistortionModelByType(const DistortionModelType model_type)
{
  switch (model_type)
  {
    case DistortionModelType::RADTAN4:
    case DistortionModelType::RADTAN5:
      return "plumb_bob";
    case DistortionModelType::RATIONAL8:
      return "rational_polynomial";
    case DistortionModelType::THIN_PRISM12:
      return "thin_prism_fisheye";
    case DistortionModelType::TILTED14:
      return "tilted";
    case DistortionModelType::NONE:
      return "plumb_bob";
    case DistortionModelType::KB4:
    case DistortionModelType::KB8:
    case DistortionModelType::UNKNOWN:
      break;
  }
  return "plumb_bob";
}

inline constexpr std::size_t kFisheyeCanonicalMaxD = 8U;
inline constexpr std::size_t kFisheyeLegacyMaxD = 4U;

// The interchange distortion_model name of a fisheye law and its correction,
// or an empty string for a combination camxiom does not support. ROS names
// the KB4 polynomial "equidistant" and has no name for the plain equidistant
// law; a zero KB4 is that law, so it is written as "equidistant" too.
inline std::string fisheyeDistortionModelString(
  const ProjectionModelType projection_type, const DistortionModelType distortion_type
)
{
  switch (detail_impl::angleLawOf(projection_type, distortion_type))
  {
    case detail_impl::AngleLaw::EQUIDISTANT:
    case detail_impl::AngleLaw::KB4:
      return "equidistant";
    case detail_impl::AngleLaw::KB8:
      return "kb8";
    case detail_impl::AngleLaw::EQUISOLID:
      return "equisolid";
    case detail_impl::AngleLaw::STEREOGRAPHIC:
      return "stereographic";
    case detail_impl::AngleLaw::ORTHOGRAPHIC:
      return "orthographic";
    case detail_impl::AngleLaw::INVALID:
      break;
  }
  return "";
}

// camxiom's own name for a coefficient-free fisheye law. Unlike the ROS
// name above it reads back as the same law with no correction.
inline std::string fisheyeLawName(const ProjectionModelType projection_type)
{
  switch (projection_type)
  {
    case ProjectionModelType::EQUIDISTANT:
      return "ideal_equidistant";
    case ProjectionModelType::EQUISOLID:
      return "equisolid";
    case ProjectionModelType::STEREOGRAPHIC:
      return "stereographic";
    case ProjectionModelType::ORTHOGRAPHIC:
      return "orthographic";
    default:
      break;
  }
  return "";
}

template <typename T>
StatusCode validateCompatModel(const CameraModelT<T> &model)
{
  if constexpr (std::is_same_v<T, float>) return validateCameraModel(model);
  else return validateCameraModel64(model);
}

template <typename T, typename AcceptProjection>
inline StatusCode makeAndValidateModel(
  const CameraInfo &camera_info, const AcceptProjection &accept_projection,
  CameraModelT<T> &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  if constexpr (std::is_same_v<T, float>) model_out = makeCameraModel(camera_info);
  else model_out = makeCameraModel64(camera_info);
  const StatusCode validation_status = validateCompatModel(model_out);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (!accept_projection(model_out.projection.type))
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode makeAndValidateProjectionModel(
  const CameraInfo &camera_info, const ProjectionModelType projection_type,
  CameraModelT<T> &model_out
)
{
  return makeAndValidateModel(
    camera_info,
    [projection_type](const ProjectionModelType type) { return type == projection_type; }, model_out
  );
}

// Any fisheye law: the CameraInfo name picks which one.
template <typename T>
inline StatusCode makeAndValidateFisheyeModel(
  const CameraInfo &camera_info, CameraModelT<T> &model_out
)
{
  return makeAndValidateModel(
    camera_info, [](const ProjectionModelType type) { return isFisheyeProjection(type); }, model_out
  );
}

template <typename T>
inline StatusCode validatePinholeModelForExport(const CameraModelT<T> &model)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::PINHOLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (distortionSpaceOf(model.distortion.type) == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode validateFisheyeModelForExport(const CameraModelT<T> &model)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (!isFisheyeProjection(model.projection.type))
  {
    return StatusCode::INVALID_MODEL;
  }
  if (distortionSpaceOf(model.distortion.type) == DistortionSpace::PLANE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode validateOmniModelForExport(const CameraModelT<T> &model)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::OMNIDIRECTIONAL)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (distortionSpaceOf(model.distortion.type) == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode validateDoubleSphereModelForExport(const CameraModelT<T> &model)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::DOUBLE_SPHERE)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (distortionSpaceOf(model.distortion.type) == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

template <typename T>
inline StatusCode validateEucmModelForExport(const CameraModelT<T> &model)
{
  const StatusCode validation_status = validateCompatModel(model);
  if (validation_status != StatusCode::OK)
  {
    return validation_status;
  }
  if (model.projection.type != ProjectionModelType::EUCM)
  {
    return StatusCode::INVALID_MODEL;
  }
  if (distortionSpaceOf(model.distortion.type) == DistortionSpace::ANGLE)
  {
    return StatusCode::INVALID_MODEL;
  }
  return StatusCode::OK;
}

}  // namespace
}  // namespace camxiom

#endif  // CAMXIOM__COMPAT__INTERNAL_HPP
