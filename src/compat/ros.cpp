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

#include "camxiom/ros.hpp"

#if __has_include(<sensor_msgs/msg/camera_info.hpp>)

#include "compat/internal.hpp"

namespace camxiom
{
namespace
{

CameraInfo toCameraInfo(const sensor_msgs::msg::CameraInfo &camera_info)
{
  CameraInfo info;
  info.k = camera_info.k;
  info.d = camera_info.d;
  info.distortion_model = camera_info.distortion_model;
  info.r = camera_info.r;
  info.p = camera_info.p;
  info.width = camera_info.width;
  info.height = camera_info.height;

  // sensor_msgs/CameraInfo defines K and P in FULL-RESOLUTION coordinates,
  // while the image accompanying the message is the ROI crop decimated by
  // the binning factors. Fold both into K / P — the same adjustment
  // image_geometry::PinholeCameraModel applies — so the resulting model maps
  // the published image's pixel coordinates instead of being silently off by
  // the binning factor / ROI offset. Per the message spec, binning 0 means 1
  // and an ROI with width/height 0 means the full image. (D is expressed in
  // normalised coordinates and needs no adjustment.)
  const double bx = (camera_info.binning_x > 1U) ? static_cast<double>(camera_info.binning_x) : 1.0;
  const double by = (camera_info.binning_y > 1U) ? static_cast<double>(camera_info.binning_y) : 1.0;
  const auto &roi = camera_info.roi;
  const bool cropped = (roi.x_offset != 0U || roi.y_offset != 0U) ||
                       (roi.width != 0U && roi.width != camera_info.width) ||
                       (roi.height != 0U && roi.height != camera_info.height);
  if (bx != 1.0 || by != 1.0 || cropped)
  {
    const double ox = static_cast<double>(roi.x_offset);
    const double oy = static_cast<double>(roi.y_offset);
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
    const std::uint32_t roi_w = (roi.width != 0U) ? roi.width : camera_info.width;
    const std::uint32_t roi_h = (roi.height != 0U) ? roi.height : camera_info.height;
    info.width = static_cast<std::uint32_t>(static_cast<double>(roi_w) / bx);
    info.height = static_cast<std::uint32_t>(static_cast<double>(roi_h) / by);
  }
  return info;
}

// The exported K/P describe the model's own (already binned/cropped) pixel
// frame — toCameraInfo() above folds binning/ROI INTO K/P on import. Stale
// binning/ROI metadata left in the outgoing message would make the next
// importer apply that adjustment a SECOND time (silently halving a binning=2
// camera's focal length on an in-place republish), so every export resets
// them to the message spec's "no binning / full image" defaults.
// width/height stay untouched: the model does not know the image size — the
// caller owns those fields.
void resetBinningAndRoi(sensor_msgs::msg::CameraInfo &camera_info_out)
{
  camera_info_out.binning_x = 0U;
  camera_info_out.binning_y = 0U;
  camera_info_out.roi = sensor_msgs::msg::RegionOfInterest{};
}

/// Copy the fields the core POD export populates (k/d/r/p/distortion_model)
/// into the ROS message. width/height stay untouched — the caller owns those.
void copyExportedFields(const CameraInfo &info, sensor_msgs::msg::CameraInfo &camera_info_out)
{
  camera_info_out.k = info.k;
  camera_info_out.d = info.d;
  camera_info_out.r = info.r;
  camera_info_out.p = info.p;
  camera_info_out.distortion_model = info.distortion_model;
}

}  // namespace

CameraModel makeCameraModel(const sensor_msgs::msg::CameraInfo &camera_info)
{
  return makeCameraModel(toCameraInfo(camera_info));
}

// ---------------------------------------------------------------------------
// Pinhole
// ---------------------------------------------------------------------------

StatusCode importPinholeCameraInfo(
  const sensor_msgs::msg::CameraInfo &camera_info, const PinholeCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }

  if (profile == PinholeCompatProfile::CANONICAL || profile == PinholeCompatProfile::ROS_CAMERA_INFO_RAW)
  {
    const CameraInfo info = toCameraInfo(camera_info);
    return makeAndValidateProjectionModel(info, ProjectionModelType::PINHOLE, model_out);
  }

  if (camera_info.d.size() != static_cast<std::size_t>(expected_count))
  {
    return StatusCode::INVALID_INPUT;
  }

  PinholeExternalModel external_model;
  // K must go through the binning/ROI adjustment in toCameraInfo.
  external_model.K = toCameraInfo(camera_info).k;
  external_model.D = camera_info.d;
  return importPinholeModel(external_model, profile, model_out);
}

StatusCode exportPinholeCameraInfo(
  const CameraModel &model, const PinholeCompatProfile profile,
  sensor_msgs::msg::CameraInfo &camera_info_out
)
{
  CameraInfo info;
  const StatusCode status = exportPinholeCameraInfo(model, profile, info);
  if (status != StatusCode::OK)
  {
    return status;
  }
  resetBinningAndRoi(camera_info_out);
  copyExportedFields(info, camera_info_out);
  return StatusCode::OK;
}

// ---------------------------------------------------------------------------
// Fisheye
// ---------------------------------------------------------------------------

StatusCode importFisheyeCameraInfo(
  const sensor_msgs::msg::CameraInfo &camera_info, const FisheyeCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }

  if (profile == FisheyeCompatProfile::CANONICAL || profile == FisheyeCompatProfile::ROS_CAMERA_INFO_EQUIDISTANT)
  {
    const CameraInfo info = toCameraInfo(camera_info);
    return makeAndValidateProjectionModel(info, ProjectionModelType::FISHEYE_THETA, model_out);
  }

  if (camera_info.d.size() != static_cast<std::size_t>(expected_count))
  {
    return StatusCode::INVALID_INPUT;
  }

  FisheyeExternalModel external_model;
  // K must go through the binning/ROI adjustment in toCameraInfo.
  external_model.K = toCameraInfo(camera_info).k;
  external_model.D = camera_info.d;
  return importFisheyeModel(external_model, profile, model_out);
}

StatusCode exportFisheyeCameraInfo(
  const CameraModel &model, const FisheyeCompatProfile profile,
  sensor_msgs::msg::CameraInfo &camera_info_out
)
{
  CameraInfo info;
  const StatusCode status = exportFisheyeCameraInfo(model, profile, info);
  if (status != StatusCode::OK)
  {
    return status;
  }
  resetBinningAndRoi(camera_info_out);
  copyExportedFields(info, camera_info_out);
  return StatusCode::OK;
}

// ---------------------------------------------------------------------------
// Omnidirectional
// ---------------------------------------------------------------------------

StatusCode importOmnidirectionalCameraInfo(
  const sensor_msgs::msg::CameraInfo &camera_info, const OmnidirectionalCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  if (profile == OmnidirectionalCompatProfile::CANONICAL)
  {
    const CameraInfo info = toCameraInfo(camera_info);
    return makeAndValidateProjectionModel(info, ProjectionModelType::OMNIDIRECTIONAL, model_out);
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count <= 0)
  {
    return StatusCode::INVALID_INPUT;
  }

  OmnidirectionalExternalModel external_model;
  // K must go through the binning/ROI adjustment in toCameraInfo.
  external_model.K = toCameraInfo(camera_info).k;

  if (camera_info.d.size() == static_cast<std::size_t>(expected_count + 1))
  {
    external_model.xi = camera_info.d[0];
    external_model.D.assign(camera_info.d.begin() + 1, camera_info.d.end());
  }
  else
  {
    return StatusCode::INVALID_INPUT;
  }

  return importOmnidirectionalModel(external_model, profile, model_out);
}

StatusCode exportOmnidirectionalCameraInfo(
  const CameraModel &model, const OmnidirectionalCompatProfile profile,
  sensor_msgs::msg::CameraInfo &camera_info_out
)
{
  CameraInfo info;
  const StatusCode status = exportOmnidirectionalCameraInfo(model, profile, info);
  if (status != StatusCode::OK)
  {
    return status;
  }
  resetBinningAndRoi(camera_info_out);
  copyExportedFields(info, camera_info_out);
  return StatusCode::OK;
}

// ---------------------------------------------------------------------------
// Double sphere
// ---------------------------------------------------------------------------

StatusCode importDoubleSphereCameraInfo(
  const sensor_msgs::msg::CameraInfo &camera_info, const DoubleSphereCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }

  if (camera_info.d.size() < 2U)
  {
    return StatusCode::INVALID_INPUT;
  }

  DoubleSphereExternalModel external_model;
  // K must go through the binning/ROI adjustment in toCameraInfo.
  external_model.K = toCameraInfo(camera_info).k;
  external_model.xi = camera_info.d[0];
  external_model.alpha = camera_info.d[1];
  external_model.D.assign(camera_info.d.begin() + 2, camera_info.d.end());

  if (expected_count >= 0 && external_model.D.size() != static_cast<std::size_t>(expected_count))
  {
    return StatusCode::INVALID_INPUT;
  }

  return importDoubleSphereModel(external_model, profile, model_out);
}

StatusCode exportDoubleSphereCameraInfo(
  const CameraModel &model, const DoubleSphereCompatProfile profile,
  sensor_msgs::msg::CameraInfo &camera_info_out
)
{
  CameraInfo info;
  const StatusCode status = exportDoubleSphereCameraInfo(model, profile, info);
  if (status != StatusCode::OK)
  {
    return status;
  }
  resetBinningAndRoi(camera_info_out);
  copyExportedFields(info, camera_info_out);
  return StatusCode::OK;
}

// ---------------------------------------------------------------------------
// EUCM
// ---------------------------------------------------------------------------

StatusCode importEucmCameraInfo(
  const sensor_msgs::msg::CameraInfo &camera_info, const EucmCompatProfile profile,
  CameraModel &model_out
)
{
  if (!isFiniteArray9(camera_info.k) || !isFiniteVector(camera_info.d))
  {
    return StatusCode::INVALID_INPUT;
  }

  const int expected_count = expectedDistortionCount(profile);
  if (expected_count == -2)
  {
    return StatusCode::INVALID_INPUT;
  }

  if (camera_info.d.size() < 2U)
  {
    return StatusCode::INVALID_INPUT;
  }

  EucmExternalModel external_model;
  // K must go through the binning/ROI adjustment in toCameraInfo.
  external_model.K = toCameraInfo(camera_info).k;
  external_model.alpha = camera_info.d[0];
  external_model.beta = camera_info.d[1];
  external_model.D.assign(camera_info.d.begin() + 2, camera_info.d.end());

  if (expected_count >= 0 && external_model.D.size() != static_cast<std::size_t>(expected_count))
  {
    return StatusCode::INVALID_INPUT;
  }

  return importEucmModel(external_model, profile, model_out);
}

StatusCode exportEucmCameraInfo(
  const CameraModel &model, const EucmCompatProfile profile,
  sensor_msgs::msg::CameraInfo &camera_info_out
)
{
  CameraInfo info;
  const StatusCode status = exportEucmCameraInfo(model, profile, info);
  if (status != StatusCode::OK)
  {
    return status;
  }
  resetBinningAndRoi(camera_info_out);
  copyExportedFields(info, camera_info_out);
  return StatusCode::OK;
}

}  // namespace camxiom

#endif  // __has_include(<sensor_msgs/msg/camera_info.hpp>)
