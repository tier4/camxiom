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

  // Fold binning/ROI into K / P (the image_geometry adjustment), so the
  // resulting model maps the published image's pixel coordinates instead of
  // being silently off by the binning factor / ROI offset.
  BinningAndRoi crop;
  crop.binning_x = camera_info.binning_x;
  crop.binning_y = camera_info.binning_y;
  crop.x_offset = camera_info.roi.x_offset;
  crop.y_offset = camera_info.roi.y_offset;
  crop.width = camera_info.roi.width;
  crop.height = camera_info.roi.height;
  foldBinningAndRoi(crop, info);
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

StatusCode exportCameraInfo(const CameraModel &model, sensor_msgs::msg::CameraInfo &camera_info_out)
{
  CameraInfo info;
  const StatusCode status = exportCameraInfo(model, info);
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
