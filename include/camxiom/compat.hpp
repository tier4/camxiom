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

#ifndef CAMXIOM__COMPAT_HPP
#define CAMXIOM__COMPAT_HPP

#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace camxiom
{

/// ROS-free, OpenCV-free description of a raw camera (the canonical interchange
/// type used by every compat conversion). It mirrors the field layout of a ROS
/// sensor_msgs/CameraInfo message but carries no ROS dependency, so the heavy
/// K/D/distortion-model parsing in makeCameraModel() lives entirely in the
/// dependency-free core. The optional ROS layer (camxiom/ros.hpp) and OpenCV
/// layer (camxiom/opencv.hpp) only do trivial field copies into this struct.
struct CameraInfo
{
  std::array<double, 9> k{
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};  // intrinsic camera matrix (row-major)
  std::vector<double> d{};                           // distortion coefficients
  std::string distortion_model{};                    // distortion model string
  std::array<double, 9> r{
    {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};  // rectification matrix (row-major)
  std::array<double, 12> p{
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};  // projection matrix (row-major)
  std::uint32_t width{0U};
  std::uint32_t height{0U};
};

/// Build a CameraModel from a raw camera description (canonical mapping; the
/// distortion-model string and coefficient packing follow the camxiom
/// convention shared with the import*CameraInfo profiles below). This is the
/// single place the K/D/distortion-model parsing lives; ROS and OpenCV adapters
/// funnel through it.
[[nodiscard]] CameraModel makeCameraModel(const CameraInfo &camera_info);

enum class PinholeCompatProfile : std::uint8_t {
  CANONICAL = 0U,
  OPENCV_CALIB3D_D4,
  OPENCV_CALIB3D_D5,
  OPENCV_CALIB3D_D8,
  OPENCV_CALIB3D_D12,
  OPENCV_CALIB3D_D14,
  ROS_CAMERA_INFO_RAW
};

enum class FisheyeCompatProfile : std::uint8_t {
  CANONICAL = 0U,
  OPENCV_FISHEYE_D4,
  ROS_CAMERA_INFO_EQUIDISTANT
};

enum class OmnidirectionalCompatProfile : std::uint8_t {
  CANONICAL = 0U,
  OPENCV_OMNIDIR_D4,
  OPENCV_OMNIDIR_D5,
  OPENCV_OMNIDIR_D8
};

enum class DoubleSphereCompatProfile : std::uint8_t { CANONICAL = 0U, BASALT_D0, BASALT_D4 };

enum class EucmCompatProfile : std::uint8_t { CANONICAL = 0U, KALIBR_D0, KALIBR_D4 };

struct PinholeExternalModel
{
  std::array<double, 9> K{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  std::vector<double> D{};
};

struct FisheyeExternalModel
{
  std::array<double, 9> K{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  std::vector<double> D{};
  DistortionModelType distortion_type{DistortionModelType::UNKNOWN};
};

struct OmnidirectionalExternalModel
{
  std::array<double, 9> K{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  std::vector<double> D{};
  double xi{0.0};
};

struct DoubleSphereExternalModel
{
  std::array<double, 9> K{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  std::vector<double> D{};
  double xi{0.0};
  double alpha{0.0};
};

struct EucmExternalModel
{
  std::array<double, 9> K{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
  std::vector<double> D{};
  double alpha{0.0};
  double beta{1.0};
};

[[nodiscard]] StatusCode importPinholeModel(
  const PinholeExternalModel &external_model, PinholeCompatProfile profile, CameraModel &model_out
);

[[nodiscard]] StatusCode exportPinholeModel(
  const CameraModel &model, PinholeCompatProfile profile, PinholeExternalModel &external_model_out
);

[[nodiscard]] StatusCode importFisheyeModel(
  const FisheyeExternalModel &external_model, FisheyeCompatProfile profile, CameraModel &model_out
);

[[nodiscard]] StatusCode exportFisheyeModel(
  const CameraModel &model, FisheyeCompatProfile profile, FisheyeExternalModel &external_model_out
);

[[nodiscard]] StatusCode importOmnidirectionalModel(
  const OmnidirectionalExternalModel &external_model, OmnidirectionalCompatProfile profile,
  CameraModel &model_out
);

[[nodiscard]] StatusCode exportOmnidirectionalModel(
  const CameraModel &model, OmnidirectionalCompatProfile profile,
  OmnidirectionalExternalModel &external_model_out
);

[[nodiscard]] StatusCode importDoubleSphereModel(
  const DoubleSphereExternalModel &external_model, DoubleSphereCompatProfile profile,
  CameraModel &model_out
);

[[nodiscard]] StatusCode exportDoubleSphereModel(
  const CameraModel &model, DoubleSphereCompatProfile profile,
  DoubleSphereExternalModel &external_model_out
);

[[nodiscard]] StatusCode importEucmModel(
  const EucmExternalModel &external_model, EucmCompatProfile profile, CameraModel &model_out
);

[[nodiscard]] StatusCode exportEucmModel(
  const CameraModel &model, EucmCompatProfile profile, EucmExternalModel &external_model_out
);

/// Export a calibrated model into the ROS-free CameraInfo POD: K/D packed per
/// the profile, distortion_model set to the family's canonical string, r set to
/// identity and p filled as [K | 0]. width/height are left untouched (the
/// caller owns the image geometry). This is the single source of the CameraInfo
/// packing; the optional ROS layer (camxiom/ros.hpp) delegates here and only
/// copies fields into sensor_msgs.
[[nodiscard]] StatusCode exportPinholeCameraInfo(
  const CameraModel &model, PinholeCompatProfile profile, CameraInfo &camera_info_out
);

[[nodiscard]] StatusCode exportFisheyeCameraInfo(
  const CameraModel &model, FisheyeCompatProfile profile, CameraInfo &camera_info_out
);

[[nodiscard]] StatusCode exportOmnidirectionalCameraInfo(
  const CameraModel &model, OmnidirectionalCompatProfile profile, CameraInfo &camera_info_out
);

[[nodiscard]] StatusCode exportDoubleSphereCameraInfo(
  const CameraModel &model, DoubleSphereCompatProfile profile, CameraInfo &camera_info_out
);

[[nodiscard]] StatusCode exportEucmCameraInfo(
  const CameraModel &model, EucmCompatProfile profile, CameraInfo &camera_info_out
);

const char *toString(PinholeCompatProfile profile);
const char *toString(FisheyeCompatProfile profile);
const char *toString(OmnidirectionalCompatProfile profile);
const char *toString(DoubleSphereCompatProfile profile);
const char *toString(EucmCompatProfile profile);

}  // namespace camxiom

#endif  // CAMXIOM__COMPAT_HPP
