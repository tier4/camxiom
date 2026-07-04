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

#ifndef CAMXIOM__CORE_HPP
#define CAMXIOM__CORE_HPP

// Convenience umbrella for the RUNTIME geometry layer (camxiom::core).
//
// Pulls in every public header of the Eigen-only runtime core: types, camera
// model, projection (float + double + AutoDiff template), Jacobians, inverse
// LUT, batch projection, remap, and the ROS/OpenCV-free CameraInfo compat POD.
//
// This header stays dependency-light on purpose: it does NOT pull in Ceres
// (calibration layer) nor the optional ROS / OpenCV interop declarations, so a
// translation unit that only needs ray<->pixel geometry pays for nothing else.
//   * For the calibration layer (seed / init estimators / PnP / intrinsics)
//     include <camxiom/calibration.hpp>.
//   * For the optional interop layers include <camxiom/ros.hpp> /
//     <camxiom/opencv.hpp>, or the all-in-one <camxiom/camxiom.hpp>.
//
// Individual headers remain available; this is purely a convenience aggregate
// and changes no existing behaviour.

// --- Library version macros --------------------------------------------------
#include "camxiom/version.hpp"

// --- Core data types --------------------------------------------------------
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

// --- Camera model: parse / validate / factory helpers -----------------------
#include "camxiom/model.hpp"
#include "camxiom/model_compare.hpp"

// --- Validated-once immutable camera model (valid by construction) -----------
#include "camxiom/validated_model.hpp"

// --- ROS-free, OpenCV-free K/D <-> CameraModel interchange (CameraInfo POD) --
#include "camxiom/camera_info_yaml.hpp"
#include "camxiom/compat.hpp"

// --- Projection: ray <-> pixel (float, double, AutoDiff template) -----------
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/projection_template.hpp"

// --- Analytical Jacobians ---------------------------------------------------
#include "camxiom/jacobian.hpp"
#include "camxiom/jacobian64.hpp"
#include "camxiom/jacobian_batch.hpp"
#include "camxiom/jacobian_batch64.hpp"
#include "camxiom/jacobian_with_distortion_deriv64.hpp"

// --- Inverse LUT (O(1) approximate pixelToRay) ------------------------------
#include "camxiom/lut.hpp"
#include "camxiom/lut64.hpp"

// --- Batch projection -------------------------------------------------------
#include "camxiom/batch.hpp"
#include "camxiom/batch64.hpp"

// --- Image-side remap / rectify maps ----------------------------------------
#include "camxiom/remap.hpp"
#include "camxiom/remap_kernel.hpp"

#endif  // CAMXIOM__CORE_HPP
