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

#ifndef CAMXIOM__CAMXIOM_HPP
#define CAMXIOM__CAMXIOM_HPP

// All-in-one convenience umbrella for camxiom.
//
// Includes everything the library can offer:
//   * the runtime geometry core   (<camxiom/core.hpp>)
//   * the calibration layer        (<camxiom/calibration.hpp>)
//   * the optional ROS / OpenCV interop layers
//
// The interop headers are self-guarded on their CAMXIOM_HAS_* build flag plus
// nothing when sensor_msgs / OpenCV headers are not reachable. Likewise the
// calibration API is Ceres-free at the surface, so this header compiles in a
// core-only build; only linking a calibration/interop symbol requires the
// corresponding layer to have been built (link `camxiom::camxiom`).
//
// This is the broadest include and therefore the heaviest: prefer
// <camxiom/core.hpp> alone when you only need ray<->pixel runtime geometry.
//
// Individual headers remain available; this is purely a convenience aggregate
// and changes no existing behaviour.

// --- Runtime geometry core (Eigen-only) -------------------------------------
#include "camxiom/core.hpp"

// --- Calibration layer (seed / init / PnP / intrinsics) ---------------------
#include "camxiom/calibration.hpp"

// --- Optional interop layers (harmless no-ops when the toolkit is absent) ---
#include "camxiom/opencv.hpp"
#include "camxiom/opencv/pnp.hpp"
#include "camxiom/ros.hpp"

#endif  // CAMXIOM__CAMXIOM_HPP
