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

#ifndef CAMXIOM__CALIBRATION_HPP
#define CAMXIOM__CALIBRATION_HPP

// Convenience umbrella for the CALIBRATION layer (camxiom::calib).
//
// Pulls in every public header of the calibration layer: the data-independent
// default seed, all initial-guess estimators (linear + Ceres-polished), the
// Ceres-based PnP solver, and the single-pass intrinsics calibrator.
//
// These headers are all Ceres-free at the API surface (Ceres is hidden behind
// the PnpSolver PIMPL), so this umbrella is safe to *include* even in a
// core-only build. USING the calibration symbols still requires linking the
// camxiom_calib layer, i.e. building with CAMXIOM_WITH_CERES (link
// `camxiom::calib` or the `camxiom::camxiom` umbrella target).
//
// This header does not implicitly pull in <camxiom/core.hpp>; include the core
// header (or the all-in-one <camxiom/camxiom.hpp>) for the runtime geometry
// types the calibration API operates on.
//
// Individual headers remain available; this is purely a convenience aggregate
// and changes no existing behaviour.

// --- Data-independent default camera-model seed -----------------------------
#include "camxiom/default_seed.hpp"

// --- Initial-guess estimators (linear + Ceres-polished) ---------------------
#include "camxiom/init/dlt_pnp.hpp"
#include "camxiom/init/double_sphere.hpp"
#include "camxiom/init/eucm.hpp"
#include "camxiom/init/homography.hpp"
#include "camxiom/init/kb4_fisheye.hpp"
#include "camxiom/init/mei_omni.hpp"
#include "camxiom/init/pinhole_opencv.hpp"
#include "camxiom/init/pinhole_zhang.hpp"
#include "camxiom/init/planar_observation.hpp"

// --- Ceres-based PnP solver (Ceres hidden behind a PIMPL) -------------------
#include "camxiom/optimizer/pnp_flag.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"

// --- Single-pass intrinsics calibration -------------------------------------
#include "camxiom/calib/convert.hpp"
#include "camxiom/calib/intrinsics.hpp"

#endif  // CAMXIOM__CALIBRATION_HPP
