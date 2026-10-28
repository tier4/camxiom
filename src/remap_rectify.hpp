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

#ifndef CAMXIOM__REMAP_RECTIFY_HPP
#define CAMXIOM__REMAP_RECTIFY_HPP

// Internal (NOT installed) helper: fitting the scale of a rectify output
// camera. makeRectifiedOutputModel() scales an output template by one common
// factor s (fx, fy and skew together); this helper finds the two scales that
// alpha = 0 and alpha = 1 stand for. Larger s means a narrower view.

#include "camxiom/types.hpp"  // CameraModel, ImageSize

#include <Eigen/Core>

namespace camxiom::detail
{

/// `inscribed_scale`: the smallest s at which every ray the output shows is
///   a valid source ray landing inside the source image (alpha = 0).
/// `contained_scale`: the largest s at which every valid source ray in the
///   image that the output keeps lands inside the output image (alpha = 1).
/// Each is on the safe side of the exact scale for the double evaluation of
/// the model formulas (up to rounding), within 3e-4 of it unless the search
/// could not settle a patch, and meaningful only when its `_valid` flag is
/// set.
struct RectifyScaleFit
{
  float inscribed_scale{0.0f};
  float contained_scale{0.0f};
  bool inscribed_valid{false};
  bool contained_valid{false};
};

/// The number of patches each search refines at most; past it, a search
/// returns the bound it holds (on the safe side, less tight).
constexpr int kRectifyMaxPatches = 1 << 18;

/// `unit_output` is the output template at s = 1 with its principal point at
/// the output image centre. Only the bounds that are asked for are computed.
RectifyScaleFit fitRectifyScale(
  const CameraModel &src_model, const ImageSize &src_size, const CameraModel &unit_output,
  const ImageSize &out_size, const Eigen::Matrix3f &src_from_output, float source_margin_px,
  const SolverOptions &solver_options, bool need_inscribed, bool need_contained,
  int max_patches = kRectifyMaxPatches
);

/// The output model at scale s: fx, fy and skew of `unit_output` times s.
CameraModel scaleOutputModel(const CameraModel &unit_output, float scale);

}  // namespace camxiom::detail

#endif  // CAMXIOM__REMAP_RECTIFY_HPP
