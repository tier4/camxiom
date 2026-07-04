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

// Internal (NOT installed) helper: the rectify-target focal computation
// extracted from remap.cpp (LOADMAP R1). This is the single source-boundary
// pass that produces the per-axis inscribed / circumscribed focal estimates
// consumed by makeRectifiedOutputModel() and buildRectifyMap(). It is a pure
// numeric helper with no public-API surface; splitting it out of the 1198-line
// remap.cpp is a readability-only move -- behaviour and numerics are identical
// to the former in-file anonymous-namespace implementation.

#include "camxiom/remap.hpp"  // ImageSize
#include "camxiom/types.hpp"  // CameraModel

#include <Eigen/Core>

namespace camxiom::detail
{

// Per-axis output of the unified rectify-focal helper.
//
// `*_inscribed`: maximum focal such that all 4 source edges land inside the
//   output frame on the corresponding side (per-axis tight bound).
// `*_circumscribed`: focal that exactly contains the overall extremes of all
//   sampled source rays in the output frame (alpha=1-style widest-FOV fit,
//   D36 / D45).
//
// `inscribed_valid` is false when at least one edge contributed no sample
// (rays past the theta clamp, etc.) — the inscribed rectangle is then
// degenerate. `circumscribed_valid` is false when no source ray survived the
// clamp at all.
struct RectifyFocalPairs
{
  float fx_inscribed{0.0f};
  float fy_inscribed{0.0f};
  float fx_circumscribed{0.0f};
  float fy_circumscribed{0.0f};
  bool inscribed_valid{false};
  bool circumscribed_valid{false};
};

// Single source-boundary pass producing both inscribed and circumscribed
// per-axis focal estimates (see remap_rectify.cpp for the detailed contract).
// Each caller checks the `_valid` flag for the policy it cares about.
RectifyFocalPairs computeRectifyFocalPairs(
  const CameraModel &src_model, const ImageSize &src_size, const ImageSize &out_size,
  const Eigen::Matrix3f &src_from_output_rotation, int boundary_sample_count, float source_margin_px
);

}  // namespace camxiom::detail

#endif  // CAMXIOM__REMAP_RECTIFY_HPP
