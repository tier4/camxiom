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

#ifndef CAMXIOM__CALIB__MAPPING_COMPARE_HPP
#define CAMXIOM__CALIB__MAPPING_COMPARE_HPP

#include "camxiom/types.hpp"

namespace camxiom::calib
{

/// Domain over which two models' pixel->ray mappings are compared: a uniform
/// pixel lattice spanning the full image rectangle. This is a MECHANISM:
/// camxiom evaluates the mappings and reports the numbers; what counts as
/// "the same mapping" (thresholds, clustering, required-domain policy) is the
/// caller's decision.
struct MappingDomainSpec
{
  int image_width{0};
  int image_height{0};
  /// Lattice resolution (columns x rows of sample pixels, corners included).
  int lattice_cols{64};
  int lattice_rows{64};
};

/// Result of comparing two models' pixel->ray mappings over the lattice.
/// Angular deltas are the angles (radians) between the two unit ray
/// directions at pixels where BOTH mappings are valid; validity mismatches
/// are counted separately so a caller can hard-fail on "one model cannot
/// even cover the domain the other covers".
struct MappingComparison
{
  StatusCode status{StatusCode::INVALID_INPUT};
  int lattice_points{0};
  int both_valid{0};
  int only_a_valid{0};
  int only_b_valid{0};
  int neither_valid{0};
  double rms_angular_delta_rad{0.0};
  double p95_angular_delta_rad{0.0};
  double max_angular_delta_rad{0.0};
};

/// Compare two camera models AS MAPPINGS (pixel -> unit ray) over the given
/// lattice. Parameter-space comparison (model_compare.hpp) cannot distinguish
/// "different parameters, same mapping" (aliasing/compensation valleys) from
/// genuinely different cameras; this can. Both models must pass
/// validateCameraModel and the spec must be positive-sized, else
/// INVALID_INPUT.
[[nodiscard]] MappingComparison compareProjectionMappings(
  const CameraModel &model_a, const CameraModel &model_b, const MappingDomainSpec &spec
);

/// How much of the lattice the model's pixel->ray mapping covers.
struct MappingDomainReport
{
  StatusCode status{StatusCode::INVALID_INPUT};
  int lattice_points{0};
  int valid_points{0};
};

/// Evaluate a single model's pixel->ray validity over the lattice, so a
/// caller can require "the mapping is valid over the (rectangular) export
/// domain" before trusting or persisting a model.
[[nodiscard]] MappingDomainReport validateMappingDomain(
  const CameraModel &model, const MappingDomainSpec &spec
);

}  // namespace camxiom::calib

#endif  // CAMXIOM__CALIB__MAPPING_COMPARE_HPP
