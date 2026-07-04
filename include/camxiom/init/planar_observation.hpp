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

#ifndef CAMXIOM__INIT__PLANAR_OBSERVATION_HPP
#define CAMXIOM__INIT__PLANAR_OBSERVATION_HPP

#include <Eigen/Core>

namespace camxiom::init
{

/// One observation of a planar calibration target.
///
/// `board_pts` contains target-plane (X, Y) coordinates and `image_pts`
/// contains the corresponding pixels. Both matrices store one point per
/// column and must have the same number of columns.
struct PlanarObservation
{
  Eigen::Matrix2Xd board_pts;
  Eigen::Matrix2Xd image_pts;
};

}  // namespace camxiom::init

#endif  // CAMXIOM__INIT__PLANAR_OBSERVATION_HPP
