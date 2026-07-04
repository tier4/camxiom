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

#ifndef CAMXIOM__LUT64_HPP
#define CAMXIOM__LUT64_HPP

#include "camxiom/lut.hpp"
#include "camxiom/types64.hpp"

namespace camxiom
{

namespace detail
{
// double LUT uses the double-precision solver options (see lut.hpp).
template <>
struct LutSolverOptions<double>
{
  using type = SolverOptions64;
};
}  // namespace detail

/// Pre-computed lookup table for fast approximate inverse projection (double).
///
/// Alias of the scalar-templated InverseLutT<double> (see camxiom/lut.hpp); the
/// float32 counterpart is `InverseLut`. Both share one class template (#1 step 5),
/// so the public API and object layout are unchanged.
using InverseLut64 = InverseLutT<double>;

}  // namespace camxiom

#endif  // CAMXIOM__LUT64_HPP
