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

#ifndef CAMXIOM__INTERNAL__CONSTANTS_HPP
#define CAMXIOM__INTERNAL__CONSTANTS_HPP

// Math constants used throughout camxiom.
// Kept locally so the library has no dependency on external "common" helpers.
//
// STABILITY NOTE: this header lives under internal/ and is EXCLUDED from the
// public API snapshot, but it is installed (public headers reference it in
// default member initializers) and the namespace is camxiom::constants.
// Treat its contents as an implementation detail with no compatibility
// guarantee — consumers should define their own pi rather than depend on
// these names surviving a release.

#if __cplusplus >= 202002L
#include <numbers>
#endif

namespace camxiom::constants
{

#if __cplusplus >= 202002L
inline constexpr double kPi = std::numbers::pi;
inline constexpr double kHalfPi = std::numbers::pi / 2.0;
#else
inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kHalfPi = 1.57079632679489661923;
#endif

inline constexpr float kPiF = static_cast<float>(kPi);
inline constexpr float kHalfPiF = static_cast<float>(kHalfPi);

}  // namespace camxiom::constants

#endif  // CAMXIOM__INTERNAL__CONSTANTS_HPP
