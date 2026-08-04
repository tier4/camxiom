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

#ifndef CAMXIOM__INTERNAL__PREPARED_PROJECTION_HPP
#define CAMXIOM__INTERNAL__PREPARED_PROJECTION_HPP

// Model-derived projection state.
//
// Two validity tests in the per-model projection cores read nothing from the
// ray or pixel being projected, only from the CameraModel:
//
//   * the FOV cap compares z against norm * cos(theta_max) — double-sphere,
//     EUCM and omnidirectional, forward and inverse;
//   * the double-sphere bijectivity bound (Usenko et al. 2018, eq. 43-45)
//     compares z against -w2 * d1, and w2 costs a square root and two
//     divisions to derive from xi and alpha.
//
// Evaluated inside the kernel, both make every projected point re-derive a
// value that is constant across a whole point cloud. This type carries them so
// a caller that projects many points against one model derives them once.
//
// STABILITY NOTE: like internal/constants.hpp, this header lives under
// internal/ and is EXCLUDED from the public API snapshot, but it is installed
// because a public header (validated_model.hpp) stores this type by value.
// Treat it as an implementation detail with no compatibility guarantee.
//
// The factory that fills it in, detail_impl::prepareProjection(), lives in
// src/detail/projection_common.hpp: only translation units that actually
// project need it, and it pulls in the double-sphere math.

namespace camxiom::detail_impl
{

template <typename T>
struct PreparedProjectionT
{
  /// theta_max caps the FOV below the wide-angle default of pi.
  bool has_theta_cap{false};
  /// cos(theta_max), read only when has_theta_cap. Defaults to cos(pi) = -1 so
  /// an uncapped model accepts every forward direction even if a caller
  /// ignores the flag.
  T cos_theta_max{T(-1)};
  /// -w2 of the double-sphere bijectivity region (Usenko et al. 2018,
  /// eq. 43-45). Zero for every other projection type, which never reads it.
  T ds_neg_w2{T(0)};
};

using PreparedProjection = PreparedProjectionT<float>;
using PreparedProjection64 = PreparedProjectionT<double>;

}  // namespace camxiom::detail_impl

#endif  // CAMXIOM__INTERNAL__PREPARED_PROJECTION_HPP
