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

#ifndef CAMXIOM__CALIB__UNCERTAINTY_DETAIL_HPP
#define CAMXIOM__CALIB__UNCERTAINTY_DETAIL_HPP

// Internal (non-installed) header — pure, side-effect-free helpers backing the
// C3/C4/C5 calibration diagnostics in calib/intrinsics.cpp. Split out so the
// tricky pieces can be unit-tested directly, without running a full calibrate():
//   * the basis-invariant weak-subspace aggregation (C5 ④(b)), and
//   * the near-bound proximity test (C5 ⑤).
// Lives under src/ (a PRIVATE include of camxiom_calib), so it is NOT part of
// the public API surface scanned by the Rel1a snapshot test.

#include "optimizer/pnp/pnp_parameter_bounds.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace camxiom::calib::detail
{

// ---- C5 ④(b): basis-invariant weak-subspace aggregation ------------------

/// Threshold, on the per-parameter projector-diagonal score in [0, 1], above
/// which a parameter is reported as participating in the weak subspace.
inline constexpr double kUnderdeterminedScoreThreshold = 0.1;

struct WeakSubspaceScores
{
  Eigen::VectorXd score;                     // length p, (P_weak)_ii in [0, 1]
  std::vector<std::string> underdetermined;  // labels with score >= threshold
};

/// Aggregate the contribution of each parameter to the weak subspace spanned by
/// the columns of `weak_basis` (p x k: the eigenvectors of the column-scaled
/// reduced normal matrix whose singular value is below the observability
/// threshold). The per-parameter score is the diagonal of the orthogonal
/// projector P_weak = V Vᵀ, i.e. score_i = Σ_k V(i,k)².
///
/// Because P_weak = (V Q)(V Q)ᵀ for ANY orthogonal Q, the scores — and hence
/// the reported labels — are INVARIANT to the eigenbasis chosen within the weak
/// subspace. A single-eigenvector readout lacks this property when several
/// small eigenvalues are near-degenerate (the basis can arbitrarily mix them).
///
/// `labels` is parallel to the rows of `weak_basis`. A label whose score
/// reaches `score_threshold` is reported; the score denotes participation in
/// the weak subspace, not that the parameter is individually unobservable.
inline WeakSubspaceScores aggregateWeakSubspace(
  const Eigen::MatrixXd &weak_basis, const std::vector<std::string> &labels,
  double score_threshold = kUnderdeterminedScoreThreshold
)
{
  WeakSubspaceScores out;
  const Eigen::Index p = weak_basis.rows();
  if (weak_basis.cols() == 0)
  {
    out.score = Eigen::VectorXd::Zero(p);
    return out;
  }
  out.score = weak_basis.cwiseAbs2().rowwise().sum();  // diag(V Vᵀ)
  for (Eigen::Index i = 0; i < p; ++i)
  {
    if (out.score(i) >= score_threshold && i < static_cast<Eigen::Index>(labels.size()))
    {
      out.underdetermined.push_back(labels[static_cast<std::size_t>(i)]);
    }
  }
  return out;
}

// ---- C5 ⑤: near-bound proximity test -------------------------------------

/// Per-side absolute / relative tolerances for the "final estimate sits at/near
/// a box bound" test. Each side is scaled by its OWN bound and the estimate, so
/// a very wide artificial bound on one side never inflates the tolerance on the
/// opposite side (e.g. the default focal upper ~10*image_height must not make
/// the focal lower ~1 px look "near").
inline constexpr double kNearBoundAbsTol = 1e-6;
inline constexpr double kNearBoundRelTol = 1e-3;

struct BoundProximity
{
  bool near_lower{false};
  bool near_upper{false};
  bool anyNear() const { return near_lower || near_upper; }
};

/// Classify whether `x` sits at/near the lower and/or upper side of `bound`.
/// Only sides that exist (has_lower / has_upper) are tested, so a genuinely
/// one-sided bound (e.g. EUCM beta) is never compared against a missing side.
inline BoundProximity classifyBoundProximity(double x, const optimizer::detail::ScalarBound &bound)
{
  BoundProximity result;
  if (bound.has_lower)
  {
    const double tol =
      std::max(kNearBoundAbsTol, kNearBoundRelTol * std::max(std::abs(bound.lower), std::abs(x)));
    result.near_lower = std::abs(x - bound.lower) <= tol;
  }
  if (bound.has_upper)
  {
    const double tol =
      std::max(kNearBoundAbsTol, kNearBoundRelTol * std::max(std::abs(bound.upper), std::abs(x)));
    result.near_upper = std::abs(x - bound.upper) <= tol;
  }
  return result;
}

}  // namespace camxiom::calib::detail

#endif  // CAMXIOM__CALIB__UNCERTAINTY_DETAIL_HPP
