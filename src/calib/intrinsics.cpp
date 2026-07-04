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

#include "camxiom/calib/intrinsics.hpp"

#include "calib/uncertainty_detail.hpp"
#include "camxiom/init/dlt_pnp.hpp"
#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"
#include "jacobian/full_jacobian64_internal.hpp"
#include "optimizer/pnp/angle_axis_jacobian.hpp"
#include "optimizer/pnp/pnp_parameter_bounds.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace camxiom::calib
{

namespace
{

/// Convert a rotation matrix to a Rodrigues vector (axis * angle).
/// Local copy of the established camxiom helper (mei_omni.cpp); kept
/// per-file rather than shared per the accepted camxiom pattern.
Eigen::Vector3d rotationMatrixToAngleAxis(const Eigen::Matrix3d &R)
{
  const Eigen::AngleAxisd aa(R);
  return aa.axis() * aa.angle();
}

/// Convert a Rodrigues vector back to a rotation matrix.
Eigen::Matrix3d angleAxisToRotationMatrix(const Eigen::Vector3d &rvec)
{
  const double angle = rvec.norm();
  if (!(angle > 0.0))
  {
    return Eigen::Matrix3d::Identity();
  }
  const Eigen::AngleAxisd aa(angle, rvec / angle);
  return aa.toRotationMatrix();
}

/// True iff every parameter that calibrate() will hand back is finite.
bool resultIsFinite(
  const CameraModel &model, const std::vector<Eigen::Matrix3d> &R,
  const std::vector<Eigen::Vector3d> &t
)
{
  if (!std::isfinite(static_cast<double>(model.intrinsics.fx)) ||
      !std::isfinite(static_cast<double>(model.intrinsics.fy)) ||
      !std::isfinite(static_cast<double>(model.intrinsics.cx)) ||
      !std::isfinite(static_cast<double>(model.intrinsics.cy)) ||
      !std::isfinite(static_cast<double>(model.intrinsics.skew)))
  {
    return false;
  }
  for (float c : model.distortion.coeffs)
  {
    if (!std::isfinite(static_cast<double>(c)))
    {
      return false;
    }
  }
  if (!std::isfinite(static_cast<double>(model.projection.xi)) ||
      !std::isfinite(static_cast<double>(model.projection.alpha)) ||
      !std::isfinite(static_cast<double>(model.projection.beta)) ||
      !std::isfinite(static_cast<double>(model.projection.theta_max)))
  {
    return false;
  }
  for (const auto &Ri : R)
  {
    if (!Ri.allFinite())
    {
      return false;
    }
  }
  for (const auto &ti : t)
  {
    if (!ti.allFinite())
    {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Parameter uncertainty (C3) helpers.
//
// We reconstruct the reduced normal equations JᵀJ at the solution using the
// SAME analytic full Jacobian the ANALYTICAL solver path uses
// (rayToPixelWithFullJacobian64), marginalise the per-view extrinsics out via
// the Schur complement, and scale by the residual variance to obtain the
// linearised (Gauss-Newton) covariance of the free intrinsic / distortion /
// projection parameters. This depends only on core geometry (Eigen +
// rayToPixelWithFullJacobian64), never on Ceres internals, and therefore is
// faithful to what the solver optimised while keeping the public API change
// confined to CalibrationResult.
// ---------------------------------------------------------------------------

/// ∂(R(omega)·p)/∂omega — the SAME single-source formula the solver's
/// analytical batch cost and GN path use (optimizer/pnp/
/// angle_axis_jacobian.hpp), so the rebuilt normal equations stay faithful
/// to the solver by construction.
using camxiom::optimizer::detail::angleAxisPointJacobian;

/// Which intrinsic / distortion / projection parameter a free column maps to.
enum class ParamKind { FX, FY, CX, CY, DIST, PROJ };
struct FreeParam
{
  ParamKind kind;
  int idx;  // DIST coeff index, or PROJ slot index (0=xi,1=alpha,2=beta)
};

/// Enumerate the free (not locked) intrinsic parameter columns, in the canonical
/// order fx, fy, cx, cy, dist[k], projection slots. Fills `labels` in parallel.
std::vector<FreeParam> enumerateFreeParams(
  const CameraModel &model, camxiom::optimizer::PnpFlag flags, std::vector<std::string> &labels
)
{
  namespace opt = camxiom::optimizer;
  std::vector<FreeParam> entries;
  labels.clear();

  if (!opt::hasFlag(flags, opt::PnpFlag::FIX_FOCAL_LENGTHS))
  {
    entries.push_back({ParamKind::FX, 0});
    labels.emplace_back("fx");
    entries.push_back({ParamKind::FY, 0});
    labels.emplace_back("fy");
  }
  if (!opt::hasFlag(flags, opt::PnpFlag::FIX_PRINCIPAL_POINTS))
  {
    entries.push_back({ParamKind::CX, 0});
    labels.emplace_back("cx");
    entries.push_back({ParamKind::CY, 0});
    labels.emplace_back("cy");
  }

  const int dc = std::min(static_cast<int>(model.distortion.count), 14);
  for (int k = 0; k < dc; ++k)
  {
    const auto bit = static_cast<opt::PnpFlag>(std::uint64_t{1} << (8 + k));
    if (!opt::hasFlag(flags, bit))
    {
      entries.push_back({ParamKind::DIST, k});
      labels.emplace_back("dist[" + std::to_string(k) + "]");
    }
  }

  if (!opt::hasFlag(flags, opt::PnpFlag::FIX_PROJECTION_PARAMS))
  {
    static const char *const kProjLabel[3] = {"xi", "alpha", "beta"};
    std::vector<int> free_slots;
    switch (model.projection.type)
    {
      case camxiom::ProjectionModelType::OMNIDIRECTIONAL:
        free_slots = {0};
        break;
      case camxiom::ProjectionModelType::DOUBLE_SPHERE:
        free_slots = {0, 1};
        break;
      case camxiom::ProjectionModelType::EUCM:
        free_slots = {1, 2};
        break;
      case camxiom::ProjectionModelType::PINHOLE:
      case camxiom::ProjectionModelType::FISHEYE_THETA:
      case camxiom::ProjectionModelType::UNKNOWN:
        break;
    }
    for (int s : free_slots)
    {
      entries.push_back({ParamKind::PROJ, s});
      labels.emplace_back(kProjLabel[s]);
    }
  }
  return entries;
}

/// Reduced free-parameter normal matrix S = A − Σ_v B_v D_v⁻¹ B_vᵀ (per-view
/// extrinsics marginalised via the Schur complement), together with the raw
/// residual sum of squares and the degrees of freedom needed to scale it.
///
/// Returns false ONLY when the reduced normal matrix cannot be BUILT (C5 #8):
/// no free parameters, a per-view size mismatch, a per-view pose block that is
/// not numerically positive-definite (C5 #9), or a non-finite S. A non-positive
/// `dof` does NOT fail here — that is a C3-only condition (see fillUncertainty),
/// because C4 (observability) needs only S and is independent of the residual
/// variance. `dof` may therefore be returned <= 0.
struct ReducedNormal
{
  Eigen::MatrixXd S;                // p × p (free-parameter information)
  double rss{0.0};                  // residual sum of squares over valid points
  long dof{0};                      // 2*N_valid − p_free − d_pose (may be <= 0)
  std::vector<std::string> labels;  // p labels, parallel to S rows/cols
};

bool buildReducedNormal(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t,
  camxiom::optimizer::PnpFlag flags, ReducedNormal &out
)
{
  namespace opt = camxiom::optimizer;
  const std::size_t nv = views.size();
  if (nv == 0 || R.size() != nv || t.size() != nv)
  {
    return false;
  }

  std::vector<std::string> labels;
  const std::vector<FreeParam> params = enumerateFreeParams(model, flags, labels);
  const int p = static_cast<int>(params.size());
  if (p == 0)
  {
    return false;
  }

  const bool fix_ext = opt::hasFlag(flags, opt::PnpFlag::FIX_EXTRINSICS);
  const camxiom::CameraModel64 m64 = camxiom::toCameraModel64(model);
  // The model is fixed for the whole rebuild: validate ONCE, then use the
  // unchecked Jacobian entry in the point loop. An invalid model cannot
  // yield a reduced normal matrix (previously: every point would have been
  // rejected point-by-point, leaving S empty), so fail S construction here.
  if (camxiom::validateCameraModel64(m64) != StatusCode::OK)
  {
    return false;
  }
  const double fx = static_cast<double>(model.intrinsics.fx);
  const double fy = static_cast<double>(model.intrinsics.fy);
  const double sk = static_cast<double>(model.intrinsics.skew);

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
  std::vector<Eigen::MatrixXd> Bv(nv, Eigen::MatrixXd::Zero(p, 6));
  std::vector<Eigen::Matrix<double, 6, 6>> Dv(nv, Eigen::Matrix<double, 6, 6>::Zero());
  std::vector<std::size_t> view_valid(nv, 0);  // valid points per view

  double rss = 0.0;
  std::size_t n_valid = 0;

  // Scratch intrinsic Jacobian block (2 × p), reused across all points to avoid
  // a per-point heap allocation. p is constant for the whole call.
  Eigen::MatrixXd Ji(2, p);

  for (std::size_t v = 0; v < nv; ++v)
  {
    const auto &view = views[v];
    const Eigen::Matrix3d &Ri = R[v];
    const Eigen::Vector3d &ti = t[v];
    const Eigen::Vector3d omega = rotationMatrixToAngleAxis(Ri);
    const std::size_t n = view.world_points.size();
    if (view.image_points.size() != n)
    {
      return false;
    }

    for (std::size_t j = 0; j < n; ++j)
    {
      const Eigen::Vector3d &Pw = view.world_points[j];
      const Eigen::Vector3d p_cam = Ri * Pw + ti;
      const camxiom::FullProjectionJacobian64 fj =
        camxiom::detail::rayToPixelWithFullJacobian64Unchecked(m64, p_cam);
      if (fj.status != StatusCode::OK)
      {
        continue;  // invalid points carry no geometric information (penalty only)
      }

      const Eigen::Vector2d &obs = view.image_points[j];
      const double du = obs.x() - fj.pixel.u;
      const double dv = obs.y() - fj.pixel.v;
      rss += du * du + dv * dv;
      ++n_valid;
      ++view_valid[v];

      // Intrinsic Jacobian block (2 × p), free columns only. Mirrors the
      // analytical batch cost's ∂residual/∂param signs exactly.
      Ji.setZero();
      for (int c = 0; c < p; ++c)
      {
        const FreeParam &e = params[static_cast<std::size_t>(c)];
        switch (e.kind)
        {
          case ParamKind::FX:
            Ji(0, c) = -fj.xd;
            break;
          case ParamKind::FY:
            Ji(1, c) = -fj.yd;
            break;
          case ParamKind::CX:
            Ji(0, c) = -1.0;
            break;
          case ParamKind::CY:
            Ji(1, c) = -1.0;
            break;
          case ParamKind::DIST: {
            const auto k = static_cast<std::size_t>(e.idx);
            Ji(0, c) = -(fx * fj.dxd_ddist[k] + sk * fj.dyd_ddist[k]);
            Ji(1, c) = -(fy * fj.dyd_ddist[k]);
            break;
          }
          case ParamKind::PROJ: {
            const auto k = static_cast<std::size_t>(e.idx);
            Ji(0, c) = -(fx * fj.dxd_dproj[k] + sk * fj.dyd_dproj[k]);
            Ji(1, c) = -(fy * fj.dyd_dproj[k]);
            break;
          }
        }
      }

      A.noalias() += Ji.transpose() * Ji;

      if (!fix_ext)
      {
        const Eigen::Matrix<double, 2, 3> &Jp = fj.J_point;
        const Eigen::Matrix3d J_aa = angleAxisPointJacobian(omega, Pw);
        Eigen::Matrix<double, 2, 6> Je;
        Je.block<2, 3>(0, 0) = -(Jp * J_aa);  // ∂residual/∂rvec
        Je.block<2, 3>(0, 3) = -Jp;           // ∂residual/∂tvec
        Bv[v].noalias() += Ji.transpose() * Je;
        Dv[v].noalias() += Je.transpose() * Je;
      }
    }
  }

  // Marginalise the per-view extrinsics via the Schur complement. Only ACTIVE
  // views (>= 1 valid point) contribute (C5 #8): a view with zero valid points
  // has its pose absent from the residual graph (D_v = B_v = 0), so it is
  // skipped from the Schur sum AND excluded from the pose degrees of freedom —
  // it must not, on its own, make S unbuildable.
  Eigen::MatrixXd S = A;
  long n_active = 0;
  if (!fix_ext)
  {
    for (std::size_t v = 0; v < nv; ++v)
    {
      if (view_valid[v] == 0)
      {
        continue;  // inactive pose block: contributes nothing, costs no DOF
      }
      ++n_active;

      // C5 #9: do not trust LDLT::info()==Success alone. Confirm numerical
      // positive-definiteness from the pivots (all strictly positive, no
      // extreme min/max spread) and the finiteness of the Schur update. A
      // degenerate active pose block means S is not well-formed -> unavailable.
      // No silent regularisation / pseudo-inverse (C5 #3 addendum).
      Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Dv[v]);
      if (ldlt.info() != Eigen::Success)
      {
        return false;
      }
      const Eigen::Matrix<double, 6, 1> pivots = ldlt.vectorD();
      const double dmin = pivots.minCoeff();
      const double dmax = pivots.maxCoeff();
      if (!(dmin > 0.0) || !std::isfinite(dmax) || dmin < 1e-12 * dmax)
      {
        return false;
      }
      const Eigen::MatrixXd DinvBt = ldlt.solve(Bv[v].transpose());  // 6 × p
      if (!DinvBt.allFinite())
      {
        return false;
      }
      S.noalias() -= Bv[v] * DinvBt;
    }
  }

  if (!S.allFinite())
  {
    return false;
  }

  const long d_pose = fix_ext ? 0L : 6L * n_active;
  out.S = std::move(S);
  out.rss = rss;
  out.dof = 2L * static_cast<long>(n_valid) - static_cast<long>(p) - d_pose;
  out.labels = std::move(labels);
  return true;
}

/// C3: fill the per-parameter standard deviations from the reduced normal
/// matrix. Leaves uncertainty_available=false (and the vectors empty) when the
/// covariance cannot be trusted. This owns the residual-variance conditions
/// (C5 #8): a non-positive dof or a non-finite RSS / variance means there is no
/// meaningful covariance (never returns negative variances or NaNs), and a
/// rank-deficient S (which is itself the C4 degeneracy signal) fails the LLT.
void fillUncertainty(const ReducedNormal &rn, CalibrationResult &result)
{
  if (rn.dof <= 0 || !std::isfinite(rn.rss))
  {
    return;  // too few effective observations: no covariance (C3-only condition)
  }
  const double sigma2 = rn.rss / static_cast<double>(rn.dof);
  if (!std::isfinite(sigma2))
  {
    return;
  }

  // Covariance of the free intrinsic block = sigma^2 · S⁻¹. LLT doubles as a
  // positive-definiteness (full-rank) test.
  const Eigen::Index p = rn.S.rows();
  Eigen::LLT<Eigen::MatrixXd> llt(rn.S);
  if (llt.info() != Eigen::Success)
  {
    return;
  }
  const Eigen::MatrixXd Sinv = llt.solve(Eigen::MatrixXd::Identity(p, p));

  std::vector<double> stds(static_cast<std::size_t>(p));
  for (Eigen::Index i = 0; i < p; ++i)
  {
    const double var = sigma2 * Sinv(i, i);
    if (!std::isfinite(var))
    {
      return;
    }
    stds[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, var));
  }

  result.parameter_std = std::move(stds);
  result.uncertainty_labels = rn.labels;
  result.uncertainty_available = true;
}

/// C4: observability / degeneracy diagnostic from the SAME reduced normal
/// matrix. Jacobi-scales S to unit diagonal (so the condition number is
/// scale-invariant across the mixed units of focal / principal-point /
/// distortion), then reads the singular spectrum of the column-scaled system.
/// Diagnostic only (D46): it never mutates lock_flags or drops views, and never
/// silently regularises a degenerate S (C5 #3 addendum). Sets the C5
/// observability_available/observability_ok state contract (C5 #10):
///   * eigen-solver failure or a clearly-negative eigenvalue -> available=false;
///   * a zero/negative diagonal or a rank deficiency -> available=true, ok=false,
///     min_singular_value=0, normalized_condition_number=+inf;
///   * otherwise available=true and ok reflects the singular-value threshold.
void fillObservability(const ReducedNormal &rn, CalibrationResult &result)
{
  const Eigen::Index p = rn.S.rows();
  const double kInf = std::numeric_limits<double>::infinity();

  // Symmetrise before reading the diagonal (kill rounding asymmetry, C5 #10).
  const Eigen::MatrixXd S = 0.5 * (rn.S + rn.S.transpose());

  // Jacobi (diagonal) preconditioning: S~ = D^{-1/2} S D^{-1/2}, unit diagonal.
  // A non-positive / non-finite diagonal means that parameter direction carries
  // no information: a SUCCESSFULLY diagnosed rank deficiency.
  std::vector<std::string> zero_info;
  Eigen::VectorXd inv_sqrt_diag(p);
  for (Eigen::Index i = 0; i < p; ++i)
  {
    const double dii = S(i, i);
    if (!(dii > 0.0) || !std::isfinite(dii))
    {
      zero_info.push_back(rn.labels[static_cast<std::size_t>(i)]);
      inv_sqrt_diag(i) = 0.0;
    }
    else
    {
      inv_sqrt_diag(i) = 1.0 / std::sqrt(dii);
    }
  }
  if (!zero_info.empty())
  {
    result.observability_available = true;
    result.observability_ok = false;
    result.min_singular_value = 0.0;
    result.normalized_condition_number = kInf;  // rank-deficient: +inf, not 0/NaN
    result.underdetermined_parameters = std::move(zero_info);
    return;
  }

  Eigen::MatrixXd Sn = inv_sqrt_diag.asDiagonal() * S * inv_sqrt_diag.asDiagonal();
  Sn = 0.5 * (Sn + Sn.transpose());  // kill rounding asymmetry

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sn);
  if (es.info() != Eigen::Success)
  {
    // The eigenvalue solver itself failed: no diagnostic value was produced.
    result.observability_available = false;
    result.observability_ok = false;
    result.min_singular_value = 0.0;
    result.normalized_condition_number = 0.0;  // "not computed", not healthy
    return;
  }

  const Eigen::VectorXd &evals = es.eigenvalues();  // ascending
  const double lambda_max_raw = evals(p - 1);
  // A clearly-negative eigenvalue (beyond rounding) is a numerical breakdown of
  // a matrix that should be positive-semidefinite: treat as unavailable.
  const double neg_tol = 1e-9 * std::max(1.0, std::abs(lambda_max_raw));
  if (evals(0) < -neg_tol)
  {
    result.observability_available = false;
    result.observability_ok = false;
    result.min_singular_value = 0.0;
    result.normalized_condition_number = 0.0;
    return;
  }

  // Analysis succeeded (even if degenerate): clamp tiny negatives to 0.
  const double lambda_min = std::max(0.0, evals(0));
  const double lambda_max = std::max(0.0, lambda_max_raw);
  const double s_min = std::sqrt(lambda_min);
  const double s_max = std::sqrt(lambda_max);

  result.observability_available = true;
  result.min_singular_value = s_min;
  result.normalized_condition_number = (s_min > 0.0) ? (s_max / s_min) : kInf;

  // Conservative degeneracy threshold on the smallest singular value of the
  // column-scaled Jacobian. For a unit-diagonal S the eigenvalues average 1, so
  // a healthy system keeps s_min well above this; a coupled / unobservable
  // direction drives s_min toward 0.
  constexpr double kMinSingular = 1e-3;
  result.observability_ok =
    std::isfinite(result.normalized_condition_number) && s_min >= kMinSingular;

  if (!result.observability_ok)
  {
    // Name the parameters participating in the WEAK SUBSPACE (all eigenmodes
    // whose singular value is below the threshold), aggregated basis-invariantly
    // via the projector P_weak = V_weak V_weakᵀ (C5 ④(b)). Reading a single
    // eigenvector is unstable when several small eigenvalues are near-degenerate.
    // Eigenvalues are ascending, so the weak modes are the leading columns.
    Eigen::Index k = 0;
    while (k < p && std::sqrt(std::max(0.0, evals(k))) < kMinSingular)
    {
      ++k;
    }
    const Eigen::MatrixXd weak_basis = es.eigenvectors().leftCols(k);
    const detail::WeakSubspaceScores scores = detail::aggregateWeakSubspace(weak_basis, rn.labels);
    result.underdetermined_parameters = scores.underdetermined;
  }
}

/// C5 ⑤: near-bound proximity. Independent of S (C5 #11): computed purely from
/// the final estimate, the lock flags, and the SAME box-bound descriptor the
/// solver used (so solver and diagnostic never disagree). Reports FREE, bounded
/// parameters whose final value sits at/near a bound; fixed parameters (absent
/// from the free set) and unbounded ones (distortion coefficients,
/// OMNIDIRECTIONAL xi) are never listed. This is populated even when the C3
/// covariance is unavailable, so a "parked on a bound" parameter is not lost.
void fillNearBounds(
  const CameraModel &model, camxiom::optimizer::PnpFlag flags,
  const camxiom::optimizer::PnpBound &lower_bound, const camxiom::optimizer::PnpBound &upper_bound,
  CalibrationResult &result
)
{
  namespace opt = camxiom::optimizer;
  std::vector<std::string> labels;
  const std::vector<FreeParam> params = enumerateFreeParams(model, flags, labels);
  if (params.empty())
  {
    return;
  }

  const opt::detail::CalibrationParameterBounds bounds =
    opt::detail::computeCalibrationParameterBounds(model, flags, lower_bound, upper_bound);

  std::vector<std::string> near;
  for (std::size_t c = 0; c < params.size(); ++c)
  {
    const FreeParam &e = params[c];
    const opt::detail::ScalarBound *b = nullptr;
    double x = 0.0;
    switch (e.kind)
    {
      case ParamKind::FX:
        b = &bounds.focal_lengths[0];
        x = static_cast<double>(model.intrinsics.fx);
        break;
      case ParamKind::FY:
        b = &bounds.focal_lengths[1];
        x = static_cast<double>(model.intrinsics.fy);
        break;
      case ParamKind::CX:
        b = &bounds.principal_points[0];
        x = static_cast<double>(model.intrinsics.cx);
        break;
      case ParamKind::CY:
        b = &bounds.principal_points[1];
        x = static_cast<double>(model.intrinsics.cy);
        break;
      case ParamKind::DIST:
        continue;  // distortion coefficients are unbounded
      case ParamKind::PROJ: {
        b = &bounds.projection[static_cast<std::size_t>(e.idx)];
        const double proj[3] = {
          static_cast<double>(model.projection.xi), static_cast<double>(model.projection.alpha),
          static_cast<double>(model.projection.beta)};
        x = proj[static_cast<std::size_t>(e.idx)];
        break;
      }
    }
    if (b == nullptr || (!b->has_lower && !b->has_upper))
    {
      continue;  // unbounded free parameter
    }
    if (detail::classifyBoundProximity(x, *b).anyNear())
    {
      near.push_back(labels[c]);
    }
  }

  result.uncertainty_has_parameters_near_bounds = !near.empty();
  result.parameters_at_or_near_bounds = std::move(near);
}

/// Driver: build the reduced normal matrix once and fill both the C3
/// per-parameter uncertainty and the C4 observability diagnostic from it.
void computeCalibrationDiagnostics(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t,
  camxiom::optimizer::PnpFlag flags, CalibrationResult &result
)
{
  ReducedNormal rn;
  if (!buildReducedNormal(views, model, R, t, flags, rn))
  {
    return;
  }
  fillUncertainty(rn, result);
  fillObservability(rn, result);
}

/// Status-only failure result: every other field keeps its default member
/// initialiser (atomic failure contract). Replaces the previous positional
/// aggregate initialisations, which silently shift meaning whenever a field
/// is inserted into CalibrationResult (it has grown to 16 fields with
/// C3/C4/C5).
CalibrationResult failedCalibrationResult(const StatusCode status)
{
  CalibrationResult result;
  result.status = status;
  return result;
}

}  // namespace

CalibrationResult calibrate(
  const std::vector<CalibrationView> &views, const CameraModel &initial_model, PnpFlag lock_flags,
  const CalibrationOptions &options
)
{
  // --------------------------------------------------------------------
  // Step 1: validation. Any failure returns an otherwise-default result
  // (atomic: never partially filled).
  // --------------------------------------------------------------------
  if (views.empty())
  {
    return failedCalibrationResult(StatusCode::INVALID_INPUT);
  }
  if (options.image_width <= 0 || options.image_height <= 0)
  {
    return failedCalibrationResult(StatusCode::INVALID_INPUT);
  }
  for (const auto &view : views)
  {
    if (view.world_points.size() != view.image_points.size())
    {
      return failedCalibrationResult(StatusCode::INVALID_INPUT);
    }
    // init::estimatePoseDLT requires N >= 6 correspondences per view.
    if (view.world_points.size() < 6U)
    {
      return failedCalibrationResult(StatusCode::INVALID_INPUT);
    }
  }
  if (camxiom::validateCameraModel(initial_model) != StatusCode::OK)
  {
    return failedCalibrationResult(StatusCode::INVALID_MODEL);
  }

  CameraModel solve_initial_model = initial_model;
  double pp_lower_x = 0.0;
  double pp_lower_y = 0.0;
  double pp_upper_x = 0.0;
  double pp_upper_y = 0.0;
  if (options.apply_principal_point_bounds)
  {
    const double ref_x = options.principal_point_reference_x;
    const double ref_y = options.principal_point_reference_y;
    const double r = options.principal_point_bound_relative_tolerance;
    if (!std::isfinite(ref_x) || !std::isfinite(ref_y) || !std::isfinite(r) || ref_x <= 0.0 || ref_y <= 0.0 || r < 0.0 || r >= 1.0)
    {
      return failedCalibrationResult(StatusCode::INVALID_INPUT);
    }
    pp_lower_x = ref_x * (1.0 - r);
    pp_lower_y = ref_y * (1.0 - r);
    pp_upper_x = ref_x * (1.0 + r);
    pp_upper_y = ref_y * (1.0 + r);

    // A previous warm start may already have escaped to an image edge. Bring it
    // back into the feasible region before DLT pose initialisation as well as
    // before Ceres sees the parameter block.
    solve_initial_model.intrinsics.cx = static_cast<float>(
      std::clamp(static_cast<double>(solve_initial_model.intrinsics.cx), pp_lower_x, pp_upper_x)
    );
    solve_initial_model.intrinsics.cy = static_cast<float>(
      std::clamp(static_cast<double>(solve_initial_model.intrinsics.cy), pp_lower_y, pp_upper_y)
    );
    // CameraModel stores intrinsics as float. Include the rounded seed exactly
    // so Ceres never sees an initial value a few ulps outside a double bound.
    pp_lower_x = std::min(pp_lower_x, static_cast<double>(solve_initial_model.intrinsics.cx));
    pp_lower_y = std::min(pp_lower_y, static_cast<double>(solve_initial_model.intrinsics.cy));
    pp_upper_x = std::max(pp_upper_x, static_cast<double>(solve_initial_model.intrinsics.cx));
    pp_upper_y = std::max(pp_upper_y, static_cast<double>(solve_initial_model.intrinsics.cy));
  }

  const std::size_t n_views = views.size();

  // --------------------------------------------------------------------
  // Step 2: per-view DLT-PnP poses (uniform across all models, pinhole
  // included). A non-OK return on ANY view aborts the whole call: the
  // app, not camxiom, owns view-skipping strategy (D46).
  // --------------------------------------------------------------------
  const camxiom::CameraModel64 m64 = camxiom::toCameraModel64(solve_initial_model);

  // Pose init lifts each observed pixel to a bearing ray via the model's
  // inverse (pixelToRay). A strong forward distortion (e.g. plumb_bob with a
  // large k1) is non-monotonic past its fold-over radius, so a board in the
  // far image periphery has pixels with no preimage and pixelToRay fails --
  // which would abort the whole solve. The distortion is only a refinement
  // the forward Ceres pass below recovers, so when the full-model lift fails
  // we retry the pose init with distortion disabled: the base projection is
  // invertible over its FOV, giving a usable initial pose that the forward
  // pass then refines under the full distortion. This mirrors OpenCV/Zhang,
  // whose initial extrinsics are computed with distortion off and refined
  // forward. A genuinely degenerate board (collinear / coincident corners)
  // still fails the distortion-free lift too and aborts, so a real bad view
  // is not silently accepted.
  camxiom::CameraModel initial_model_nodist = solve_initial_model;
  initial_model_nodist.distortion = camxiom::DistortionModel{};
  const camxiom::CameraModel64 m64_nodist = camxiom::toCameraModel64(initial_model_nodist);

  std::vector<Eigen::Matrix3d> R_views(n_views);
  std::vector<Eigen::Vector3d> t_views(n_views);

  for (std::size_t i = 0; i < n_views; ++i)
  {
    const auto &view = views[i];
    const Eigen::Index n = static_cast<Eigen::Index>(view.world_points.size());

    Eigen::Matrix3Xd world(3, n);
    Eigen::Matrix2Xd image(2, n);
    for (Eigen::Index j = 0; j < n; ++j)
    {
      const std::size_t jj = static_cast<std::size_t>(j);
      world.col(j) = view.world_points[jj];
      image.col(j) = view.image_points[jj];
    }

    Eigen::Matrix3d R_i;
    Eigen::Vector3d t_i;
    StatusCode dlt_status = camxiom::init::estimatePoseDLT(m64, world, image, R_i, t_i);
    if (dlt_status != StatusCode::OK)
    {
      // Full-model lift failed (most often a peripheral board past the
      // distortion's fold-over radius). Retry with distortion disabled.
      dlt_status = camxiom::init::estimatePoseDLT(m64_nodist, world, image, R_i, t_i);
    }
    if (dlt_status != StatusCode::OK)
    {
      // Both lifts failed: the view is genuinely degenerate (collinear /
      // coincident corners). Propagate the status verbatim (D46: the app,
      // not camxiom, owns any further view-skipping strategy).
      return failedCalibrationResult(dlt_status);
    }
    R_views[i] = R_i;
    t_views[i] = t_i;
  }

  // --------------------------------------------------------------------
  // Step 3: build PnpSolver inputs.
  // --------------------------------------------------------------------
  camxiom::optimizer::ObjectPointSets obj_sets;
  camxiom::optimizer::ImagePointSets img_sets;
  obj_sets.reserve(n_views);
  img_sets.reserve(n_views);
  for (const auto &view : views)
  {
    const std::size_t m = view.world_points.size();
    camxiom::optimizer::ObjectPoints obj;
    camxiom::optimizer::ImagePoints img;
    obj.reserve(m);
    img.reserve(m);
    for (std::size_t j = 0; j < m; ++j)
    {
      obj.push_back(view.world_points[j]);
      img.push_back(view.image_points[j]);
    }
    obj_sets.push_back(std::move(obj));
    img_sets.push_back(std::move(img));
  }

  camxiom::optimizer::PnpInitialGuess guess;
  guess.camera_model = solve_initial_model;
  guess.rvecs.reserve(n_views);
  guess.tvecs.reserve(n_views);
  for (std::size_t i = 0; i < n_views; ++i)
  {
    guess.rvecs.push_back(rotationMatrixToAngleAxis(R_views[i]));
    guess.tvecs.push_back(t_views[i]);
  }

  camxiom::optimizer::PnpSolverOptions opts;
  opts.solver_options.max_num_iterations = options.max_iterations;
  opts.solver_options.function_tolerance = options.function_tolerance;
  opts.solver_options.parameter_tolerance = options.parameter_tolerance;
  opts.huber_loss_delta = options.huber_loss_delta;

  if (options.apply_initial_value_bounds)
  {
    // D43: box-constrain focal length to the seed +/- tol.
    // NOTE: PnpBound only exposes focal_lengths / principal_points / rot
    // / trans -- there is NO per-distortion or per-projection-parameter
    // bound knob. Distortion / projection parameters are instead held by
    // `lock_flags`, which is the app's responsibility per D46. We do not
    // attempt to emulate a distortion/projection +/-tol here.
    const double r = options.bound_relative_tolerance;
    const double fx = static_cast<double>(initial_model.intrinsics.fx);
    const double fy = static_cast<double>(initial_model.intrinsics.fy);
    opts.lower_bound.focal_lengths = Eigen::Vector2d(fx * (1.0 - r), fy * (1.0 - r));
    opts.upper_bound.focal_lengths = Eigen::Vector2d(fx * (1.0 + r), fy * (1.0 + r));
  }
  else
  {
    camxiom::optimizer::detail::widenDefaultPnpUpperBounds(
      opts, options.image_width, options.image_height,
      static_cast<double>(initial_model.intrinsics.fx),
      static_cast<double>(initial_model.intrinsics.fy)
    );
  }

  if (options.apply_principal_point_bounds)
  {
    opts.lower_bound.principal_points = Eigen::Vector2d(pp_lower_x, pp_lower_y);
    opts.upper_bound.principal_points = Eigen::Vector2d(pp_upper_x, pp_upper_y);
  }
  else if (options.apply_initial_value_bounds)
  {
    // Preserve the legacy combined-bounds behaviour for callers that have not
    // opted into an explicit principal-point reference.
    const double r = options.bound_relative_tolerance;
    const double cx = static_cast<double>(solve_initial_model.intrinsics.cx);
    const double cy = static_cast<double>(solve_initial_model.intrinsics.cy);
    opts.lower_bound.principal_points = Eigen::Vector2d(cx * (1.0 - r), cy * (1.0 - r));
    opts.upper_bound.principal_points = Eigen::Vector2d(cx * (1.0 + r), cy * (1.0 + r));
  }

  // --------------------------------------------------------------------
  // Step 4: a single Ceres pass. This is the ONLY solve() call -- no
  // loop, no restart, no staging (D46).
  // --------------------------------------------------------------------
  camxiom::optimizer::PnpSolver solver;
  camxiom::optimizer::PnpResult pnp;
  solver.solve(obj_sets, img_sets, guess, pnp, opts, lock_flags);

  // solve() returns best-effort poses even on non-convergence (handled below
  // via pnp.success -> NON_CONVERGED), but rejects invalid inputs at its
  // entry checks and then leaves the result EMPTY. Guard on the pose counts
  // rather than the bool so the best-effort path stays intact while an
  // empty result can never be indexed below (out-of-bounds UB).
  if (pnp.rvecs.size() != n_views || pnp.tvecs.size() != n_views)
  {
    return failedCalibrationResult(StatusCode::INVALID_INPUT);
  }

  // --------------------------------------------------------------------
  // Step 5: assemble result + diagnostics.
  // --------------------------------------------------------------------
  CalibrationResult result;
  result.camera_model = pnp.camera_model;
  result.per_view_rotations.resize(n_views);
  result.per_view_translations.resize(n_views);
  for (std::size_t i = 0; i < n_views; ++i)
  {
    result.per_view_rotations[i] = angleAxisToRotationMatrix(pnp.rvecs[i]);
    result.per_view_translations[i] = pnp.tvecs[i];
  }

  const camxiom::optimizer::PnpSummary &summary = solver.lastSummary();
  result.iterations_used =
    static_cast<int>(summary.num_successful_steps + summary.num_unsuccessful_steps);

  result.rms_reprojection_error_px = computeReprojErrors(
    views, result.camera_model, result.per_view_rotations, result.per_view_translations,
    result.per_view_rms_px, result.max_reprojection_error_px
  );

  if (options.compute_per_point_residuals)
  {
    result.per_point_residuals = computePerPointResiduals(
      views, result.camera_model, result.per_view_rotations, result.per_view_translations
    );
  }

  if (!resultIsFinite(result.camera_model, result.per_view_rotations, result.per_view_translations))
  {
    result.status = StatusCode::NUMERIC_ERROR;
  }
  else if (!pnp.success || !summary.converged)
  {
    // Best-effort: keep the model/poses/diagnostics; the app may still
    // use them (e.g. as a warm start). Only the status flags the issue.
    //
    // pnp.success alone is not enough: on the Ceres path it is built from
    // IsSolutionUsable(), which stays true for a run that stopped at
    // max_num_iterations without meeting any tolerance (see the PnpSummary
    // doc note). OK is documented as "converged", so gate on
    // summary.converged too — the same contract convertCameraModel enforces.
    result.status = StatusCode::NON_CONVERGED;
  }
  else
  {
    result.status = StatusCode::OK;
  }

  // --------------------------------------------------------------------
  // Step 6: parameter uncertainty (C3) + observability diagnostic (C4) +
  // near-bound proximity (C5 ⑤). Only when a usable model + poses exist (OK or
  // best-effort NON_CONVERGED).
  //
  //   * Near-bound is computed FIRST and independently of the reduced normal
  //     matrix (C5 #11): a parameter parked on a box bound is reported even if
  //     the covariance cannot be formed. It reuses the SAME solver bounds
  //     (opts.lower_bound / opts.upper_bound) via the shared descriptor.
  //   * C3 (uncertainty) + C4 (observability) then share one reduced normal
  //     matrix S. Each is self-guarding: C3 leaves uncertainty_available=false
  //     when dof<=0 / rank-deficient, while C4 (which needs only S) can still
  //     report observability_available. If S itself cannot be built, both stay
  //     unavailable (fail-closed defaults).
  // --------------------------------------------------------------------
  if (options.estimate_uncertainty && (result.status == StatusCode::OK || result.status == StatusCode::NON_CONVERGED))
  {
    fillNearBounds(result.camera_model, lock_flags, opts.lower_bound, opts.upper_bound, result);
    computeCalibrationDiagnostics(
      views, result.camera_model, result.per_view_rotations, result.per_view_translations,
      lock_flags, result
    );
  }

  return result;
}

double computeReprojErrors(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t,
  std::vector<double> &per_view_rms_out, double &max_err_out
)
{
  // Defensive: calibrate() always passes consistent sizes, but guard
  // against external misuse.
  if (R.size() != views.size() || t.size() != views.size())
  {
    per_view_rms_out.clear();
    max_err_out = 0.0;
    return 0.0;
  }

  const camxiom::CameraModel64 m64 = camxiom::toCameraModel64(model);

  per_view_rms_out.assign(views.size(), 0.0);
  max_err_out = 0.0;

  double total_sum_sq = 0.0;
  std::size_t total_valid = 0;

  for (std::size_t i = 0; i < views.size(); ++i)
  {
    const auto &view = views[i];
    const Eigen::Matrix3d &Ri = R[i];
    const Eigen::Vector3d &ti = t[i];

    double view_sum_sq = 0.0;
    std::size_t view_valid = 0;

    const std::size_t n = view.world_points.size();
    for (std::size_t j = 0; j < n; ++j)
    {
      const Eigen::Vector3d P_cam = Ri * view.world_points[j] + ti;
      const camxiom::PixelResult64 pr = camxiom::rayToPixel64(m64, P_cam);
      if (pr.status != StatusCode::OK)
      {
        continue;
      }
      const Eigen::Vector2d &obs = view.image_points[j];
      const double du = pr.pixel.u - obs.x();
      const double dv = pr.pixel.v - obs.y();
      const double res_sq = du * du + dv * dv;

      view_sum_sq += res_sq;
      ++view_valid;

      const double res = std::sqrt(res_sq);
      if (res > max_err_out)
      {
        max_err_out = res;
      }
    }

    if (view_valid > 0)
    {
      per_view_rms_out[i] = std::sqrt(view_sum_sq / static_cast<double>(view_valid));
    }
    else
    {
      per_view_rms_out[i] = 0.0;
    }

    total_sum_sq += view_sum_sq;
    total_valid += view_valid;
  }

  if (total_valid == 0)
  {
    return 0.0;
  }
  return std::sqrt(total_sum_sq / static_cast<double>(total_valid));
}

std::vector<std::vector<Eigen::Vector2d>> computePerPointResiduals(
  const std::vector<CalibrationView> &views, const CameraModel &model,
  const std::vector<Eigen::Matrix3d> &R, const std::vector<Eigen::Vector3d> &t
)
{
  // Defensive: mirror computeReprojErrors' no-op on inconsistent sizes.
  if (R.size() != views.size() || t.size() != views.size())
  {
    return {};
  }

  const camxiom::CameraModel64 m64 = camxiom::toCameraModel64(model);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  std::vector<std::vector<Eigen::Vector2d>> residuals(views.size());
  for (std::size_t i = 0; i < views.size(); ++i)
  {
    const auto &view = views[i];
    const std::size_t n = view.world_points.size();
    residuals[i].assign(n, Eigen::Vector2d(nan, nan));

    for (std::size_t j = 0; j < n; ++j)
    {
      const Eigen::Vector3d P_cam = R[i] * view.world_points[j] + t[i];
      const camxiom::PixelResult64 pr = camxiom::rayToPixel64(m64, P_cam);
      if (pr.status != StatusCode::OK)
      {
        continue;  // stays NaN so indices align with the input view
      }
      const Eigen::Vector2d &obs = view.image_points[j];
      residuals[i][j] = Eigen::Vector2d(pr.pixel.u - obs.x(), pr.pixel.v - obs.y());
    }
  }
  return residuals;
}

}  // namespace camxiom::calib
