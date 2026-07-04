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

#include "optimizer/pnp/pnp_gauss_newton.hpp"

#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"  // validateCameraModel64
#include "camxiom/projection64.hpp"
#include "jacobian/full_jacobian64_internal.hpp"
#include "optimizer/pnp/angle_axis_jacobian.hpp"

#include <Eigen/Dense>

#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>

namespace camxiom::optimizer
{
namespace
{

using namespace camxiom;
// Rodrigues derivative d(R(omega)*p)/d(omega): shared single source (see
// optimizer/pnp/angle_axis_jacobian.hpp).
using camxiom::optimizer::detail::angleAxisPointJacobian;

constexpr double kInvalidPenaltyScale = 1e3;

// Huber weight: w(r) such that ρ'(r²) ≈ w * r²
//   If |r| <= δ:  w = 1
//   If |r| >  δ:  w = δ / |r|
inline double huberWeight(double residual_norm, double delta)
{
  if (delta <= 0.0) return 1.0;
  return (residual_norm <= delta) ? 1.0 : delta / residual_norm;
}

// True Huber objective ρ(|r|). This — not the IRLS surrogate w * |r|² —
// is what the huberWeight-ed normal equations descend (∇Σρ = -2 JᵀWr), so
// it is the only cost against which a trial step may be accepted or
// rejected: the surrogate's gradient disagrees with JᵀWr near the optimum
// (factor 2 on inliers vs 1 on outliers), which would deadlock the damping
// loop by rejecting genuinely downhill steps.
inline double huberCost(double residual_norm, double delta)
{
  if (delta <= 0.0 || residual_norm <= delta)
  {
    return residual_norm * residual_norm;
  }
  return 2.0 * delta * residual_norm - delta * delta;
}

}  // namespace

GaussNewtonViewResult solveViewGaussNewton(
  const CameraModel64 &model64, const std::vector<Eigen::Vector3d> &object_points,
  const std::vector<Eigen::Vector2d> &image_points, Eigen::Vector3d &rvec, Eigen::Vector3d &tvec,
  const GaussNewtonOptions &opts
)
{
  const int N = static_cast<int>(object_points.size());
  GaussNewtonViewResult result{};

  if (N == 0)
  {
    result.converged = false;
    return result;
  }

  // The model is fixed for the whole GN solve (this path requires
  // FIX_INTRINSICS): validate ONCE here instead of once per point per
  // iteration inside rayToPixelWithFullJacobian64. On an invalid model every
  // point takes the same gradient-bearing penalty branch as before (it only
  // reads p_cam / rvec / p_obj, never fj's fields).
  const bool model_valid = camxiom::validateCameraModel64(model64) == StatusCode::OK;

  // Single evaluation pass: cost of (rv, tv) plus the normal equations
  // JᵀWJ δ = JᵀW r. Shared by the initial linearisation and every trial
  // step so the residual / penalty / weight definition cannot drift apart.
  const auto evaluate = [&](
                          const Eigen::Vector3d &rv, const Eigen::Vector3d &tv,
                          Eigen::Matrix<double, 6, 6> &JtWJ, Eigen::Matrix<double, 6, 1> &JtWr
                        ) -> double {
    JtWJ.setZero();
    JtWr.setZero();

    Eigen::Matrix3d R;
    {
      double aa[3] = {rv.x(), rv.y(), rv.z()};
      double rm[9];
      ceres::AngleAxisToRotationMatrix(aa, rm);
      R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rm);
    }

    double total_cost = 0.0;

    for (int j = 0; j < N; ++j)
    {
      const Eigen::Vector3d &p_obj = object_points[static_cast<std::size_t>(j)];
      const Eigen::Vector2d &obs = image_points[static_cast<std::size_t>(j)];

      const Eigen::Vector3d p_cam = R * p_obj + tv;

      const FullProjectionJacobian64 fj =
        model_valid ? camxiom::detail::rayToPixelWithFullJacobian64Unchecked(model64, p_cam)
                    : FullProjectionJacobian64{};  // default status != OK -> penalty branch

      if (fj.status != StatusCode::OK)
      {
        const double eps_z = 0.1;
        if (p_cam.z() >= eps_z)
        {
          // In front of the camera but unprojectable (theta_max, fold-over,
          // out-of-domain model): the z-pulling gradient below would push
          // the point *further* out of the FOV, so contribute a constant
          // penalty with zero gradient instead — the same semantics as the
          // AUTO_DIFF cost. The surrounding valid points then drive the
          // pose back into the FOV.
          const double pen = kInvalidPenaltyScale * eps_z;
          total_cost += huberCost(Eigen::Vector2d(pen, pen).norm(), opts.huber_delta);
          continue;
        }

        // Behind (or grazing) the camera: gradient-bearing penalty that
        // pulls the point back in front:
        // penalty = kScale * (eps - z_cam)  for both residual components
        // ∂penalty/∂z_cam = -kScale
        // ∂z_cam/∂rvec = J_aa row 2,  ∂z_cam/∂tvec = [0,0,1]
        const double pen = kInvalidPenaltyScale * (eps_z - p_cam.z());
        const Eigen::Vector2d r_pen(pen, pen);
        const double r_pen_norm = r_pen.norm();
        const double w_pen = huberWeight(r_pen_norm, opts.huber_delta);

        const Eigen::Matrix3d J_aa = angleAxisPointJacobian(rv, p_obj);
        Eigen::Matrix<double, 2, 6> J_pen;
        // ∂proj/∂rvec = kScale * J_aa.row(2), ∂proj/∂tvec = kScale * [0,0,1]
        J_pen.row(0).head<3>() = kInvalidPenaltyScale * J_aa.row(2);
        J_pen.row(0).tail<3>() << 0.0, 0.0, kInvalidPenaltyScale;
        J_pen.row(1) = J_pen.row(0);

        JtWJ.noalias() += w_pen * (J_pen.transpose() * J_pen);
        JtWr.noalias() += w_pen * (J_pen.transpose() * r_pen);
        total_cost += huberCost(r_pen_norm, opts.huber_delta);
        continue;
      }

      // Residual: obs - proj
      const Eigen::Vector2d r(obs.x() - fj.pixel.u, obs.y() - fj.pixel.v);
      const double r_norm = r.norm();

      // Huber weight
      const double w = huberWeight(r_norm, opts.huber_delta);

      // Jacobian ∂(u,v)/∂(rvec, tvec) — 2×6
      // ∂(u,v)/∂rvec = Jp * J_aa   (2×3)
      // ∂(u,v)/∂tvec = Jp           (2×3)
      const Eigen::Matrix3d J_aa = angleAxisPointJacobian(rv, p_obj);
      const auto &Jp = fj.J_point;  // 2×3

      Eigen::Matrix<double, 2, 6> J;
      J.leftCols<3>() = Jp * J_aa;
      J.rightCols<3>() = Jp;

      // ∂residual/∂params = -J, but normal eq uses J on projection side.
      // r = obs - proj(params)
      // We want to solve:  Σ Jᵢᵀ W Jᵢ δ = Σ Jᵢᵀ W rᵢ
      //   where Jᵢ = ∂proj/∂params, δ is the update, rᵢ = obs - proj
      JtWJ.noalias() += w * (J.transpose() * J);
      JtWr.noalias() += w * (J.transpose() * r);

      total_cost += huberCost(r_norm, opts.huber_delta);
    }

    return total_cost;
  };

  Eigen::Matrix<double, 6, 6> JtWJ;
  Eigen::Matrix<double, 6, 1> JtWr;
  double current_cost = evaluate(rvec, tvec, JtWJ, JtWr);
  // final_cost always reflects the parameters actually returned: the old
  // loop applied the last step blindly and reported the cost of the
  // *previous* iterate.
  result.final_cost = current_cost;

  // Levenberg-style damping. The old fixed 1e-12 accepted every step
  // unconditionally, so a near-singular JtWJ (thin board, near-collinear
  // points) could throw the pose to a much worse point with no way back —
  // and the tolerance checks could still declare convergence there.
  double damping = 1e-12;
  constexpr double kMaxDamping = 1e8;

  for (int iter = 0; iter < opts.max_iterations; ++iter)
  {
    result.iterations = iter + 1;

    // Check gradient convergence
    if (JtWr.lpNorm<Eigen::Infinity>() < opts.gradient_tolerance)
    {
      result.converged = true;
      break;
    }

    // Try steps with growing damping until one does not increase the cost.
    bool accepted = false;
    bool stop = false;
    while (damping <= kMaxDamping)
    {
      Eigen::Matrix<double, 6, 6> H = JtWJ;
      H.diagonal().array() += damping;
      const Eigen::Matrix<double, 6, 1> delta = H.llt().solve(JtWr);

      if (!delta.allFinite())
      {
        damping = std::max(damping * 10.0, 1e-8);
        continue;
      }

      // Check parameter convergence
      if (delta.norm() < opts.parameter_tolerance * (rvec.norm() + tvec.norm() + opts.parameter_tolerance))
      {
        result.converged = true;
        stop = true;
        break;
      }

      const Eigen::Vector3d trial_rvec = rvec + delta.head<3>();
      const Eigen::Vector3d trial_tvec = tvec + delta.tail<3>();
      Eigen::Matrix<double, 6, 6> JtWJ_trial;
      Eigen::Matrix<double, 6, 1> JtWr_trial;
      const double trial_cost = evaluate(trial_rvec, trial_tvec, JtWJ_trial, JtWr_trial);

      if (std::isfinite(trial_cost) && trial_cost <= current_cost)
      {
        rvec = trial_rvec;
        tvec = trial_tvec;
        JtWJ = JtWJ_trial;
        JtWr = JtWr_trial;
        const double cost_change = current_cost - trial_cost;
        current_cost = trial_cost;
        result.final_cost = current_cost;
        damping = std::max(damping * 0.1, 1e-12);
        accepted = true;

        // Check function convergence on an actually accepted step
        if (cost_change < opts.function_tolerance * current_cost)
        {
          result.converged = true;
          stop = true;
        }
        break;
      }
      damping = std::max(damping * 10.0, 1e-8);
    }

    if (stop)
    {
      break;
    }
    if (!accepted)
    {
      // No damping level decreased the cost: stationary point or genuine
      // divergence. Keep the best-known parameters; converged stays false.
      break;
    }
  }

  return result;
}

}  // namespace camxiom::optimizer
