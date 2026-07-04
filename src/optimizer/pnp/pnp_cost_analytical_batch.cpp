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

#include "optimizer/pnp/pnp_cost_analytical_batch.hpp"

#include "camxiom/jacobian_with_distortion_deriv64.hpp"
#include "camxiom/model.hpp"  // validateCameraModel64
#include "camxiom/projection64.hpp"
#include "camxiom/types64.hpp"
#include "jacobian/full_jacobian64_internal.hpp"
#include "model/distortion_aux.hpp"
#include "model/internal.hpp"  // shrinkThetaMaxToPolynomialMonotoneRange
#include "optimizer/pnp/angle_axis_jacobian.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>

namespace
{

constexpr double kInvalidPenaltyScale = 1e3;

// Copy updated coefficients into the model, then rebuild the derived
// distortion state via the shared single-source helper (same thresholds as
// the compat factory and the PnP solver write-back).
inline void rebuildDistortionFlags64(
  camxiom::DistortionModel64 &dist, const double *new_coeffs, int dc
)
{
  for (int i = 0; i < dc; ++i) dist.coeffs[static_cast<std::size_t>(i)] = new_coeffs[i];

  camxiom::detail::rebuildDistortionAuxState(dist);
}

}  // anonymous namespace

namespace camxiom::optimizer
{
namespace
{

using namespace camxiom;
// Rodrigues derivative d(R(omega)*p)/d(omega): shared single source (see
// optimizer/pnp/angle_axis_jacobian.hpp).
using camxiom::optimizer::detail::angleAxisPointJacobian;

// -----------------------------------------------------------------------
// Batch Analytical CostFunction — one per view, 2*N residuals
// -----------------------------------------------------------------------
class PnpCostAnalyticalBatch : public ceres::CostFunction
{
public:
  PnpCostAnalyticalBatch(
    const std::vector<Eigen::Vector3d> &object_points,
    const std::vector<Eigen::Vector2d> &image_points, const CameraModel64 &model_template,
    int effective_dist_count, int actual_dist_count
  )
  : object_points_(object_points),
    image_points_(image_points),
    model_template_(model_template),
    actual_dist_count_(actual_dist_count),
    num_points_(static_cast<int>(object_points.size()))
  {
    set_num_residuals(2 * num_points_);
    auto *sizes = mutable_parameter_block_sizes();
    sizes->push_back(3);                     // rvec
    sizes->push_back(3);                     // tvec
    sizes->push_back(2);                     // focal_lengths
    sizes->push_back(2);                     // principal_points
    sizes->push_back(effective_dist_count);  // dist_coeffs
    sizes->push_back(3);                     // projection_params [xi, alpha, beta]
  }

  bool Evaluate(const double *const *parameters, double *residuals, double **jacobians)
    const override
  {
    const double *rvec = parameters[0];
    const double *tvec = parameters[1];
    const double *focal = parameters[2];
    const double *pp = parameters[3];
    const double *dist = parameters[4];
    const double *proj = parameters[5];

    const int edc = parameter_block_sizes()[4];
    const int N = num_points_;
    const int dc = actual_dist_count_;

    // Build CameraModel64 once per view, with full auxiliary state rebuild
    CameraModel64 model = model_template_;
    model.intrinsics.fx = focal[0];
    model.intrinsics.fy = focal[1];
    model.intrinsics.cx = pp[0];
    model.intrinsics.cy = pp[1];
    rebuildDistortionFlags64(model.distortion, dist, dc);
    model.projection.xi = proj[0];
    model.projection.alpha = proj[1];
    model.projection.beta = proj[2];

    // Precompute rotation matrix once for the entire view
    const Eigen::Vector3d omega(rvec[0], rvec[1], rvec[2]);
    Eigen::Matrix3d R;
    {
      double angle_axis[3] = {rvec[0], rvec[1], rvec[2]};
      double rot_mat[9];
      ceres::AngleAxisToRotationMatrix(angle_axis, rot_mat);
      // Ceres stores column-major
      R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::ColMajor>>(rot_mat);
    }
    const Eigen::Vector3d t(tvec[0], tvec[1], tvec[2]);

    const double fx = focal[0];
    const double fy = focal[1];
    const double sk = model.intrinsics.skew;

    // Zero Jacobians if requested
    if (jacobians)
    {
      if (jacobians[0]) std::fill_n(jacobians[0], 2 * N * 3, 0.0);
      if (jacobians[1]) std::fill_n(jacobians[1], 2 * N * 3, 0.0);
      if (jacobians[2]) std::fill_n(jacobians[2], 2 * N * 2, 0.0);
      if (jacobians[3]) std::fill_n(jacobians[3], 2 * N * 2, 0.0);
      if (jacobians[4]) std::fill_n(jacobians[4], 2 * N * edc, 0.0);
      if (jacobians[5]) std::fill_n(jacobians[5], 2 * N * 3, 0.0);
    }

    // The model is fixed for the whole Evaluate call: validate it ONCE here
    // instead of once per point inside rayToPixelWithFullJacobian64. On an
    // invalid model (parameters drifted out of domain mid-optimisation) every
    // point takes the same gradient-bearing penalty branch as before (the
    // branch only reads p_cam / omega / p_obj, never fj's fields).
    bool model_valid_now = camxiom::validateCameraModel64(model) == StatusCode::OK;
    if (!model_valid_now && camxiom::detail::shrinkThetaMaxToPolynomialMonotoneRange(model))
    {
      // Free polynomial-fisheye coefficients move the theta_d fold mid-solve
      // while theta_max stays frozen at the seed's cap. Without this shrink,
      // the validator's endpoint check turns EVERY residual of any
      // coefficient step whose polynomial folds below that stale cap into
      // the all-points penalty above — a wall in the cost landscape that
      // pins wide-FOV fits (real >=180-deg lenses routinely fit folding
      // polynomials) at theta_d(cap) == 0 instead of the optimum. Shrinking
      // the derived cap keeps the model self-consistent; points past the
      // shrunk cap still take the per-point penalty branch below, which is
      // the correct cost shape. theta_max is not a Ceres parameter, so the
      // shrink is local to this Evaluate; the FINAL model's theta_max is
      // (re)derived independently by writeBack() via updateThetaMax().
      model_valid_now = camxiom::validateCameraModel64(model) == StatusCode::OK;
    }

    for (int j = 0; j < N; ++j)
    {
      const Eigen::Vector3d &p_obj = object_points_[static_cast<std::size_t>(j)];
      const Eigen::Vector2d &obs = image_points_[static_cast<std::size_t>(j)];

      // P_cam = R * P_obj + t
      const Eigen::Vector3d p_cam = R * p_obj + t;

      // Forward projection with full Jacobian (unchecked: validated above)
      const FullProjectionJacobian64 fj =
        model_valid_now ? camxiom::detail::rayToPixelWithFullJacobian64Unchecked(model, p_cam)
                        : FullProjectionJacobian64{};  // default status != OK -> penalty branch

      const int r0 = 2 * j;
      const int r1 = 2 * j + 1;

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
          residuals[r0] = kInvalidPenaltyScale * eps_z;
          residuals[r1] = kInvalidPenaltyScale * eps_z;
          // Jacobians stay zeroed.
          continue;
        }

        // Behind (or grazing) the camera: gradient-bearing penalty that
        // pulls the point back in front via z_cam.
        residuals[r0] = kInvalidPenaltyScale * (eps_z - p_cam.z());
        residuals[r1] = kInvalidPenaltyScale * (eps_z - p_cam.z());
        if (jacobians)
        {
          // ∂penalty/∂z_cam = -kInvalidPenaltyScale
          // ∂z_cam/∂rvec = J_aa row 2,  ∂z_cam/∂tvec = [0,0,1]
          const Eigen::Matrix3d J_aa = angleAxisPointJacobian(omega, p_obj);
          if (jacobians[0])
          {
            for (int k = 0; k < 3; ++k)
            {
              jacobians[0][r0 * 3 + k] = -kInvalidPenaltyScale * J_aa(2, k);
              jacobians[0][r1 * 3 + k] = -kInvalidPenaltyScale * J_aa(2, k);
            }
          }
          if (jacobians[1])
          {
            jacobians[1][r0 * 3 + 0] = 0.0;
            jacobians[1][r0 * 3 + 1] = 0.0;
            jacobians[1][r0 * 3 + 2] = -kInvalidPenaltyScale;
            jacobians[1][r1 * 3 + 0] = 0.0;
            jacobians[1][r1 * 3 + 1] = 0.0;
            jacobians[1][r1 * 3 + 2] = -kInvalidPenaltyScale;
          }
          // Other Jacobians remain zero (already zeroed above)
        }
        continue;
      }

      residuals[r0] = obs.x() - fj.pixel.u;
      residuals[r1] = obs.y() - fj.pixel.v;

      if (!jacobians) continue;

      const auto &Jp = fj.J_point;  // 2×3

      // ∂residual/∂rvec  (row-major: [r0*3 .. r0*3+2], [r1*3 .. r1*3+2])
      if (jacobians[0])
      {
        const Eigen::Matrix3d J_aa = angleAxisPointJacobian(omega, p_obj);
        const Eigen::Matrix<double, 2, 3> J_rv = -(Jp * J_aa);
        // Row-major layout: row r0 at offset r0*3, row r1 at offset r1*3
        jacobians[0][r0 * 3 + 0] = J_rv(0, 0);
        jacobians[0][r0 * 3 + 1] = J_rv(0, 1);
        jacobians[0][r0 * 3 + 2] = J_rv(0, 2);
        jacobians[0][r1 * 3 + 0] = J_rv(1, 0);
        jacobians[0][r1 * 3 + 1] = J_rv(1, 1);
        jacobians[0][r1 * 3 + 2] = J_rv(1, 2);
      }

      // ∂residual/∂tvec
      if (jacobians[1])
      {
        jacobians[1][r0 * 3 + 0] = -Jp(0, 0);
        jacobians[1][r0 * 3 + 1] = -Jp(0, 1);
        jacobians[1][r0 * 3 + 2] = -Jp(0, 2);
        jacobians[1][r1 * 3 + 0] = -Jp(1, 0);
        jacobians[1][r1 * 3 + 1] = -Jp(1, 1);
        jacobians[1][r1 * 3 + 2] = -Jp(1, 2);
      }

      // ∂residual/∂focal_lengths
      if (jacobians[2])
      {
        jacobians[2][r0 * 2 + 0] = -fj.xd;
        jacobians[2][r0 * 2 + 1] = 0.0;
        jacobians[2][r1 * 2 + 0] = 0.0;
        jacobians[2][r1 * 2 + 1] = -fj.yd;
      }

      // ∂residual/∂principal_points
      if (jacobians[3])
      {
        jacobians[3][r0 * 2 + 0] = -1.0;
        jacobians[3][r0 * 2 + 1] = 0.0;
        jacobians[3][r1 * 2 + 0] = 0.0;
        jacobians[3][r1 * 2 + 1] = -1.0;
      }

      // ∂residual/∂dist_coeffs
      if (jacobians[4])
      {
        for (int k = 0; k < edc; ++k)
        {
          if (k < dc)
          {
            const double dxd = fj.dxd_ddist[static_cast<std::size_t>(k)];
            const double dyd = fj.dyd_ddist[static_cast<std::size_t>(k)];
            jacobians[4][r0 * edc + k] = -(fx * dxd + sk * dyd);
            jacobians[4][r1 * edc + k] = -(fy * dyd);
          }
        }
      }

      // ∂residual/∂projection_params
      if (jacobians[5])
      {
        for (int k = 0; k < 3; ++k)
        {
          const double dxd = fj.dxd_dproj[static_cast<std::size_t>(k)];
          const double dyd = fj.dyd_dproj[static_cast<std::size_t>(k)];
          jacobians[5][r0 * 3 + k] = -(fx * dxd + sk * dyd);
          jacobians[5][r1 * 3 + k] = -(fy * dyd);
        }
      }
    }

    return true;
  }

private:
  std::vector<Eigen::Vector3d> object_points_;
  std::vector<Eigen::Vector2d> image_points_;
  CameraModel64 model_template_;
  int actual_dist_count_;
  int num_points_;
};

}  // namespace

ceres::CostFunction *createAnalyticalBatchCost(
  const std::vector<Eigen::Vector3d> &object_points,
  const std::vector<Eigen::Vector2d> &image_points, const camxiom::CameraModel &camera_model,
  int effective_dist_count
)
{
  const CameraModel64 model64 = toCameraModel64(camera_model);
  const int actual_dc =
    std::min(static_cast<int>(camera_model.distortion.count), effective_dist_count);
  return new PnpCostAnalyticalBatch(
    object_points, image_points, model64, effective_dist_count, actual_dc
  );
}

}  // namespace camxiom::optimizer
