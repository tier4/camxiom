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

#include "init/seeded_planar_init.hpp"

#include "camxiom/init/dlt_pnp.hpp"
#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "init/init_detail.hpp"
#include "optimizer/pnp/pnp_parameter_bounds.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace camxiom::init::detail
{

namespace
{

/// Minimum number of views (matches MS1-2 / MS1-4 / MS1-5 / MS1-6 convention).
constexpr std::size_t kMinViews = 3U;

/// Minimum number of point correspondences per view (DLT PnP needs 4+).
constexpr Eigen::Index kMinPointsPerView = 4;

/// Model-agnostic part of the Phase B usability check (a PnpResult from
/// PnpSolver::solve); the caller-supplied predicate covers the model's
/// projection parameters.
bool phaseBLooksUsable(
  const camxiom::optimizer::PnpResult &res, const std::size_t expected_views,
  const ProjectionParamsFiniteFn projection_params_finite
)
{
  if (!res.success) return false;
  if (!std::isfinite(res.rmse)) return false;
  if (res.rvecs.size() != expected_views) return false;
  if (res.tvecs.size() != expected_views) return false;
  for (std::size_t i = 0; i < expected_views; ++i)
  {
    if (!res.rvecs[i].allFinite() || !res.tvecs[i].allFinite())
    {
      return false;
    }
  }
  const auto &cam = res.camera_model;
  if (!(std::isfinite(cam.intrinsics.fx) && std::isfinite(cam.intrinsics.fy) &&
        std::isfinite(cam.intrinsics.cx) && std::isfinite(cam.intrinsics.cy)))
  {
    return false;
  }
  if (!(cam.intrinsics.fx > 0.0f) || !(cam.intrinsics.fy > 0.0f))
  {
    return false;
  }
  return projection_params_finite(cam);
}

}  // namespace

StatusCode runSeededPlanarInit(
  const std::vector<PlanarObservation> &views, const int image_width, const int image_height,
  const MakeSeedModelFn make_seed_model, const ProjectionParamsFiniteFn projection_params_finite,
  SeededPlanarInitOutcome &out
)
{
  // ------------------------------------------------------------------
  // 1. Input validation (no out-param mutation on failure).
  // ------------------------------------------------------------------
  const std::size_t n_views = views.size();
  if (n_views < kMinViews)
  {
    return StatusCode::INVALID_INPUT;
  }
  if (image_width <= 0 || image_height <= 0)
  {
    return StatusCode::INVALID_INPUT;
  }
  for (const auto &view : views)
  {
    const Eigen::Index m_board = view.board_pts.cols();
    const Eigen::Index m_image = view.image_pts.cols();
    if (m_board < kMinPointsPerView || m_image != m_board)
    {
      return StatusCode::INVALID_INPUT;
    }
    if (!view.board_pts.allFinite() || !view.image_pts.allFinite())
    {
      return StatusCode::INVALID_INPUT;
    }
  }

  // ------------------------------------------------------------------
  // 2. Phase A (heuristic seed): default K, model-specific projection
  //    seeds baked into make_seed_model. See the header for why a
  //    Zhang-style IAC bootstrap is invalid for these models.
  // ------------------------------------------------------------------
  const double cx = 0.5 * static_cast<double>(image_width);
  const double cy = 0.5 * static_cast<double>(image_height);
  const double fx = 0.5 * static_cast<double>(image_height);
  const double fy = fx;

  Eigen::Matrix3d K_A = Eigen::Matrix3d::Identity();
  K_A(0, 0) = fx;
  K_A(1, 1) = fy;
  K_A(0, 2) = cx;
  K_A(1, 2) = cy;

  // ------------------------------------------------------------------
  // 3. Recover Phase A per-view poses with MS1-3 estimatePoseDLT, using
  //    the heuristic-seeded model so MS1-3 sees the right forward model
  //    when normalising bearings.
  // ------------------------------------------------------------------
  const camxiom::CameraModel64 phase_a_model = make_seed_model(fx, fy, cx, cy);

  std::vector<Eigen::Matrix3d> R_views;
  std::vector<Eigen::Vector3d> t_views;
  R_views.reserve(n_views);
  t_views.reserve(n_views);
  for (const auto &view : views)
  {
    const Eigen::Matrix3Xd world_3d = liftBoardToZ0(view.board_pts);
    Eigen::Matrix3d R_i = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_i = Eigen::Vector3d::Zero();
    const StatusCode pstat = estimatePoseDLT(phase_a_model, world_3d, view.image_pts, R_i, t_i);
    if (pstat != StatusCode::OK)
    {
      return StatusCode::DEGENERATE_CONFIG;
    }
    R_views.push_back(R_i);
    t_views.push_back(t_i);
  }

  // ------------------------------------------------------------------
  // 4. Phase B: best-effort joint refinement via camxiom PnpSolver.
  //    K + extrinsics free; projection parameters LOCKED at the Phase A
  //    seeds via FIX_PROJECTION_PARAMS (D29/D30 — only what is reliably
  //    identifiable from initial-guess data; projection refinement is
  //    MS2's job). Failures are silently absorbed and Phase A's result
  //    is committed (D5/D28).
  // ------------------------------------------------------------------
  Eigen::Matrix3d K_final = K_A;
  double xi_final = phase_a_model.projection.xi;
  double alpha_final = phase_a_model.projection.alpha;
  double beta_final = phase_a_model.projection.beta;
  std::vector<Eigen::Matrix3d> R_final = R_views;
  std::vector<Eigen::Vector3d> t_final = t_views;

  {
    const camxiom::CameraModel64 model64_for_pnp =
      make_seed_model(K_A(0, 0), K_A(1, 1), K_A(0, 2), K_A(1, 2));

    camxiom::CameraModel cam_for_pnp = camxiom::fromCameraModel64(model64_for_pnp);
    camxiom::updateThetaMax(cam_for_pnp);

    if (camxiom::validateCameraModel(cam_for_pnp) == camxiom::StatusCode::OK)
    {
      // Build per-view object/image point lists in PnpSolver's expected
      // shape (std::vector of std::vector of Eigen::Vector3d / Vector2d).
      camxiom::optimizer::ObjectPointSets obj_sets;
      camxiom::optimizer::ImagePointSets img_sets;
      obj_sets.reserve(n_views);
      img_sets.reserve(n_views);
      for (const auto &view : views)
      {
        const Eigen::Index m = view.board_pts.cols();
        camxiom::optimizer::ObjectPoints obj;
        camxiom::optimizer::ImagePoints img;
        obj.reserve(static_cast<std::size_t>(m));
        img.reserve(static_cast<std::size_t>(m));
        for (Eigen::Index j = 0; j < m; ++j)
        {
          obj.emplace_back(view.board_pts(0, j), view.board_pts(1, j), 0.0);
          img.emplace_back(view.image_pts(0, j), view.image_pts(1, j));
        }
        obj_sets.push_back(std::move(obj));
        img_sets.push_back(std::move(img));
      }

      camxiom::optimizer::PnpInitialGuess guess;
      guess.camera_model = cam_for_pnp;
      guess.rvecs.reserve(n_views);
      guess.tvecs.reserve(n_views);
      for (std::size_t i = 0; i < n_views; ++i)
      {
        guess.rvecs.push_back(rotationMatrixToAngleAxis(R_views[i]));
        guess.tvecs.push_back(t_views[i]);
      }

      camxiom::optimizer::PnpSolverOptions pnp_opts;
      camxiom::optimizer::detail::widenDefaultPnpUpperBounds(
        pnp_opts, image_width, image_height, static_cast<double>(cam_for_pnp.intrinsics.fx),
        static_cast<double>(cam_for_pnp.intrinsics.fy)
      );

      camxiom::optimizer::PnpSolver solver;
      camxiom::optimizer::PnpResult result;
      const camxiom::optimizer::PnpFlag pnp_flags =
        camxiom::optimizer::PnpFlag::FIX_PROJECTION_PARAMS;
      solver.solve(obj_sets, img_sets, guess, result, pnp_opts, pnp_flags);

      if (phaseBLooksUsable(result, n_views, projection_params_finite))
      {
        K_final(0, 0) = static_cast<double>(result.camera_model.intrinsics.fx);
        K_final(1, 1) = static_cast<double>(result.camera_model.intrinsics.fy);
        K_final(0, 2) = static_cast<double>(result.camera_model.intrinsics.cx);
        K_final(1, 2) = static_cast<double>(result.camera_model.intrinsics.cy);
        K_final(0, 1) = 0.0;
        K_final(1, 0) = 0.0;
        K_final(2, 0) = 0.0;
        K_final(2, 1) = 0.0;
        K_final(2, 2) = 1.0;

        xi_final = static_cast<double>(result.camera_model.projection.xi);
        alpha_final = static_cast<double>(result.camera_model.projection.alpha);
        beta_final = static_cast<double>(result.camera_model.projection.beta);

        for (std::size_t i = 0; i < n_views; ++i)
        {
          R_final[i] = angleAxisToRotationMatrix(result.rvecs[i]);
          t_final[i] = result.tvecs[i];
        }
      }
      // else: silently fall back to Phase A.
    }
  }

  // ------------------------------------------------------------------
  // 5. Final finite sweep before mutating the out parameter.
  // ------------------------------------------------------------------
  if (!K_final.allFinite() || !std::isfinite(xi_final) || !std::isfinite(alpha_final) || !std::isfinite(beta_final))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  for (std::size_t i = 0; i < n_views; ++i)
  {
    if (!R_final[i].allFinite() || !t_final[i].allFinite())
    {
      return StatusCode::NUMERIC_ERROR;
    }
  }

  out.K = K_final;
  out.xi = xi_final;
  out.alpha = alpha_final;
  out.beta = beta_final;
  out.R_per_view = std::move(R_final);
  out.t_per_view = std::move(t_final);
  return StatusCode::OK;
}

}  // namespace camxiom::init::detail
