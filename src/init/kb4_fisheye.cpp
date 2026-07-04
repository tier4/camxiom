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

#include "camxiom/init/kb4_fisheye.hpp"

#include "camxiom/init/dlt_pnp.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/optimizer/pnp_solver.hpp"
#include "camxiom/types64.hpp"
#include "init/init_detail.hpp"
#include "model/internal.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cmath>
#include <cstddef>
#include <vector>

namespace camxiom::init
{

namespace
{

/// Minimum number of views required to estimate KB4 reliably.
///
/// KB4 has 8 unknowns total (4 intrinsics + 4 distortion). Each view adds
/// 2*M residuals at the cost of 6 unknown extrinsic parameters; with M >= 4
/// per view the per-view residual surplus is >= 2. With three views we get
/// 6 surplus residuals which is >= 4 (the distortion DOF) — sufficient to
/// identify the four KB4 coefficients in the linear LSQ step.
constexpr std::size_t kMinViews = 3U;

/// Minimum number of correspondences per view (matches MS1-2 + MS1-3).
constexpr Eigen::Index kMinPointsPerView = 4;

/// Degeneracy threshold for the KB4 linear LSQ.
///
/// A = M_total x 3 (joint fx + k1 + k2 estimation; k3 and k4 are deferred
/// to Phase B). We treat the system as rank-deficient when sigma(2) /
/// sigma(0) < 1e-12. Mirrors the pattern from MS1-1 / 1-2 / 1-3.
constexpr double kKB4DegenerateSingularRatio = 1e-12;

/// Seed incidence angle assumed at the image diagonal corner (radians).
///
/// The Phase A iteration-0 seed is a zero-distortion equidistant model, so a
/// pixel at radius r lifts to theta = r / fx_seed. Every observed corner
/// must lift successfully — one OUT_OF_FOV pixel aborts the whole init — so
/// fx_seed is chosen to place the farthest possible pixel, the image
/// diagonal corner, at ~172 deg: inside the theta_max = pi model limit with
/// margin, for any aspect ratio. (The previous seed, image_height / pi with
/// theta_max = pi/2, rejected every corner farther than height/2 from the
/// principal point, so boards placed in the image periphery — exactly what
/// good fisheye calibration coverage requires — aborted the init.)
constexpr double kSeedDiagonalTheta = 3.0;

/// Maximum number of Phase A bootstrap iterations.
///
/// Phase A suffers from a chicken-and-egg coupling: the seed focal length
/// (half-diagonal / kSeedDiagonalTheta, equidistant assumption) is
/// typically biased well below the true fx, which biases the per-view DLT
/// poses, which biases
/// the theta values fed into the linear LSQ, which then biases the
/// recovered fx. Iterating Phase A — re-running MS1-3 with each refined
/// (fx, k1, k2) and then re-solving the LSQ — breaks the coupling and
/// converges fx within 2-3 iterations on typical fisheye geometry. The
/// cap of 5 is a hard ceiling; if convergence has not been reached we
/// take the last estimate and let Phase B finish the job.
constexpr int kMaxBootstrapIters = 5;

/// Relative convergence tolerance on fx between successive Phase A
/// iterations: |fx_new - fx_old| / fx_old < 1e-3.
constexpr double kFxConvergenceTol = 1e-3;

/// Build the camxiom CameraModel64 seed used during Phase A pose recovery.
///
/// Bootstrap iteration 0 uses EQUIDISTANT distortion (k_i = 0 implicitly,
/// theta_d = theta) since we have no KB4 estimate yet. Subsequent
/// iterations use KB4 with the current (k1, k2, 0, 0) Phase A estimate so
/// the MS1-3 PnP sees a more accurate forward model and returns better
/// poses, which in turn feed back into the LSQ.
camxiom::CameraModel64 makeSeedModel(
  double fx, double fy, double cx, double cy, bool use_kb4, const Eigen::Vector4d &kb4_coeffs
)
{
  camxiom::CameraModel64 model{};
  model.intrinsics.fx = fx;
  model.intrinsics.fy = fy;
  model.intrinsics.cx = cx;
  model.intrinsics.cy = cy;
  model.intrinsics.skew = 0.0;

  model.projection.type = camxiom::ProjectionModelType::FISHEYE_THETA;

  model.distortion.space = camxiom::DistortionSpace::ANGLE;
  model.distortion.coeffs.fill(0.0);
  model.distortion.is_rational = false;
  model.distortion.has_thin_prism = false;
  model.distortion.has_tilt = false;
  model.distortion.tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  model.distortion.inv_tilt_matrix = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  if (use_kb4)
  {
    model.distortion.type = camxiom::DistortionModelType::KB4;
    for (int i = 0; i < 4; ++i)
    {
      model.distortion.coeffs[static_cast<std::size_t>(i)] = kb4_coeffs(i);
    }
    model.distortion.count = 4U;
  }
  else
  {
    model.distortion.type = camxiom::DistortionModelType::EQUIDISTANT;
    model.distortion.count = 0U;
  }

  // theta_max doubles as the bracket top of the inverse theta solve inside
  // pixelToRay64, so it must never exceed the distortion polynomial's
  // monotone range: mid-bootstrap (k1, k2) estimates can fold the KB4
  // polynomial over well below pi (theta_d(pi) can even go negative), which
  // would reject every pixel as OUT_OF_FOV and abort the init. Reuse the
  // same monotone-range rule as camxiom::updateThetaMax. (A pi/2 cap here
  // would instead reject peripheral / >180-deg-FOV observations outright.)
  {
    camxiom::DistortionModel dist32{};
    dist32.type = model.distortion.type;
    dist32.space = camxiom::DistortionSpace::ANGLE;
    dist32.count = model.distortion.count;
    dist32.coeffs.fill(0.0f);
    for (int i = 0; i < 4; ++i)
    {
      dist32.coeffs[static_cast<std::size_t>(i)] =
        static_cast<float>(model.distortion.coeffs[static_cast<std::size_t>(i)]);
    }
    model.projection.theta_max =
      static_cast<double>(camxiom::detail::defaultFisheyeThetaMax(dist32));
  }

  return model;
}

// liftBoardToZ0 / rotationMatrixToAngleAxis / angleAxisToRotationMatrix are
// shared with the other init estimators via init/init_detail.hpp.
using detail::liftBoardToZ0;

/// Run the 3-DOF Phase A linear least squares for (fx, k1, k2) given a
/// per-view set of MS1-3 poses.
///
/// Builds the M_total x 3 design matrix
///     A_row = [ theta_true,  theta_true^3,  theta_true^5 ]
///     b_row = sqrt((u - cx)^2 + (v - cy)^2)
/// solves it via BDCSVD with Hartley-style column scaling, and recovers
/// fx = a(0), k1 = a(1)/fx, k2 = a(2)/fx. Returns INVALID_INPUT /
/// DEGENERATE_CONFIG / NUMERIC_ERROR consistent with the public API.
///
/// On OK, fx_out / k1_out / k2_out are populated; on non-OK they are left
/// untouched.
StatusCode solvePhaseALsq(
  const std::vector<PlanarObservation> &views, const std::vector<Eigen::Matrix3d> &R_per_view,
  const std::vector<Eigen::Vector3d> &t_per_view, double cx, double cy, double &fx_out,
  double &k1_out, double &k2_out
)
{
  const std::size_t n_views = views.size();

  Eigen::Index m_total = 0;
  for (const auto &view : views)
  {
    m_total += view.board_pts.cols();
  }

  Eigen::MatrixXd A(m_total, 3);
  Eigen::VectorXd b(m_total);

  Eigen::Index row = 0;
  for (std::size_t v = 0; v < n_views; ++v)
  {
    const auto &view = views[v];
    const Eigen::Matrix3d &R_v = R_per_view[v];
    const Eigen::Vector3d &t_v = t_per_view[v];
    const Eigen::Index m = view.board_pts.cols();

    for (Eigen::Index j = 0; j < m; ++j)
    {
      const Eigen::Vector3d world_pt(view.board_pts(0, j), view.board_pts(1, j), 0.0);
      const Eigen::Vector3d p_cam = R_v * world_pt + t_v;
      const double xy_norm = std::hypot(p_cam.x(), p_cam.y());
      const double theta_true = std::atan2(xy_norm, p_cam.z());

      const double du = view.image_pts(0, j) - cx;
      const double dv = view.image_pts(1, j) - cy;
      const double r_pixel_obs = std::hypot(du, dv);

      const double t2 = theta_true * theta_true;
      const double t3 = theta_true * t2;
      const double t5 = t3 * t2;

      A(row, 0) = theta_true;
      A(row, 1) = t3;
      A(row, 2) = t5;
      b(row) = r_pixel_obs;
      ++row;
    }
  }

  if (!A.allFinite() || !b.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  Eigen::Vector3d col_scale;
  for (int j = 0; j < 3; ++j)
  {
    col_scale(j) = A.col(j).norm();
    if (!(col_scale(j) > 0.0) || !std::isfinite(col_scale(j)))
    {
      return StatusCode::DEGENERATE_CONFIG;
    }
    A.col(j) /= col_scale(j);
  }

  Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd &sigma = svd.singularValues();
  if (sigma.size() < 3)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  const double sigma_largest = sigma(0);
  if (!(sigma_largest > 0.0) || !std::isfinite(sigma_largest))
  {
    return StatusCode::NUMERIC_ERROR;
  }
  if (sigma(2) / sigma_largest < kKB4DegenerateSingularRatio)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  const Eigen::Vector3d a_scaled = svd.solve(b);
  if (!a_scaled.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  Eigen::Vector3d a_sol;
  for (int j = 0; j < 3; ++j)
  {
    a_sol(j) = a_scaled(j) / col_scale(j);
  }
  if (!a_sol.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  const double fx_recovered = a_sol(0);
  if (!std::isfinite(fx_recovered) || !(fx_recovered > 0.0))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  fx_out = fx_recovered;
  k1_out = a_sol(1) / fx_recovered;
  k2_out = a_sol(2) / fx_recovered;
  return StatusCode::OK;
}

/// Validate the result of Phase B (a PnpResult from PnpSolver::solve).
///
/// Phase B is best-effort: if the solver returns failure or yields non-
/// finite output we discard its result and keep Phase A. The caller is
/// expected to gate on this predicate before mutating result_out.
bool phaseBLooksUsable(const camxiom::optimizer::PnpResult &res, std::size_t expected_views)
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
  for (int i = 0; i < 4; ++i)
  {
    if (!std::isfinite(cam.distortion.coeffs[static_cast<std::size_t>(i)]))
    {
      return false;
    }
  }
  return true;
}

using detail::angleAxisToRotationMatrix;
using detail::rotationMatrixToAngleAxis;

}  // namespace

StatusCode estimateKB4Init(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  KB4InitResult &result_out
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
  // 2. Seed: cx = W/2, cy = H/2, fx = fy = half_diagonal /
  //    kSeedDiagonalTheta (equidistant assumption: r_pixel = f * theta
  //    with the diagonal corner at ~172 deg). This guarantees every
  //    pixel of the image lifts to theta < pi under the iteration-0
  //    seed regardless of aspect ratio. The seed focal length is
  //    typically biased well below truth, which is what motivates the
  //    bootstrap loop below.
  // ------------------------------------------------------------------
  const double cx = 0.5 * static_cast<double>(image_width);
  const double cy = 0.5 * static_cast<double>(image_height);
  double fx = std::hypot(cx, cy) / kSeedDiagonalTheta;

  Eigen::Vector4d D_A = Eigen::Vector4d::Zero();
  std::vector<Eigen::Matrix3d> R_views;
  std::vector<Eigen::Vector3d> t_views;
  R_views.reserve(n_views);
  t_views.reserve(n_views);

  // ------------------------------------------------------------------
  // 3. Phase A bootstrap loop.
  //
  //    The seed fx is biased, which biases the MS1-3 poses, which
  //    biases the theta values fed into the LSQ, which then biases the
  //    recovered fx. Iterating MS1-3 + LSQ with each refined fx breaks
  //    the chicken-and-egg coupling and converges fx within 2-3
  //    iterations under typical fisheye geometry. We cap at 5
  //    iterations; if convergence has not been reached we take the
  //    last estimate and let Phase B finish polishing.
  //
  //    Iteration 0: EQUIDISTANT seed (no KB4 estimate yet).
  //    Iteration >=1: KB4 seed with current (k1, k2, 0, 0).
  //
  //    On any failure inside the loop we return immediately; we never
  //    keep partial results from an aborted iteration.
  // ------------------------------------------------------------------
  for (int iter = 0; iter < kMaxBootstrapIters; ++iter)
  {
    // 3a. Build the seed model for this iteration.
    const bool use_kb4 = (iter > 0);
    const camxiom::CameraModel64 iter_seed = makeSeedModel(fx, fx, cx, cy, use_kb4, D_A);

    // 3b. Re-run MS1-3 for every view with the current seed model.
    R_views.clear();
    t_views.clear();
    for (const auto &view : views)
    {
      const Eigen::Matrix3Xd world_3d = liftBoardToZ0(view.board_pts);
      Eigen::Matrix3d R_i = Eigen::Matrix3d::Identity();
      Eigen::Vector3d t_i = Eigen::Vector3d::Zero();
      const StatusCode status = estimatePoseDLT(iter_seed, world_3d, view.image_pts, R_i, t_i);
      if (status != StatusCode::OK)
      {
        return StatusCode::DEGENERATE_CONFIG;
      }
      R_views.push_back(R_i);
      t_views.push_back(t_i);
    }

    // 3c. 3-DOF LSQ on (fx, k1, k2) using the freshly recovered poses.
    double fx_new = 0.0;
    double k1_new = 0.0;
    double k2_new = 0.0;
    const StatusCode lsq_status =
      solvePhaseALsq(views, R_views, t_views, cx, cy, fx_new, k1_new, k2_new);
    if (lsq_status != StatusCode::OK)
    {
      return lsq_status;
    }

    // 3d. Convergence check: relative change in fx < tol.
    //
    //    Skip the check on iteration 0: the EQUIDISTANT seed projects
    //    r = fx_seed * theta, and the iter-0 LSQ inverts the same
    //    relation on the resulting poses — so fx_new ~ fx_seed by
    //    construction even though both are biased. We must let at
    //    least one iteration with the KB4 seed feed back into MS1-3
    //    before we can claim convergence.
    const double rel_change = std::abs(fx_new - fx) / fx;
    fx = fx_new;
    D_A(0) = k1_new;
    D_A(1) = k2_new;
    D_A(2) = 0.0;
    D_A(3) = 0.0;

    if (iter > 0 && rel_change < kFxConvergenceTol)
    {
      break;
    }
  }

  // ------------------------------------------------------------------
  // 4. Assemble the Phase A K_A from the converged fx.
  // ------------------------------------------------------------------
  Eigen::Matrix3d K_A = Eigen::Matrix3d::Identity();
  K_A(0, 0) = fx;
  K_A(1, 1) = fx;  // fx = fy assumed (square pixels).
  K_A(0, 2) = cx;
  K_A(1, 2) = cy;

  if (!K_A.allFinite() || !D_A.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // ------------------------------------------------------------------
  // 5. Phase B: best-effort joint refinement via camxiom PnpSolver.
  //
  //    PnpSolver's default flags (NONE) leave K, D, projection params,
  //    and per-view extrinsics all free — exactly the joint refinement
  //    contract we need. We hand it a CameraModel built from Phase A's
  //    K + D and the per-view rvecs/tvecs in angle-axis form, then run
  //    once. Any failure or non-finite output is silently absorbed and
  //    Phase A's result is committed to result_out.
  // ------------------------------------------------------------------
  Eigen::Matrix3d K_final = K_A;
  Eigen::Vector4d D_final = D_A;
  std::vector<Eigen::Matrix3d> R_final = R_views;
  std::vector<Eigen::Vector3d> t_final = t_views;

  {
    camxiom::CameraModel64 model64_for_pnp = makeSeedModel(
      K_A(0, 0), K_A(1, 1), K_A(0, 2), K_A(1, 2),
      /*use_kb4=*/true, D_A
    );

    camxiom::CameraModel cam_for_pnp = camxiom::fromCameraModel64(model64_for_pnp);
    camxiom::updateThetaMax(cam_for_pnp);

    if (camxiom::validateCameraModel(cam_for_pnp) == camxiom::StatusCode::OK)
    {
      // Build the per-view object/image point lists in PnpSolver's expected
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
      // Bounded principal-point upper bound: the default 5000 is too low
      // for some imagers; we widen it to twice the image extent which is
      // generous yet still rejects runaway updates.
      const double pp_upper =
        2.0 * std::max(static_cast<double>(image_width), static_cast<double>(image_height));
      pnp_opts.upper_bound.principal_points = Eigen::Vector2d(pp_upper, pp_upper);
      // Same for focal length: a 10x slack relative to the image height
      // covers very wide and very narrow lenses while staying finite.
      const double f_upper = 10.0 * static_cast<double>(image_height);
      pnp_opts.upper_bound.focal_lengths = Eigen::Vector2d(f_upper, f_upper);

      camxiom::optimizer::PnpSolver solver;
      camxiom::optimizer::PnpResult result;
      // Lock k3 and k4 (KB4 distortion indices 2 and 3) at the Phase A
      // value of 0.0. Under sub-pixel noise on typical fisheye calibration
      // geometries (e.g. theta in [0, ~1 rad]) the theta^7 and theta^9
      // columns of the linearised KB4 design matrix are not uniquely
      // identifiable: refining them creates aliasing local minima that
      // Ceres falls into, producing |D| inflation by orders of magnitude
      // while reprojection RMS stays acceptable. We therefore keep k3 = k4
      // = 0 here and let MS2's full IntrinsicsCalibrator free them later
      // when more data / better strategies are available.
      const camxiom::optimizer::PnpFlag pnp_flags =
        camxiom::optimizer::PnpFlag::FIX_DIST_2 | camxiom::optimizer::PnpFlag::FIX_DIST_3;
      solver.solve(obj_sets, img_sets, guess, result, pnp_opts, pnp_flags);

      if (phaseBLooksUsable(result, n_views))
      {
        // Commit the refined model and poses.
        K_final(0, 0) = static_cast<double>(result.camera_model.intrinsics.fx);
        K_final(1, 1) = static_cast<double>(result.camera_model.intrinsics.fy);
        K_final(0, 2) = static_cast<double>(result.camera_model.intrinsics.cx);
        K_final(1, 2) = static_cast<double>(result.camera_model.intrinsics.cy);
        K_final(0, 1) = 0.0;  // skew normalised away
        K_final(1, 0) = 0.0;
        K_final(2, 0) = 0.0;
        K_final(2, 1) = 0.0;
        K_final(2, 2) = 1.0;

        for (int i = 0; i < 4; ++i)
        {
          D_final(i) =
            static_cast<double>(result.camera_model.distortion.coeffs[static_cast<std::size_t>(i)]);
        }

        for (std::size_t i = 0; i < n_views; ++i)
        {
          R_final[i] = angleAxisToRotationMatrix(result.rvecs[i]);
          t_final[i] = result.tvecs[i];
        }
      }
      // else: silently fall back to Phase A (already in K_final / D_final
      // / R_final / t_final).
    }
  }

  // ------------------------------------------------------------------
  // 6. Final finite sweep before mutating the out parameter.
  // ------------------------------------------------------------------
  if (!K_final.allFinite() || !D_final.allFinite())
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

  result_out.K = K_final;
  result_out.D = D_final;
  result_out.R_per_view = std::move(R_final);
  result_out.t_per_view = std::move(t_final);
  return StatusCode::OK;
}

}  // namespace camxiom::init
