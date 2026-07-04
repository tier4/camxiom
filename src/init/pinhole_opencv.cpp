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

#include "camxiom/init/pinhole_opencv.hpp"

#include "camxiom/init/homography.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <cmath>
#include <cstddef>

namespace camxiom::init
{

namespace
{

constexpr Eigen::Index kMinPointsPerView = 4;

}  // namespace

StatusCode estimatePinholeOpenCv(
  const std::vector<PlanarObservation> &views, int image_width, int image_height,
  double aspect_ratio, Eigen::Matrix3d &K_out
)
{
  if (views.empty() || image_width <= 0 || image_height <= 0 || !std::isfinite(aspect_ratio) || aspect_ratio < 0.0)
  {
    return StatusCode::INVALID_INPUT;
  }

  const double cx = 0.5 * static_cast<double>(image_width - 1);
  const double cy = 0.5 * static_cast<double>(image_height - 1);
  Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(views.size()), 2);
  Eigen::VectorXd b(2 * static_cast<Eigen::Index>(views.size()));

  for (std::size_t i = 0; i < views.size(); ++i)
  {
    const PlanarObservation &view = views[i];
    const Eigen::Index count = view.board_pts.cols();
    if (count < kMinPointsPerView || view.image_pts.cols() != count ||
        !view.board_pts.allFinite() || !view.image_pts.allFinite())
    {
      return StatusCode::INVALID_INPUT;
    }

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    const StatusCode homography_status = estimateHomographyDLT(view.board_pts, view.image_pts, H);
    if (homography_status != StatusCode::OK)
    {
      return homography_status;
    }

    // Move the image origin to the fixed principal point. Homographies are
    // scale-invariant, so estimateHomographyDLT's Frobenius normalisation is
    // compatible with OpenCV's equations below.
    H.row(0) -= cx * H.row(2);
    H.row(1) -= cy * H.row(2);

    Eigen::Vector3d h = H.col(0);
    Eigen::Vector3d v = H.col(1);
    Eigen::Vector3d d1 = 0.5 * (h + v);
    Eigen::Vector3d d2 = 0.5 * (h - v);

    const double h_norm = h.norm();
    const double v_norm = v.norm();
    const double d1_norm = d1.norm();
    const double d2_norm = d2.norm();
    if (!(h_norm > 0.0) || !(v_norm > 0.0) || !(d1_norm > 0.0) || !(d2_norm > 0.0) || !std::isfinite(h_norm) || !std::isfinite(v_norm) || !std::isfinite(d1_norm) || !std::isfinite(d2_norm))
    {
      return StatusCode::DEGENERATE_CONFIG;
    }

    h /= h_norm;
    v /= v_norm;
    d1 /= d1_norm;
    d2 /= d2_norm;

    const Eigen::Index row = 2 * static_cast<Eigen::Index>(i);
    A(row, 0) = h.x() * v.x();
    A(row, 1) = h.y() * v.y();
    b(row) = -h.z() * v.z();
    A(row + 1, 0) = d1.x() * d2.x();
    A(row + 1, 1) = d1.y() * d2.y();
    b(row + 1) = -d1.z() * d2.z();
  }

  if (!A.allFinite() || !b.allFinite())
  {
    return StatusCode::NUMERIC_ERROR;
  }

  // OpenCV requests DECOMP_NORMAL | DECOMP_SVD, i.e. SVD on the 2x2 normal
  // equations rather than directly on the tall design matrix.
  const Eigen::Matrix2d normal = A.transpose() * A;
  const Eigen::Vector2d rhs = A.transpose() * b;
  Eigen::JacobiSVD<Eigen::Matrix2d> svd(normal, Eigen::ComputeFullU | Eigen::ComputeFullV);
  svd.setThreshold(Eigen::NumTraits<double>::epsilon());
  if (svd.rank() < 2)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }
  const Eigen::Vector2d inverse_focal_sq = svd.solve(rhs);
  if (!inverse_focal_sq.allFinite() || inverse_focal_sq.x() == 0.0 || inverse_focal_sq.y() == 0.0)
  {
    return StatusCode::NUMERIC_ERROR;
  }
  // The unknowns are 1/fx^2 and 1/fy^2: a non-positive solution has no
  // physical focal length. OpenCV's cvInitIntrinsicParams2D papers over it
  // with sqrt(abs(.)), silently returning a garbage focal from degenerate
  // view geometry (e.g. near-frontal-parallel boards); report the
  // degeneracy instead of seeding the refinement with fiction.
  if (inverse_focal_sq.x() <= 0.0 || inverse_focal_sq.y() <= 0.0)
  {
    return StatusCode::DEGENERATE_CONFIG;
  }

  double fx = std::sqrt(1.0 / inverse_focal_sq.x());
  double fy = std::sqrt(1.0 / inverse_focal_sq.y());
  if (aspect_ratio > 0.0)
  {
    const double shared = (fx + fy) / (aspect_ratio + 1.0);
    fx = aspect_ratio * shared;
    fy = shared;
  }
  if (!(fx > 0.0) || !(fy > 0.0) || !std::isfinite(fx) || !std::isfinite(fy))
  {
    return StatusCode::NUMERIC_ERROR;
  }

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = fx;
  K(1, 1) = fy;
  K(0, 2) = cx;
  K(1, 2) = cy;
  K_out = K;
  return StatusCode::OK;
}

}  // namespace camxiom::init
