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

// AutoDiff (ceres::Jet) vs analytic projection-Jacobian agreement (LOADMAP T4).
//
// camxiom carries TWO independent forward-projection implementations:
//   * the analytic runtime path (src/jacobian/jacobian_impl.hpp) which returns
//     the closed-form 2x3 Jacobian d(u,v)/d(X,Y,Z) via rayToPixelWithJacobian64;
//   * the header-only projection_template (include/camxiom/projection_template.hpp)
//     which is the SINGLE forward path used inside the Ceres PnP cost functors and
//     is instantiated with ceres::Jet for automatic differentiation.
//
// #1 unified the analytic float/double Jacobian into one template, but the
// AutoDiff template is a separate re-derivation. This test is the regression
// guard that the two never silently diverge: for representative models it seeds
// a ceres::Jet<double,3> on the camera-frame ray (X,Y,Z), runs the AutoDiff
// forward template, and checks BOTH the projected pixel AND the d(pixel)/d(ray)
// Jacobian match the analytic path to double-precision tolerance.
//
// Ceres-only test (needs ceres::Jet): registered under CAMXIOM_WITH_CERES.

#include "camxiom/internal/constants.hpp"
#include "camxiom/jacobian64.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"  // validateCameraModel64
#include "camxiom/projection_template.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <ceres/jet.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace camxiom;

namespace
{

// Cached tilt / inverse-tilt matrices for the TILTED14 model, mirroring the
// runtime precompute (src/model/internal.hpp). The analytic Jacobian consumes
// these cached matrices while projectTilted14 rebuilds them from tau; seeding
// them consistently lets the two implementations be compared directly.
void computeTiltMatrices(
  double tau_x, double tau_y, std::array<double, 9> &tilt, std::array<double, 9> &inv_tilt
)
{
  const double c_tx = std::cos(tau_x);
  const double s_tx = std::sin(tau_x);
  const double c_ty = std::cos(tau_y);
  const double s_ty = std::sin(tau_y);

  const std::array<double, 9> rot_xy{c_ty, s_ty * s_tx, -s_ty * c_tx, 0.0,        c_tx,
                                     s_tx, s_ty,        -c_ty * s_tx, c_ty * c_tx};

  const double r22 = rot_xy[8];
  const std::array<double, 9> proj_z{r22, 0.0, -rot_xy[2], 0.0, r22, -rot_xy[5], 0.0, 0.0, 1.0};

  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      double value = 0.0;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += proj_z[row * 3 + mid] * rot_xy[mid * 3 + col];
      }
      tilt[row * 3 + col] = value;
    }
  }

  const double inv_r22 = 1.0 / r22;
  const std::array<double, 9> inv_proj_z{
    inv_r22, 0.0, rot_xy[2] * inv_r22, 0.0, inv_r22, rot_xy[5] * inv_r22, 0.0, 0.0, 1.0};
  const std::array<double, 9> rot_xy_t{rot_xy[0], rot_xy[3], rot_xy[6], rot_xy[1], rot_xy[4],
                                       rot_xy[7], rot_xy[2], rot_xy[5], rot_xy[8]};
  for (int row = 0; row < 3; ++row)
  {
    for (int col = 0; col < 3; ++col)
    {
      double value = 0.0;
      for (int mid = 0; mid < 3; ++mid)
      {
        value += rot_xy_t[row * 3 + mid] * inv_proj_z[mid * 3 + col];
      }
      inv_tilt[row * 3 + col] = value;
    }
  }
}

CameraModel64 makeIntrinsics()
{
  CameraModel64 m;
  m.intrinsics.fx = 520.0;
  m.intrinsics.fy = 519.0;
  m.intrinsics.cx = 322.0;
  m.intrinsics.cy = 241.0;
  m.intrinsics.skew = 0.0;  // projection_template has no skew term
  return m;
}

CameraModel64 makePinholeNone()
{
  CameraModel64 m = makeIntrinsics();
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel64 makePinholeRadtan5()
{
  CameraModel64 m = makeIntrinsics();
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::RADTAN5;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.28;
  m.distortion.coeffs[1] = 0.11;
  m.distortion.coeffs[2] = 0.0006;
  m.distortion.coeffs[3] = -0.0004;
  m.distortion.coeffs[4] = -0.02;
  m.distortion.count = 5U;
  return m;
}

CameraModel64 makePinholeRational8()
{
  CameraModel64 m = makeIntrinsics();
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::RATIONAL8;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.30;
  m.distortion.coeffs[1] = 0.12;
  m.distortion.coeffs[2] = 0.0005;
  m.distortion.coeffs[3] = -0.0003;
  m.distortion.coeffs[4] = -0.02;
  m.distortion.coeffs[5] = 0.001;
  m.distortion.coeffs[6] = 0.0006;
  m.distortion.coeffs[7] = -0.0002;
  m.distortion.count = 8U;
  m.distortion.is_rational = true;
  return m;
}

CameraModel64 makePinholeThinPrism12()
{
  CameraModel64 m = makeIntrinsics();
  m.projection.type = ProjectionModelType::PINHOLE;
  m.distortion.type = DistortionModelType::THIN_PRISM12;
  m.distortion.space = DistortionSpace::PLANE;
  m.distortion.coeffs[0] = -0.28;
  m.distortion.coeffs[1] = 0.11;
  m.distortion.coeffs[2] = 0.0006;
  m.distortion.coeffs[3] = -0.0004;
  m.distortion.coeffs[4] = -0.02;
  m.distortion.coeffs[5] = 0.001;
  m.distortion.coeffs[6] = 0.0005;
  m.distortion.coeffs[7] = -0.0002;
  m.distortion.coeffs[8] = 0.0003;
  m.distortion.coeffs[9] = -0.0002;
  m.distortion.coeffs[10] = 0.0001;
  m.distortion.coeffs[11] = 0.0002;
  m.distortion.count = 12U;
  m.distortion.is_rational = true;
  m.distortion.has_thin_prism = true;
  return m;
}

CameraModel64 makePinholeTilted14()
{
  CameraModel64 m = makePinholeThinPrism12();
  m.distortion.type = DistortionModelType::TILTED14;
  m.distortion.coeffs[12] = 0.02;    // tau_x
  m.distortion.coeffs[13] = -0.015;  // tau_y
  m.distortion.count = 14U;
  m.distortion.has_tilt = true;
  computeTiltMatrices(
    m.distortion.coeffs[12], m.distortion.coeffs[13], m.distortion.tilt_matrix,
    m.distortion.inv_tilt_matrix
  );
  return m;
}

CameraModel64 makeFisheyeEquidistant()
{
  CameraModel64 m = makeIntrinsics();
  m.intrinsics.fx = 285.0;
  m.intrinsics.fy = 285.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<double>(camxiom::constants::kPiF) - 1e-4;
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

CameraModel64 makeFisheyeKb4()
{
  CameraModel64 m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.coeffs[0] = -0.012;
  m.distortion.coeffs[1] = 0.004;
  m.distortion.coeffs[2] = -0.0007;
  m.distortion.coeffs[3] = 0.0001;
  m.distortion.count = 4U;
  return m;
}

// The three coefficient-free trigonometric fisheye variants. Their template
// forward paths used to silently fall back to equidistant, so the AutoDiff
// PnP cost optimised the wrong model; keeping them in the representative set
// pins template <-> analytic agreement.
CameraModel64 makeFisheyeEquisolid()
{
  CameraModel64 m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::EQUISOLID;
  return m;
}

CameraModel64 makeFisheyeStereographic()
{
  CameraModel64 m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::STEREOGRAPHIC;
  return m;
}

CameraModel64 makeFisheyeOrthographic()
{
  CameraModel64 m = makeFisheyeEquidistant();
  m.distortion.type = DistortionModelType::ORTHOGRAPHIC;
  // theta_d = sin(theta) is monotone only on [0, pi/2].
  m.projection.theta_max = camxiom::constants::kHalfPi - 1e-4;
  return m;
}

CameraModel64 makeOmnidirectional()
{
  CameraModel64 m = makeIntrinsics();
  m.intrinsics.fx = 400.0;
  m.intrinsics.fy = 400.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = static_cast<double>(camxiom::constants::kPiF);
  m.projection.xi = 1.0;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel64 makeDoubleSphere()
{
  CameraModel64 m = makeIntrinsics();
  m.intrinsics.fx = 350.0;
  m.intrinsics.fy = 350.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = static_cast<double>(camxiom::constants::kPiF);
  m.projection.xi = 0.5;
  m.projection.alpha = 0.5;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel64 makeEucm()
{
  CameraModel64 m = makeIntrinsics();
  m.intrinsics.fx = 350.0;
  m.intrinsics.fy = 350.0;
  m.intrinsics.cx = 320.0;
  m.intrinsics.cy = 240.0;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = static_cast<double>(camxiom::constants::kPiF);
  m.projection.alpha = 0.6;
  m.projection.beta = 1.1;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

struct NamedModel
{
  std::string name;
  CameraModel64 model;
};

std::vector<NamedModel> representativeModels()
{
  return {
    {"pinhole+none", makePinholeNone()},
    {"pinhole+radtan5", makePinholeRadtan5()},
    {"pinhole+rational8", makePinholeRational8()},
    {"pinhole+thin_prism12", makePinholeThinPrism12()},
    {"pinhole+tilted14", makePinholeTilted14()},
    {"fisheye+equidistant", makeFisheyeEquidistant()},
    {"fisheye+kb4", makeFisheyeKb4()},
    {"fisheye+equisolid", makeFisheyeEquisolid()},
    {"fisheye+stereographic", makeFisheyeStereographic()},
    {"fisheye+orthographic", makeFisheyeOrthographic()},
    {"omnidirectional", makeOmnidirectional()},
    {"double_sphere", makeDoubleSphere()},
    {"eucm", makeEucm()},
  };
}

// Moderate off-axis rays valid (z>0, well within FOV) for every model. On-axis
// is avoided so the fisheye radial scale theta_d/r_xy stays well-conditioned.
std::vector<Eigen::Vector3d> sampleRays()
{
  return {
    Eigen::Vector3d(0.20, 0.10, 1.0),
    Eigen::Vector3d(-0.30, 0.15, 1.0),
    Eigen::Vector3d(0.50, -0.40, 1.0),
    Eigen::Vector3d(0.15, 0.25, 1.0),
  };
}

struct AutoDiffJacobian
{
  bool ok{false};
  double u{0.0};
  double v{0.0};
  Eigen::Matrix<double, 2, 3> J{Eigen::Matrix<double, 2, 3>::Zero()};
};

// Forward-project through the AutoDiff template with a Jet seeded on (X,Y,Z),
// recovering the value (a) and the d(pixel)/d(ray) Jacobian (v[0..2]).
AutoDiffJacobian autodiffJacobian(const CameraModel64 &m, const Eigen::Vector3d &ray)
{
  using Jet = ceres::Jet<double, 3>;
  namespace tpl = camxiom::projection_template;

  const Jet intrinsics[4] = {
    Jet(m.intrinsics.fx), Jet(m.intrinsics.fy), Jet(m.intrinsics.cx), Jet(m.intrinsics.cy)};

  Jet dist[14];
  for (int i = 0; i < 14; ++i)
  {
    const double c = (i < static_cast<int>(m.distortion.count))
                       ? m.distortion.coeffs[static_cast<std::size_t>(i)]
                       : 0.0;
    dist[i] = Jet(c);
  }

  const Jet xi(m.projection.xi);
  const Jet alpha(m.projection.alpha);
  const Jet beta(m.projection.beta);

  const Jet x(ray.x(), 0);
  const Jet y(ray.y(), 1);
  const Jet z(ray.z(), 2);

  Jet u;
  Jet v;
  AutoDiffJacobian out;
  out.ok = tpl::projectGenericParametric<Jet>(
    m.projection.type, m.distortion.type, intrinsics, dist, static_cast<int>(m.distortion.count),
    xi, alpha, beta, x, y, z, u, v
  );
  out.u = u.a;
  out.v = v.a;
  out.J(0, 0) = u.v[0];
  out.J(0, 1) = u.v[1];
  out.J(0, 2) = u.v[2];
  out.J(1, 0) = v.v[0];
  out.J(1, 1) = v.v[1];
  out.J(1, 2) = v.v[2];
  return out;
}

// Absolute+relative closeness: catches formula/derivative bugs (which produce
// O(magnitude) differences) while tolerating floating-point reassociation
// between the analytic form and the AutoDiff-through-forward evaluation.
::testing::AssertionResult close(double actual, double expected, double atol, double rtol)
{
  const double tol = atol + rtol * std::abs(expected);
  if (std::abs(actual - expected) <= tol)
  {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
         << "actual=" << actual << " expected=" << expected
         << " |diff|=" << std::abs(actual - expected) << " tol=" << tol;
}

}  // namespace

TEST(AutoDiffJacobian, MatchesAnalyticForwardAndJacobian)
{
  constexpr double kValAtol = 1e-6;
  constexpr double kValRtol = 1e-9;
  constexpr double kJacAtol = 1e-5;
  constexpr double kJacRtol = 1e-7;

  for (const auto &nm : representativeModels())
  {
    ASSERT_EQ(validateCameraModel64(nm.model), StatusCode::OK) << nm.name;

    for (const auto &ray : sampleRays())
    {
      const ProjectionJacobian64 analytic = rayToPixelWithJacobian64(nm.model, ray);
      ASSERT_EQ(analytic.status, StatusCode::OK)
        << nm.name << " ray=(" << ray.x() << "," << ray.y() << "," << ray.z() << ")";

      const AutoDiffJacobian ad = autodiffJacobian(nm.model, ray);
      ASSERT_TRUE(ad.ok) << nm.name;

      const std::string ctx = nm.name + " ray=(" + std::to_string(ray.x()) + "," +
                              std::to_string(ray.y()) + "," + std::to_string(ray.z()) + ")";

      EXPECT_TRUE(close(ad.u, analytic.pixel.u, kValAtol, kValRtol)) << "u " << ctx;
      EXPECT_TRUE(close(ad.v, analytic.pixel.v, kValAtol, kValRtol)) << "v " << ctx;

      for (int r = 0; r < 2; ++r)
      {
        for (int c = 0; c < 3; ++c)
        {
          EXPECT_TRUE(close(ad.J(r, c), analytic.J(r, c), kJacAtol, kJacRtol))
            << "J(" << r << "," << c << ") " << ctx;
        }
      }
    }
  }
}

// RADTAN4's dist parameter block is exactly 4 wide in the AUTO_DIFF PnP cost
// (pnp_solver sizes the Ceres block by the model's coefficient count). The
// parametric dispatch used to fall through to projectRadtan5, which reads
// dist[4] — one element past the block. Call it with a heap buffer of
// exactly 4 Jets so the ASan CI job turns any regression into a hard
// failure, and pin agreement with the analytic projection.
TEST(AutoDiffJacobian, Radtan4ReadsOnlyFourCoefficients)
{
  using Jet = ceres::Jet<double, 3>;
  namespace tpl = camxiom::projection_template;

  CameraModel64 m = makePinholeRadtan5();
  m.distortion.type = DistortionModelType::RADTAN4;
  m.distortion.coeffs[4] = 0.0;  // RADTAN4 has no k3
  m.distortion.count = 4U;
  ASSERT_EQ(validateCameraModel64(m), StatusCode::OK);

  const Jet intrinsics[4] = {
    Jet(m.intrinsics.fx), Jet(m.intrinsics.fy), Jet(m.intrinsics.cx), Jet(m.intrinsics.cy)};

  std::vector<Jet> dist(4);
  for (int i = 0; i < 4; ++i)
  {
    dist[static_cast<std::size_t>(i)] = Jet(m.distortion.coeffs[static_cast<std::size_t>(i)]);
  }

  for (const auto &ray : sampleRays())
  {
    const ProjectionJacobian64 analytic = rayToPixelWithJacobian64(m, ray);
    ASSERT_EQ(analytic.status, StatusCode::OK);

    Jet u;
    Jet v;
    const bool ok = tpl::projectGenericParametric<Jet>(
      m.projection.type, m.distortion.type, intrinsics, dist.data(),
      /*dist_count=*/4, Jet(0.0), Jet(0.0), Jet(0.0), Jet(ray.x(), 0), Jet(ray.y(), 1),
      Jet(ray.z(), 2), u, v
    );
    ASSERT_TRUE(ok);
    EXPECT_TRUE(close(u.a, analytic.pixel.u, 1e-6, 1e-9));
    EXPECT_TRUE(close(v.a, analytic.pixel.v, 1e-6, 1e-9));
  }
}
