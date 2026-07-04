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

// Tests for calib::convertCameraModel (model-family conversion by fit).
//
// Three regimes are pinned:
//   * an EXACT pair (pinhole ⊂ double sphere): the fit must reproduce the
//     source geometry to numerical precision;
//   * an APPROXIMATE pair (narrow-FOV KB4 -> pinhole+radtan5): sub-pixel fit;
//   * an UNREPRESENTABLE pair (wide fisheye -> pinhole): the conversion must
//     report the mismatch honestly (large residual and/or unrepresentable
//     grid points) instead of silently "succeeding".

#include "camxiom/calib/convert.hpp"
#include "camxiom/default_seed.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace camxiom;
using calib::convertCameraModel;
using calib::ModelConversionOptions;
using calib::ModelConversionResult;

namespace
{

CameraModel makePinhole(const float fx)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  m.projection.theta_max = static_cast<float>(constants::kHalfPi);
  return m;
}

CameraModel makeKB4(const float fx, const float k1, const float k2)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(constants::kHalfPi);
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 4U;
  m.distortion.coeffs[0] = k1;
  m.distortion.coeffs[1] = k2;
  return m;
}

CameraModel makePinholeRadtan5Seed()
{
  CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  seed.distortion.type = DistortionModelType::RADTAN5;
  seed.distortion.space = DistortionSpace::PLANE;
  seed.distortion.count = 5U;  // coefficients start at zero
  return seed;
}

CameraModel makeMeiOmni(const float fx, const float xi)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.xi = xi;
  updateThetaMax(m);
  return m;
}

CameraModel makeEucm(const float fx, const float alpha, const float beta)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.alpha = alpha;
  m.projection.beta = beta;
  updateThetaMax(m);
  return m;
}

}  // namespace

TEST(ModelConvert, PinholeToDoubleSphereIsExact)
{
  // Double sphere contains the pinhole model exactly (xi = 0, alpha = 0), so
  // the fit must reproduce the source geometry to numerical precision.
  const CameraModel src = makePinhole(500.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, seed);

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.camera_model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_EQ(res.representable_point_count, res.used_point_count);
  EXPECT_LT(res.rms_fit_error_px, 1e-2) << "exact-pair conversion should be numerically tight";
  EXPECT_EQ(validateCameraModel(res.camera_model), StatusCode::OK);
}

TEST(ModelConvert, NarrowKB4ToPinholeRadtan5IsSubPixel)
{
  // fx = 500 puts the image diagonal at theta ~= 0.67 rad (~38 deg): well
  // inside what pinhole+radtan5 can approximate.
  const CameraModel src = makeKB4(500.0f, 0.01f, -0.005f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, makePinholeRadtan5Seed());

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.camera_model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_EQ(res.representable_point_count, res.used_point_count);
  EXPECT_LT(res.rms_fit_error_px, 0.5) << "narrow-FOV KB4 should fit pinhole+radtan5 to sub-pixel";
  EXPECT_LT(res.max_fit_error_px, 2.0);
}

TEST(ModelConvert, WideFisheyeToPinholeReportsTheMismatch)
{
  // fx = 150 puts the image corners at theta > 1.2 rad (~70-90 deg) where a
  // rectilinear pinhole cannot follow. The conversion must NOT pretend this
  // worked: either part of the grid is unrepresentable by the fitted pinhole
  // or the residual is honestly large.
  const CameraModel src = makeKB4(150.0f, 0.0f, 0.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, makePinholeRadtan5Seed());

  ASSERT_GT(res.used_point_count, 0);
  const bool mismatch_reported = res.status != StatusCode::OK ||
                                 res.representable_point_count < res.used_point_count ||
                                 res.rms_fit_error_px > 1.0;
  EXPECT_TRUE(mismatch_reported) << "status=" << toString(res.status)
                                 << " rms=" << res.rms_fit_error_px
                                 << " representable=" << res.representable_point_count << "/"
                                 << res.used_point_count;
}

TEST(ModelConvert, MeiOmniToEucmIsExact)
{
  // The header's own headline use case (MEI -> EUCM) — and an EXACT pair:
  // a distortion-free MEI/omni model is the UCM, which EUCM contains at
  // beta = 1, alpha = xi/(1+xi), f' = f/(1+xi). Neither family had appeared
  // as source or destination in this suite before.
  const CameraModel src = makeMeiOmni(400.0f, 1.2f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);
  const CameraModel seed = getDefaultSeed(ProjectionModelType::EUCM, 640, 480);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, seed);

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.camera_model.projection.type, ProjectionModelType::EUCM);
  EXPECT_EQ(res.representable_point_count, res.used_point_count);
  EXPECT_LT(res.rms_fit_error_px, 1e-2)
    << "UCM subset of EUCM should convert to numerical precision";
  EXPECT_EQ(validateCameraModel(res.camera_model), StatusCode::OK);
}

TEST(ModelConvert, PinholeToEucmIsExact)
{
  // The other exact EUCM embedding: pinhole = EUCM at alpha = 0.
  const CameraModel src = makePinhole(500.0f);
  const CameraModel seed = getDefaultSeed(ProjectionModelType::EUCM, 640, 480);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, seed);

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.representable_point_count, res.used_point_count);
  EXPECT_LT(res.rms_fit_error_px, 1e-2);
}

TEST(ModelConvert, NarrowEucmToPinholeRadtan5IsSubPixel)
{
  // EUCM as SOURCE: a mild (alpha = 0.3) narrow-FOV EUCM deviates from
  // rectilinear only gently, so pinhole+radtan5 must absorb it sub-pixel.
  const CameraModel src = makeEucm(500.0f, 0.3f, 1.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);

  const ModelConversionResult res = convertCameraModel(src, 640, 480, makePinholeRadtan5Seed());

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_EQ(res.camera_model.projection.type, ProjectionModelType::PINHOLE);
  EXPECT_LT(res.rms_fit_error_px, 0.5);
  EXPECT_LT(res.max_fit_error_px, 2.0);
}

TEST(ModelConvert, TelephotoFitIsNotClampedAtTheStockBound)
{
  // fx = 7000 on 640x480 (~3 deg telephoto). The stock PnpBound upper limits
  // cap focal lengths at 5000 px; before the shared bound widening this fit
  // clamped at the cap (and a hand-tuned warm start above 5000 was outright
  // infeasible for the solver). Same-family conversion, so the fit must
  // recover the source focal length essentially exactly.
  const CameraModel src = makePinhole(7000.0f);
  ASSERT_EQ(validateCameraModel(src), StatusCode::OK);

  CameraModel seed = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 480);
  seed.intrinsics.fx = 6500.0f;  // hand-tuned warm start, above the stock cap
  seed.intrinsics.fy = 6500.0f;

  const ModelConversionResult res = convertCameraModel(src, 640, 480, seed);

  ASSERT_EQ(res.status, StatusCode::OK);
  EXPECT_NEAR(static_cast<double>(res.camera_model.intrinsics.fx), 7000.0, 5.0);
  EXPECT_NEAR(static_cast<double>(res.camera_model.intrinsics.fy), 7000.0, 5.0);
  EXPECT_LT(res.rms_fit_error_px, 1e-2);
}

TEST(ModelConvert, IterationCappedFitReportsNonConverged)
{
  // With max_iterations = 1 the solver stops at the iteration cap: Ceres
  // still deems the solution usable, but the documented contract is that OK
  // means CONVERGED. The best-effort model and fit-quality numbers must
  // still be populated so the caller can inspect what the single step
  // produced.
  const CameraModel src = makePinhole(500.0f);
  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);

  ModelConversionOptions opts;
  opts.max_iterations = 1;
  const ModelConversionResult res = convertCameraModel(src, 640, 480, seed, opts);

  EXPECT_EQ(res.status, StatusCode::NON_CONVERGED);
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.camera_model.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_GT(res.used_point_count, 0);
  EXPECT_TRUE(std::isfinite(res.rms_fit_error_px));
}

TEST(ModelConvert, RejectsBadInputs)
{
  const CameraModel src = makePinhole(500.0f);
  const CameraModel seed = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, 640, 480);

  EXPECT_EQ(convertCameraModel(src, 0, 480, seed).status, StatusCode::INVALID_INPUT);

  ModelConversionOptions opts;
  opts.grid_cols = 1;
  EXPECT_EQ(convertCameraModel(src, 640, 480, seed, opts).status, StatusCode::INVALID_INPUT);

  // cols x rows overflows int (2^32): must be rejected up front, not fed
  // into a signed-overflow product / multi-gigabyte reserve.
  ModelConversionOptions huge;
  huge.grid_cols = 65536;
  huge.grid_rows = 65536;
  EXPECT_EQ(convertCameraModel(src, 640, 480, seed, huge).status, StatusCode::INVALID_INPUT);

  // Note: a negative focal passes validateCameraModel (only |fx| <= eps is
  // rejected — a mirrored model is still solvable), so break the seed with an
  // unknown projection type instead.
  CameraModel broken = seed;
  broken.projection.type = ProjectionModelType::UNKNOWN;
  EXPECT_EQ(convertCameraModel(src, 640, 480, broken).status, StatusCode::INVALID_MODEL);
}
