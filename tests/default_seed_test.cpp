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

// Tests for camxiom::getDefaultSeed — a pure data-independent factory.
//
// This is NOT an init estimator: no synthetic checkerboard, no noise, no
// ground-truth recovery loop.  The three test families are:
//   1. ExactFieldValues  — assert every seed field matches the documented
//                          contract for each of the 5 projection models.
//   2. ValidatesOk       — for all 5 models across a range of image sizes,
//                          validateCameraModel(seed) == StatusCode::OK.
//   3. InvalidInputSentinel — zero / negative dims and UNKNOWN model type
//                             return a sentinel whose projection.type is
//                             UNKNOWN and validateCameraModel returns
//                             INVALID_MODEL.

#include "camxiom/default_seed.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/types.hpp"

#include <gtest/gtest.h>

#include <cmath>

using camxiom::CameraModel;
using camxiom::DistortionModelType;
using camxiom::DistortionSpace;
using camxiom::getDefaultSeed;
using camxiom::ProjectionModelType;
using camxiom::StatusCode;
using camxiom::validateCameraModel;

// ---------------------------------------------------------------------------
// 1. ExactFieldValues
// ---------------------------------------------------------------------------
//
// For each model we test at two sizes:
//   - square 800x800: cx == cy == 400; fx/fy from height (same as width here)
//   - non-square 1920x1080: cx == 960, cy == 540; fx/fy from height (1080)
// A bug that swaps width and height in cx/cy or fx/fy will fail one of these.

TEST(DefaultSeed, ExactFieldValuesPinholeSquare)
{
  const int w = 800, h = 800;
  const CameraModel m = getDefaultSeed(ProjectionModelType::PINHOLE, w, h);

  // projection
  EXPECT_EQ(m.projection.type, ProjectionModelType::PINHOLE);

  // intrinsics: cx from width, cy/fx/fy from height (both 400 here)
  const float expected_c = static_cast<float>(h / 2.0);
  const float expected_fx = static_cast<float>(h / 2.0);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, expected_c);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  // distortion: no distortion
  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesPinholeNonSquare)
{
  const int w = 1920, h = 1080;
  const CameraModel m = getDefaultSeed(ProjectionModelType::PINHOLE, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::PINHOLE);

  // cx comes from width (1920), cy from height (1080)
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  // fx and fy from height (h/2 == 540)
  const float expected_fx = static_cast<float>(h / 2.0);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesFisheyeThetaSquare)
{
  const int w = 800, h = 800;
  const CameraModel m = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::FISHEYE_THETA);
  // theta_max == pi (inclusive bound in the validator; using kPiF exactly)
  EXPECT_FLOAT_EQ(m.projection.theta_max, camxiom::constants::kPiF);

  // fx = fy = h / pi; mirroring impl: double division then narrow to float
  const float expected_fx = static_cast<float>(static_cast<double>(h) / camxiom::constants::kPi);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  // KB4 angle distortion, k1..k4 = 0
  EXPECT_EQ(m.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(m.distortion.space, DistortionSpace::ANGLE);
  EXPECT_EQ(m.distortion.count, 4U);
  EXPECT_EQ(m.distortion.is_rational, false);
  EXPECT_EQ(m.distortion.has_thin_prism, false);
  EXPECT_EQ(m.distortion.has_tilt, false);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesFisheyeThetaNonSquare)
{
  const int w = 1920, h = 1080;
  const CameraModel m = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_FLOAT_EQ(m.projection.theta_max, camxiom::constants::kPiF);

  // cx from width, cy/fx/fy from height
  const float expected_fx = static_cast<float>(static_cast<double>(h) / camxiom::constants::kPi);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(m.distortion.space, DistortionSpace::ANGLE);
  EXPECT_EQ(m.distortion.count, 4U);
  EXPECT_EQ(m.distortion.is_rational, false);
  EXPECT_EQ(m.distortion.has_thin_prism, false);
  EXPECT_EQ(m.distortion.has_tilt, false);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesOmnidirectionalSquare)
{
  const int w = 800, h = 800;
  const CameraModel m = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::OMNIDIRECTIONAL);
  EXPECT_FLOAT_EQ(m.projection.xi, 1.0f);

  const float expected_fx = static_cast<float>(h / 2.0);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesOmnidirectionalNonSquare)
{
  const int w = 1920, h = 1080;
  const CameraModel m = getDefaultSeed(ProjectionModelType::OMNIDIRECTIONAL, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::OMNIDIRECTIONAL);
  EXPECT_FLOAT_EQ(m.projection.xi, 1.0f);

  const float expected_fx = static_cast<float>(h / 2.0);  // 540
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));  // 960
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));  // 540
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
}

TEST(DefaultSeed, ExactFieldValuesDoubleSphereSquare)
{
  const int w = 800, h = 800;
  const CameraModel m = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_FLOAT_EQ(m.projection.xi, -0.2f);
  EXPECT_FLOAT_EQ(m.projection.alpha, 0.5f);

  const float expected_fx = static_cast<float>(h / 2.0);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesDoubleSphereNonSquare)
{
  const int w = 1920, h = 1080;
  const CameraModel m = getDefaultSeed(ProjectionModelType::DOUBLE_SPHERE, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::DOUBLE_SPHERE);
  EXPECT_FLOAT_EQ(m.projection.xi, -0.2f);
  EXPECT_FLOAT_EQ(m.projection.alpha, 0.5f);

  const float expected_fx = static_cast<float>(h / 2.0);  // 540
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));  // 960
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));  // 540
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
}

TEST(DefaultSeed, ExactFieldValuesEucmSquare)
{
  const int w = 800, h = 800;
  const CameraModel m = getDefaultSeed(ProjectionModelType::EUCM, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::EUCM);
  EXPECT_FLOAT_EQ(m.projection.alpha, 0.5f);
  EXPECT_FLOAT_EQ(m.projection.beta, 1.0f);

  const float expected_fx = static_cast<float>(h / 2.0);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
  for (std::size_t i = 0; i < m.distortion.coeffs.size(); ++i)
  {
    EXPECT_EQ(m.distortion.coeffs[i], 0.0f) << "coeffs[" << i << "] != 0";
  }
}

TEST(DefaultSeed, ExactFieldValuesEucmNonSquare)
{
  const int w = 1920, h = 1080;
  const CameraModel m = getDefaultSeed(ProjectionModelType::EUCM, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::EUCM);
  EXPECT_FLOAT_EQ(m.projection.alpha, 0.5f);
  EXPECT_FLOAT_EQ(m.projection.beta, 1.0f);

  const float expected_fx = static_cast<float>(h / 2.0);  // 540
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));  // 960
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));  // 540
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::NONE);
  EXPECT_EQ(m.distortion.space, DistortionSpace::NONE);
  EXPECT_EQ(m.distortion.count, 0U);
}

// Also exercise the representative 1280x960 size mentioned in the spec
// for the FISHEYE_THETA model (non-power-of-2 height gives a non-trivial
// h/pi value, exposing float-narrowing issues if kPi precision is wrong).
TEST(DefaultSeed, ExactFieldValuesFisheyeTheta1280x960)
{
  const int w = 1280, h = 960;
  const CameraModel m = getDefaultSeed(ProjectionModelType::FISHEYE_THETA, w, h);

  EXPECT_EQ(m.projection.type, ProjectionModelType::FISHEYE_THETA);
  EXPECT_FLOAT_EQ(m.projection.theta_max, camxiom::constants::kPiF);

  // Mirror the impl: double division, then narrow.
  const float expected_fx = static_cast<float>(static_cast<double>(h) / camxiom::constants::kPi);
  EXPECT_FLOAT_EQ(m.intrinsics.fx, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.fy, expected_fx);
  EXPECT_FLOAT_EQ(m.intrinsics.cx, static_cast<float>(w / 2.0));  // 640
  EXPECT_FLOAT_EQ(m.intrinsics.cy, static_cast<float>(h / 2.0));  // 480
  EXPECT_EQ(m.intrinsics.skew, 0.0f);

  EXPECT_EQ(m.distortion.type, DistortionModelType::KB4);
  EXPECT_EQ(m.distortion.space, DistortionSpace::ANGLE);
  EXPECT_EQ(m.distortion.count, 4U);
}

// ---------------------------------------------------------------------------
// 2. ValidatesOk
//
// For every model, across a representative set of image sizes, the returned
// seed must satisfy validateCameraModel() == StatusCode::OK.
// This is the key correctness invariant: it exercises the validator's bounds
// on theta_max (inclusive pi for FISHEYE_THETA), xi range (DS), etc.
// ---------------------------------------------------------------------------

namespace
{

// Representative sizes: tiny, square, wide landscape, tall portrait, large.
struct SizeParam
{
  int w, h;
  const char *label;
};

constexpr SizeParam kTestSizes[] = {
  {4, 4, "4x4 tiny"},
  {640, 480, "640x480 VGA"},
  {800, 800, "800x800 square"},
  {1920, 1080, "1920x1080 wide"},
  {480, 1280, "480x1280 tall-portrait"},
  {4096, 2160, "4096x2160 4K"},
};

constexpr ProjectionModelType kAllModels[] = {
  ProjectionModelType::PINHOLE,
  ProjectionModelType::FISHEYE_THETA,
  ProjectionModelType::OMNIDIRECTIONAL,
  ProjectionModelType::DOUBLE_SPHERE,
  ProjectionModelType::EUCM,
};

}  // namespace

TEST(DefaultSeed, ValidatesOkAllModelsAllSizes)
{
  for (const auto &sz : kTestSizes)
  {
    for (const auto &mt : kAllModels)
    {
      const CameraModel m = getDefaultSeed(mt, sz.w, sz.h);
      EXPECT_EQ(validateCameraModel(m), StatusCode::OK)
        << "validateCameraModel failed for model " << static_cast<int>(mt) << " at size "
        << sz.label << " (" << sz.w << "x" << sz.h << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// 3. InvalidInputSentinel
//
// Invalid inputs must return a default-constructed sentinel:
//   projection.type == ProjectionModelType::UNKNOWN
//   validateCameraModel(result) == StatusCode::INVALID_MODEL
// ---------------------------------------------------------------------------

TEST(DefaultSeed, InvalidInputZeroWidth)
{
  // Dim check runs before the model switch (documented in the header).
  const CameraModel m = getDefaultSeed(ProjectionModelType::PINHOLE, 0, 480);
  EXPECT_EQ(m.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(DefaultSeed, InvalidInputZeroHeight)
{
  const CameraModel m = getDefaultSeed(ProjectionModelType::PINHOLE, 640, 0);
  EXPECT_EQ(m.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(DefaultSeed, InvalidInputNegativeBothDims)
{
  const CameraModel m = getDefaultSeed(ProjectionModelType::PINHOLE, -10, -10);
  EXPECT_EQ(m.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}

TEST(DefaultSeed, InvalidInputUnknownModelValidDims)
{
  // Valid dims but unrecognised model type.
  const CameraModel m = getDefaultSeed(ProjectionModelType::UNKNOWN, 640, 480);
  EXPECT_EQ(m.projection.type, ProjectionModelType::UNKNOWN);
  EXPECT_EQ(validateCameraModel(m), StatusCode::INVALID_MODEL);
}
