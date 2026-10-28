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

#include "camxiom/calib/mapping_compare.hpp"

#include "camxiom/default_seed.hpp"
#include "camxiom/types.hpp"

#include <gtest/gtest.h>

#include <cmath>

using camxiom::CameraModel;
using camxiom::StatusCode;
using camxiom::calib::compareProjectionMappings;
using camxiom::calib::MappingComparison;
using camxiom::calib::MappingDomainReport;
using camxiom::calib::MappingDomainSpec;
using camxiom::calib::validateMappingDomain;

namespace
{

CameraModel pinholeRadtan5(double fx, double fy, double k1 = 0.0)
{
  CameraModel model = camxiom::getDefaultSeed(camxiom::ProjectionModelType::PINHOLE, 1920, 1200);
  model.intrinsics.fx = static_cast<float>(fx);
  model.intrinsics.fy = static_cast<float>(fy);
  model.distortion.type = camxiom::DistortionModelType::RADTAN5;
  model.distortion.count = 5U;
  model.distortion.coeffs[0] = static_cast<float>(k1);
  return model;
}

MappingDomainSpec fullFrameSpec()
{
  MappingDomainSpec spec;
  spec.image_width = 1920;
  spec.image_height = 1200;
  spec.lattice_cols = 32;
  spec.lattice_rows = 20;
  return spec;
}

}  // namespace

TEST(MappingCompare, IdenticalModelsHaveZeroDelta)
{
  const CameraModel model = pinholeRadtan5(1620.0, 1620.0, -0.1);
  const MappingComparison cmp = compareProjectionMappings(model, model, fullFrameSpec());

  ASSERT_EQ(cmp.status, StatusCode::OK);
  EXPECT_EQ(cmp.lattice_points, 32 * 20);
  EXPECT_EQ(cmp.both_valid, cmp.lattice_points);
  EXPECT_EQ(cmp.only_a_valid, 0);
  EXPECT_EQ(cmp.only_b_valid, 0);
  // acos near cosine == 1 has an intrinsic double-precision noise floor of
  // ~sqrt(eps) = 1.5e-8 rad; anything below 1e-7 rad is "identical".
  EXPECT_NEAR(cmp.rms_angular_delta_rad, 0.0, 1e-7);
  EXPECT_NEAR(cmp.max_angular_delta_rad, 0.0, 1e-7);
}

TEST(MappingCompare, DifferentFocalLengthsProduceAngularDelta)
{
  // 10% focal difference on a ~30 deg half-FOV camera: border rays differ by
  // a couple of degrees -- far above numeric noise, well below pi.
  const MappingComparison cmp = compareProjectionMappings(
    pinholeRadtan5(1620.0, 1620.0), pinholeRadtan5(1782.0, 1782.0), fullFrameSpec()
  );

  ASSERT_EQ(cmp.status, StatusCode::OK);
  EXPECT_EQ(cmp.both_valid, cmp.lattice_points);
  EXPECT_GT(cmp.rms_angular_delta_rad, 0.005);
  EXPECT_LT(cmp.max_angular_delta_rad, 0.2);
  EXPECT_GE(cmp.max_angular_delta_rad, cmp.p95_angular_delta_rad);
  EXPECT_GE(cmp.p95_angular_delta_rad, cmp.rms_angular_delta_rad * 0.5);
}

TEST(MappingCompare, RejectsInvalidSpecAndModels)
{
  MappingDomainSpec bad = fullFrameSpec();
  bad.image_width = 0;
  const CameraModel model = pinholeRadtan5(1620.0, 1620.0);
  EXPECT_EQ(compareProjectionMappings(model, model, bad).status, StatusCode::INVALID_INPUT);

  CameraModel invalid = model;
  invalid.intrinsics.fx = 0.0f;
  EXPECT_EQ(
    compareProjectionMappings(invalid, model, fullFrameSpec()).status, StatusCode::INVALID_INPUT
  );
}

TEST(MappingCompare, DomainReportCoversFullPinholeFrame)
{
  const MappingDomainReport report =
    validateMappingDomain(pinholeRadtan5(1620.0, 1620.0, -0.1), fullFrameSpec());
  ASSERT_EQ(report.status, StatusCode::OK);
  EXPECT_EQ(report.lattice_points, 32 * 20);
  EXPECT_EQ(report.valid_points, report.lattice_points);
}
