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

// Tests for ValidatedCameraModel (C1): the validated-once, immutable wrapper.
//
// The contract has three parts, each locked in here:
//   1. tryMake() accepts exactly the models validateCameraModel() accepts and
//      rejects (nullopt) the ones it rejects.
//   2. For a validated model, ValidatedCameraModel::rayToPixel / pixelToRay are
//      BIT-FOR-BIT identical to the generic free-function path -- the wrapper
//      only removes the per-call re-validation + dispatch, never changes math.
//   3. The input guards (non-finite pixel, behind-camera ray, all overloads)
//      behave exactly like the generic path.
//
// CORE test: no Ceres / ROS / OpenCV, built in every configuration.

#include "camxiom/validated_model.hpp"

#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/types.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace camxiom;

namespace
{

CameraModel makePinhole(bool with_distortion)
{
  CameraModel m;
  m.intrinsics.fx = 500.0f;
  m.intrinsics.fy = 500.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::PINHOLE;
  if (with_distortion)
  {
    m.distortion.type = DistortionModelType::RADTAN5;
    m.distortion.space = DistortionSpace::PLANE;
    m.distortion.coeffs[0] = -0.28f;
    m.distortion.coeffs[1] = 0.11f;
    m.distortion.coeffs[2] = 0.0006f;
    m.distortion.coeffs[3] = -0.0004f;
    m.distortion.coeffs[4] = -0.02f;
    m.distortion.count = 5U;
  }
  else
  {
    m.distortion.type = DistortionModelType::NONE;
    m.distortion.space = DistortionSpace::NONE;
    m.distortion.count = 0U;
  }
  return m;
}

CameraModel makeFisheyeEquidistant()
{
  CameraModel m;
  m.intrinsics.fx = 280.0f;
  m.intrinsics.fy = 280.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi) - 1e-4f;
  m.distortion.type = DistortionModelType::EQUIDISTANT;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeFisheyeKb4()
{
  CameraModel m;
  m.intrinsics.fx = 285.0f;
  m.intrinsics.fy = 285.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::FISHEYE_THETA;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi) - 1e-3f;
  m.distortion.type = DistortionModelType::KB4;
  m.distortion.space = DistortionSpace::ANGLE;
  m.distortion.coeffs[0] = -0.012f;
  m.distortion.coeffs[1] = 0.004f;
  m.distortion.coeffs[2] = -0.0007f;
  m.distortion.coeffs[3] = 0.0001f;
  m.distortion.count = 4U;
  return m;
}

CameraModel makeOmnidirectional()
{
  CameraModel m;
  m.intrinsics.fx = 400.0f;
  m.intrinsics.fy = 400.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::OMNIDIRECTIONAL;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 1.0f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeDoubleSphere()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::DOUBLE_SPHERE;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.xi = 0.5f;
  m.projection.alpha = 0.5f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

CameraModel makeEucm()
{
  CameraModel m;
  m.intrinsics.fx = 350.0f;
  m.intrinsics.fy = 350.0f;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = ProjectionModelType::EUCM;
  m.projection.theta_max = static_cast<float>(camxiom::constants::kPi);
  m.projection.alpha = 0.6f;
  m.projection.beta = 1.1f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

struct NamedModel
{
  std::string name;
  CameraModel model;
};

std::vector<NamedModel> allValidModels()
{
  return {
    {"pinhole", makePinhole(false)},
    {"pinhole+radtan5", makePinhole(true)},
    {"fisheye+equidistant", makeFisheyeEquidistant()},
    {"fisheye+kb4", makeFisheyeKb4()},
    {"omnidirectional", makeOmnidirectional()},
    {"double_sphere", makeDoubleSphere()},
    {"eucm", makeEucm()},
  };
}

// A spread of rays: on-axis, moderate off-axis, wide, and behind-camera.
std::vector<Eigen::Vector3f> sampleRays()
{
  return {
    Eigen::Vector3f(0.0f, 0.0f, 1.0f).normalized(),
    Eigen::Vector3f(0.1f, 0.05f, 1.0f).normalized(),
    Eigen::Vector3f(-0.2f, 0.15f, 1.0f).normalized(),
    Eigen::Vector3f(0.5f, 0.3f, 1.0f).normalized(),
    Eigen::Vector3f(1.0f, 0.0f, 0.2f).normalized(),
    Eigen::Vector3f(0.0f, 0.0f, -1.0f).normalized(),  // behind camera
  };
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) tryMake acceptance / rejection matches validateCameraModel.
// ---------------------------------------------------------------------------

TEST(ValidatedCameraModel, TryMakeSucceedsForEveryValidModel)
{
  for (const auto &nm : allValidModels())
  {
    ASSERT_EQ(validateCameraModel(nm.model), StatusCode::OK) << nm.name;
    const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(nm.model);
    ASSERT_TRUE(vm.has_value()) << "tryMake failed for valid model " << nm.name;
    EXPECT_EQ(vm->projectionType(), nm.model.projection.type) << nm.name;
  }
}

TEST(ValidatedCameraModel, GetReturnsTheStoredModel)
{
  const CameraModel m = makePinhole(true);
  const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(m);
  ASSERT_TRUE(vm.has_value());

  const CameraModel &stored = vm->get();
  EXPECT_EQ(stored.intrinsics.fx, m.intrinsics.fx);
  EXPECT_EQ(stored.intrinsics.fy, m.intrinsics.fy);
  EXPECT_EQ(stored.intrinsics.cx, m.intrinsics.cx);
  EXPECT_EQ(stored.intrinsics.cy, m.intrinsics.cy);
  EXPECT_EQ(stored.projection.type, m.projection.type);
  EXPECT_EQ(stored.distortion.type, m.distortion.type);
  EXPECT_EQ(stored.distortion.count, m.distortion.count);
}

TEST(ValidatedCameraModel, TryMakeRejectsInvalidModels)
{
  // Default-constructed model has UNKNOWN projection type.
  EXPECT_FALSE(ValidatedCameraModel::tryMake(CameraModel{}).has_value());

  CameraModel zero_focal = makePinhole(false);
  zero_focal.intrinsics.fx = 0.0f;
  ASSERT_NE(validateCameraModel(zero_focal), StatusCode::OK);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(zero_focal).has_value());

  CameraModel non_finite = makePinhole(false);
  non_finite.intrinsics.cx = std::numeric_limits<float>::quiet_NaN();
  ASSERT_NE(validateCameraModel(non_finite), StatusCode::OK);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(non_finite).has_value());

  CameraModel bad_ds = makeDoubleSphere();
  bad_ds.projection.alpha = 1.5f;  // alpha must be in [0, 1]
  ASSERT_NE(validateCameraModel(bad_ds), StatusCode::OK);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(bad_ds).has_value());

  CameraModel pinhole_angle = makePinhole(false);
  pinhole_angle.distortion.type = DistortionModelType::EQUIDISTANT;
  pinhole_angle.distortion.space = DistortionSpace::ANGLE;
  ASSERT_NE(validateCameraModel(pinhole_angle), StatusCode::OK);
  EXPECT_FALSE(ValidatedCameraModel::tryMake(pinhole_angle).has_value());
}

// ---------------------------------------------------------------------------
// (2) Bit-for-bit equivalence with the generic free-function path.
// ---------------------------------------------------------------------------

TEST(ValidatedCameraModel, ForwardMatchesGenericExactly)
{
  for (const auto &nm : allValidModels())
  {
    const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(nm.model);
    ASSERT_TRUE(vm.has_value()) << nm.name;

    for (const auto &ray : sampleRays())
    {
      const PixelResult expected = rayToPixel(nm.model, ray);
      const PixelResult actual = vm->rayToPixel(ray);
      EXPECT_EQ(actual.status, expected.status) << nm.name;
      if (expected.status == StatusCode::OK)
      {
        EXPECT_EQ(actual.pixel.u, expected.pixel.u) << nm.name;
        EXPECT_EQ(actual.pixel.v, expected.pixel.v) << nm.name;
      }
    }
  }
}

TEST(ValidatedCameraModel, InverseMatchesGenericExactly)
{
  for (const auto &nm : allValidModels())
  {
    const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(nm.model);
    ASSERT_TRUE(vm.has_value()) << nm.name;

    // Build a set of valid pixels by projecting the forward-OK rays.
    for (const auto &ray : sampleRays())
    {
      const PixelResult px = rayToPixel(nm.model, ray);
      if (px.status != StatusCode::OK)
      {
        continue;
      }
      const RayResult expected = pixelToRay(nm.model, px.pixel);
      const RayResult actual = vm->pixelToRay(px.pixel);
      EXPECT_EQ(actual.status, expected.status) << nm.name;
      if (expected.status == StatusCode::OK)
      {
        EXPECT_EQ(actual.ray.direction.x(), expected.ray.direction.x()) << nm.name;
        EXPECT_EQ(actual.ray.direction.y(), expected.ray.direction.y()) << nm.name;
        EXPECT_EQ(actual.ray.direction.z(), expected.ray.direction.z()) << nm.name;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (3) Input-guard parity + overload consistency.
// ---------------------------------------------------------------------------

TEST(ValidatedCameraModel, RayToPixelOverloadsAgree)
{
  const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(makePinhole(true));
  ASSERT_TRUE(vm.has_value());

  const Eigen::Vector3f ray = Eigen::Vector3f(0.2f, -0.1f, 1.0f);
  const PixelResult from_vec = vm->rayToPixel(ray);
  const PixelResult from_scalars = vm->rayToPixel(ray.x(), ray.y(), ray.z());
  Ray3 ray3;
  ray3.direction = ray;
  const PixelResult from_ray3 = vm->rayToPixel(ray3);

  EXPECT_EQ(from_scalars.status, from_vec.status);
  EXPECT_EQ(from_scalars.pixel.u, from_vec.pixel.u);
  EXPECT_EQ(from_scalars.pixel.v, from_vec.pixel.v);
  EXPECT_EQ(from_ray3.status, from_vec.status);
  EXPECT_EQ(from_ray3.pixel.u, from_vec.pixel.u);
  EXPECT_EQ(from_ray3.pixel.v, from_vec.pixel.v);
}

TEST(ValidatedCameraModel, PixelToRayRejectsNonFiniteInputLikeGeneric)
{
  const CameraModel m = makePinhole(false);
  const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(m);
  ASSERT_TRUE(vm.has_value());

  const Pixel2 bad{std::numeric_limits<float>::quiet_NaN(), 100.0f};
  const RayResult generic = pixelToRay(m, bad);
  const RayResult wrapped = vm->pixelToRay(bad);
  EXPECT_EQ(generic.status, StatusCode::INVALID_INPUT);
  EXPECT_EQ(wrapped.status, StatusCode::INVALID_INPUT);

  // Scalar overload guards identically.
  EXPECT_EQ(
    vm->pixelToRay(std::numeric_limits<float>::infinity(), 5.0f).status, StatusCode::INVALID_INPUT
  );
}

TEST(ValidatedCameraModel, BehindCameraRayMatchesGeneric)
{
  const CameraModel m = makePinhole(false);
  const std::optional<ValidatedCameraModel> vm = ValidatedCameraModel::tryMake(m);
  ASSERT_TRUE(vm.has_value());

  const Eigen::Vector3f behind(0.0f, 0.0f, -1.0f);
  const PixelResult generic = rayToPixel(m, behind);
  const PixelResult wrapped = vm->rayToPixel(behind);
  EXPECT_EQ(generic.status, StatusCode::BEHIND_CAMERA);
  EXPECT_EQ(wrapped.status, generic.status);
}
