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

// Full-tier batch <-> scalar parity across every projection model.
//
// N = 71 = 8*8 + 4 + 3 so that on an AVX2 build the 8-lane main body runs
// eight full groups, the SSE tier runs one 4-lane group, and the scalar tail
// runs 3 elements — every dispatch level of the batch drivers actually
// executes. On a plain-SSE build the same N covers 17 SSE groups + tail; on
// scalar-only builds (no SIMD) the parity is trivially exact. Until this
// test, the suite's largest batch was n = 9, so the AVX2 main loops compiled
// on CI but processed a single group, and pixelToRayBatch64 had no caller at
// all.
//
// The ray/pixel sets deliberately mix valid, out-of-FOV, behind-camera,
// zero-norm, epsilon-gap and non-finite lanes: per-lane status must equal the
// scalar path exactly, valid_count must equal the scalar OK count, and OK
// pixels must match the scalar projection. For the inverse direction the
// SIMD kernels only guarantee a 1 px forward round-trip, so OK rays are
// verified by reprojection error rather than bitwise agreement.

#include "camxiom/batch.hpp"
#include "camxiom/batch64.hpp"
#include "camxiom/internal/constants.hpp"
#include "camxiom/model.hpp"
#include "camxiom/projection.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types.hpp"
#include "camxiom/types64.hpp"

#include <Eigen/Core>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace camxiom;

namespace
{

constexpr int kCount = 71;  // 8 AVX2 groups + 1 SSE group + 3 scalar tail

struct NamedModel
{
  const char *name;
  CameraModel model;
};

CameraModel makeBase(ProjectionModelType type, float fx)
{
  CameraModel m;
  m.intrinsics.fx = fx;
  m.intrinsics.fy = fx;
  m.intrinsics.cx = 320.0f;
  m.intrinsics.cy = 240.0f;
  m.projection.type = type;
  m.projection.theta_max = static_cast<float>(constants::kPi) - 1e-4f;
  m.distortion.type = DistortionModelType::NONE;
  m.distortion.space = DistortionSpace::NONE;
  m.distortion.count = 0U;
  return m;
}

std::vector<NamedModel> allModels()
{
  std::vector<NamedModel> models;

  CameraModel pinhole = makeBase(ProjectionModelType::PINHOLE, 500.0f);
  pinhole.projection.theta_max = static_cast<float>(constants::kHalfPi);
  pinhole.distortion.type = DistortionModelType::RADTAN5;
  pinhole.distortion.space = DistortionSpace::PLANE;
  pinhole.distortion.coeffs[0] = -0.05f;
  pinhole.distortion.coeffs[1] = 0.01f;
  pinhole.distortion.count = 5U;
  models.push_back({"pinhole+radtan5", pinhole});

  CameraModel fisheye = makeBase(ProjectionModelType::FISHEYE_THETA, 280.0f);
  fisheye.distortion.type = DistortionModelType::EQUIDISTANT;
  fisheye.distortion.space = DistortionSpace::ANGLE;
  models.push_back({"fisheye+equidistant", fisheye});

  // KB4 is the one family the aarch64 dispatch keeps on the SCALAR inverse
  // (simd_inverse_profitable == false in src/batch/float.cpp) — its presence
  // here exercises the profitability branch through every parity dimension,
  // including across OpenMP chunks in the large-N tests below. EQUIDISTANT
  // above is deliberately NOT in that opt-out list, so both sides of the
  // branch run.
  CameraModel kb4 = makeBase(ProjectionModelType::FISHEYE_THETA, 280.0f);
  kb4.distortion.type = DistortionModelType::KB4;
  kb4.distortion.space = DistortionSpace::ANGLE;
  kb4.distortion.count = 4U;
  kb4.distortion.coeffs[0] = 0.02f;
  kb4.distortion.coeffs[1] = -0.005f;
  kb4.distortion.coeffs[2] = 0.001f;
  kb4.distortion.coeffs[3] = -0.0002f;
  updateThetaMax(kb4);
  models.push_back({"fisheye+kb4", kb4});

  CameraModel omni = makeBase(ProjectionModelType::OMNIDIRECTIONAL, 400.0f);
  omni.projection.xi = 1.2f;
  models.push_back({"omnidirectional", omni});

  CameraModel ds = makeBase(ProjectionModelType::DOUBLE_SPHERE, 350.0f);
  ds.projection.xi = -0.2308f;
  ds.projection.alpha = 0.5741f;
  models.push_back({"double_sphere", ds});

  CameraModel eucm = makeBase(ProjectionModelType::EUCM, 350.0f);
  eucm.projection.alpha = 0.6f;
  eucm.projection.beta = 1.1f;
  models.push_back({"eucm", eucm});

  return models;
}

// 64 grid rays spanning ~[-63, 63] deg plus 7 special lanes (behind, zero,
// epsilon-gap, NaN, ultra-wide, grazing) — models disagree on which are
// valid, which is exactly what the per-lane parity must reproduce.
std::vector<Eigen::Vector3f> makeTestRays()
{
  std::vector<Eigen::Vector3f> rays;
  rays.reserve(kCount);
  for (int gy = 0; gy < 8; ++gy)
  {
    for (int gx = 0; gx < 8; ++gx)
    {
      const float x = -2.0f + 4.0f * static_cast<float>(gx) / 7.0f;
      const float y = -1.5f + 3.0f * static_cast<float>(gy) / 7.0f;
      rays.emplace_back(x, y, 1.0f);
    }
  }
  rays.emplace_back(0.0f, 0.0f, -1.0f);          // straight back
  rays.emplace_back(0.0f, 0.0f, 0.0f);           // zero norm
  rays.emplace_back(1e-4f, 0.0f, 1e-4f);         // epsilon gap
  rays.emplace_back(std::nanf(""), 0.0f, 1.0f);  // non-finite
  rays.emplace_back(2.0f, 0.0f, -0.3f);          // theta ~99 deg
  rays.emplace_back(0.0f, 3.0f, 0.1f);           // theta ~88 deg
  rays.emplace_back(-1.0f, -1.0f, 0.05f);        // grazing
  return rays;
}

// 64 grid pixels inside the 640x480 image plus 7 special lanes.
std::vector<Eigen::Vector2f> makeTestPixels()
{
  std::vector<Eigen::Vector2f> px;
  px.reserve(kCount);
  for (int gy = 0; gy < 8; ++gy)
  {
    for (int gx = 0; gx < 8; ++gx)
    {
      px.emplace_back(
        40.0f + 560.0f * static_cast<float>(gx) / 7.0f,
        30.0f + 420.0f * static_cast<float>(gy) / 7.0f
      );
    }
  }
  px.emplace_back(320.0f, 240.0f);         // principal point
  px.emplace_back(-500.0f, 240.0f);        // far outside (left)
  px.emplace_back(5000.0f, 5000.0f);       // far outside (bottom-right)
  px.emplace_back(std::nanf(""), 240.0f);  // non-finite
  px.emplace_back(0.0f, 0.0f);             // image corner
  px.emplace_back(639.0f, 479.0f);         // opposite corner
  px.emplace_back(320.0f, -80.0f);         // above the image
  return px;
}

}  // namespace

TEST(BatchParity, RayToPixelBatchMatchesScalar)
{
  const auto rays = makeTestRays();
  ASSERT_EQ(static_cast<int>(rays.size()), kCount);

  for (const NamedModel &nm : allModels())
  {
    ASSERT_EQ(validateCameraModel(nm.model), StatusCode::OK) << nm.name;

    float rays_xyz[3 * kCount];
    for (int i = 0; i < kCount; ++i)
    {
      rays_xyz[3 * i + 0] = rays[static_cast<std::size_t>(i)].x();
      rays_xyz[3 * i + 1] = rays[static_cast<std::size_t>(i)].y();
      rays_xyz[3 * i + 2] = rays[static_cast<std::size_t>(i)].z();
    }
    float u_out[kCount];
    float v_out[kCount];
    StatusCode statuses[kCount];
    const int valid = rayToPixelBatch(nm.model, rays_xyz, kCount, u_out, v_out, statuses);

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const PixelResult scalar = rayToPixel(nm.model, rays[static_cast<std::size_t>(i)]);
      ASSERT_EQ(statuses[i], scalar.status)
        << nm.name << " lane " << i << ": batch status " << toString(statuses[i]) << " vs scalar "
        << toString(scalar.status);
      if (scalar.status == StatusCode::OK)
      {
        ++expected_valid;
        EXPECT_NEAR(u_out[i], scalar.pixel.u, 1e-2f) << nm.name << " lane " << i;
        EXPECT_NEAR(v_out[i], scalar.pixel.v, 1e-2f) << nm.name << " lane " << i;
      }
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, PixelToRayBatchMatchesScalar)
{
  const auto pixels = makeTestPixels();
  ASSERT_EQ(static_cast<int>(pixels.size()), kCount);

  for (const NamedModel &nm : allModels())
  {
    ASSERT_EQ(validateCameraModel(nm.model), StatusCode::OK) << nm.name;

    float u_in[kCount];
    float v_in[kCount];
    for (int i = 0; i < kCount; ++i)
    {
      u_in[i] = pixels[static_cast<std::size_t>(i)].x();
      v_in[i] = pixels[static_cast<std::size_t>(i)].y();
    }
    float dirs[3 * kCount];
    StatusCode statuses[kCount];
    const int valid = pixelToRayBatch(nm.model, u_in, v_in, kCount, dirs, statuses);

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const RayResult scalar = pixelToRay(nm.model, Pixel2{u_in[i], v_in[i]});
      ASSERT_EQ(statuses[i], scalar.status)
        << nm.name << " lane " << i << ": batch status " << toString(statuses[i]) << " vs scalar "
        << toString(scalar.status);
      if (scalar.status != StatusCode::OK)
      {
        continue;
      }
      ++expected_valid;

      // The SIMD inverse only guarantees a 1 px forward round-trip, so OK
      // rays are validated by reprojecting the batch direction rather than
      // by bitwise agreement with the scalar solve.
      const Eigen::Vector3f dir(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
      const PixelResult back = rayToPixel(nm.model, dir);
      ASSERT_EQ(back.status, StatusCode::OK) << nm.name << " lane " << i;
      EXPECT_NEAR(back.pixel.u, u_in[i], 1.5f) << nm.name << " lane " << i;
      EXPECT_NEAR(back.pixel.v, v_in[i], 1.5f) << nm.name << " lane " << i;
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, LargeBatchParallelChunkingMatchesScalar)
{
  // Batches above the OpenMP chunking threshold (8192) run the SIMD kernels
  // in per-thread chunks (src/batch/batch_parallel.hpp). Verify the chunked
  // path agrees with the scalar path lane-for-lane and that valid_count
  // survives the reduction. N is deliberately NOT a multiple of the chunk
  // rounding (8) so the final partial chunk is exercised too.
  constexpr int kBig = 20011;
  const auto seed_rays = makeTestRays();

  for (const NamedModel &nm : allModels())
  {
    std::vector<float> rays_xyz(3 * kBig);
    for (int i = 0; i < kBig; ++i)
    {
      const Eigen::Vector3f &r = seed_rays[static_cast<std::size_t>(i) % seed_rays.size()];
      rays_xyz[3 * i + 0] = r.x();
      rays_xyz[3 * i + 1] = r.y();
      rays_xyz[3 * i + 2] = r.z();
    }
    std::vector<float> u_out(kBig);
    std::vector<float> v_out(kBig);
    std::vector<StatusCode> statuses(kBig);
    const int valid =
      rayToPixelBatch(nm.model, rays_xyz.data(), kBig, u_out.data(), v_out.data(), statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kBig; ++i)
    {
      const Eigen::Vector3f ray(rays_xyz[3 * i], rays_xyz[3 * i + 1], rays_xyz[3 * i + 2]);
      const PixelResult scalar = rayToPixel(nm.model, ray);
      ASSERT_EQ(statuses[static_cast<std::size_t>(i)], scalar.status) << nm.name << " lane " << i;
      if (scalar.status == StatusCode::OK)
      {
        ++expected_valid;
        ASSERT_NEAR(u_out[static_cast<std::size_t>(i)], scalar.pixel.u, 1e-2f)
          << nm.name << " lane " << i;
        ASSERT_NEAR(v_out[static_cast<std::size_t>(i)], scalar.pixel.v, 1e-2f)
          << nm.name << " lane " << i;
      }
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, LargeBatchParallelChunkingInverseMatchesScalar)
{
  // Inverse counterpart of the forward chunking test above. The inverse
  // carries materially more branching under chunking (Newton early-exit,
  // SIMD-forward status verification, per-model scalar-only fallbacks and
  // the aarch64 profitability gating) — until this test it never actually
  // forked into more than one OpenMP chunk.
  constexpr int kBig = 20011;
  const auto seed_pixels = makeTestPixels();

  for (const NamedModel &nm : allModels())
  {
    std::vector<float> u_in(kBig);
    std::vector<float> v_in(kBig);
    for (int i = 0; i < kBig; ++i)
    {
      const Eigen::Vector2f &p = seed_pixels[static_cast<std::size_t>(i) % seed_pixels.size()];
      u_in[static_cast<std::size_t>(i)] = p.x();
      v_in[static_cast<std::size_t>(i)] = p.y();
    }
    std::vector<float> dirs(3 * kBig);
    std::vector<StatusCode> statuses(kBig);
    const int valid =
      pixelToRayBatch(nm.model, u_in.data(), v_in.data(), kBig, dirs.data(), statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kBig; ++i)
    {
      const std::size_t s = static_cast<std::size_t>(i);
      const RayResult scalar = pixelToRay(nm.model, Pixel2{u_in[s], v_in[s]});
      ASSERT_EQ(statuses[s], scalar.status) << nm.name << " lane " << i;
      if (scalar.status != StatusCode::OK)
      {
        continue;
      }
      ++expected_valid;
      const Eigen::Vector3f dir(dirs[3 * s], dirs[3 * s + 1], dirs[3 * s + 2]);
      const PixelResult back = rayToPixel(nm.model, dir);
      ASSERT_EQ(back.status, StatusCode::OK) << nm.name << " lane " << i;
      ASSERT_NEAR(back.pixel.u, u_in[s], 1.5f) << nm.name << " lane " << i;
      ASSERT_NEAR(back.pixel.v, v_in[s], 1.5f) << nm.name << " lane " << i;
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, RayToPixelBatch64MatchesScalar)
{
  const auto rays = makeTestRays();

  for (const NamedModel &nm : allModels())
  {
    const CameraModel64 m64 = toCameraModel64(nm.model);
    ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK) << nm.name;

    Eigen::Matrix3Xd R(3, kCount);
    for (int i = 0; i < kCount; ++i)
    {
      R.col(i) = rays[static_cast<std::size_t>(i)].cast<double>();
    }
    Eigen::Matrix2Xd P(2, kCount);
    std::vector<StatusCode> statuses(kCount);
    const int valid = rayToPixelBatch64(m64, R, P, statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const PixelResult64 scalar = rayToPixel64(m64, R.col(i));
      ASSERT_EQ(statuses[static_cast<std::size_t>(i)], scalar.status) << nm.name << " lane " << i;
      if (scalar.status == StatusCode::OK)
      {
        ++expected_valid;
        EXPECT_NEAR(P(0, i), scalar.pixel.u, 1e-6) << nm.name << " lane " << i;
        EXPECT_NEAR(P(1, i), scalar.pixel.v, 1e-6) << nm.name << " lane " << i;
      }
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, RayToPixelBatch64RawMatchesScalar)
{
  // The raw-pointer forward driver hosts the double SIMD hooks (AVX2 on
  // x86, 2-wide NEON on aarch64) that the Eigen variant above never reaches
  // — cover it per lane with the same mixed valid/edge ray set.
  const auto rays = makeTestRays();

  for (const NamedModel &nm : allModels())
  {
    const CameraModel64 m64 = toCameraModel64(nm.model);
    ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK) << nm.name;

    double rays_xyz[3 * kCount];
    for (int i = 0; i < kCount; ++i)
    {
      const Eigen::Vector3d r = rays[static_cast<std::size_t>(i)].cast<double>();
      rays_xyz[3 * i + 0] = r.x();
      rays_xyz[3 * i + 1] = r.y();
      rays_xyz[3 * i + 2] = r.z();
    }
    double u_out[kCount];
    double v_out[kCount];
    std::vector<StatusCode> statuses(kCount);
    const int valid = rayToPixelBatch64(m64, rays_xyz, kCount, u_out, v_out, statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const Eigen::Vector3d ray(rays_xyz[3 * i], rays_xyz[3 * i + 1], rays_xyz[3 * i + 2]);
      const PixelResult64 scalar = rayToPixel64(m64, ray);
      ASSERT_EQ(statuses[static_cast<std::size_t>(i)], scalar.status) << nm.name << " lane " << i;
      if (scalar.status == StatusCode::OK)
      {
        ++expected_valid;
        EXPECT_NEAR(u_out[i], scalar.pixel.u, 1e-6) << nm.name << " lane " << i;
        EXPECT_NEAR(v_out[i], scalar.pixel.v, 1e-6) << nm.name << " lane " << i;
      }
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, PixelToRayBatch64EigenMatchesScalar)
{
  // The Eigen (Matrix2Xd -> Matrix3Xd) inverse overload is a distinct driver
  // from the raw-pointer one below and had zero callers anywhere in the
  // suite. Cover it lane-for-lane, including the zeroing of non-OK columns.
  const auto pixels = makeTestPixels();

  for (const NamedModel &nm : allModels())
  {
    const CameraModel64 m64 = toCameraModel64(nm.model);
    ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK) << nm.name;

    Eigen::Matrix2Xd P(2, kCount);
    for (int i = 0; i < kCount; ++i)
    {
      P(0, i) = static_cast<double>(pixels[static_cast<std::size_t>(i)].x());
      P(1, i) = static_cast<double>(pixels[static_cast<std::size_t>(i)].y());
    }
    Eigen::Matrix3Xd D = Eigen::Matrix3Xd::Constant(3, kCount, 7.0);
    std::vector<StatusCode> statuses(kCount);
    const int valid = pixelToRayBatch64(m64, P, D, statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const RayResult64 scalar = pixelToRay64(m64, Pixel2d{P(0, i), P(1, i)});
      ASSERT_EQ(statuses[static_cast<std::size_t>(i)], scalar.status) << nm.name << " lane " << i;
      if (scalar.status != StatusCode::OK)
      {
        EXPECT_TRUE(D.col(i).isZero()) << nm.name << " lane " << i;
        continue;
      }
      ++expected_valid;
      EXPECT_LT((D.col(i) - scalar.ray.direction).norm(), 1e-9) << nm.name << " lane " << i;
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, PixelToRayBatch64EigenRejectsColumnMismatch)
{
  // Shape-guard path of the Eigen inverse overload: mismatched column counts
  // must return -1, mark every lane INVALID_INPUT and zero the output.
  const CameraModel64 m64 = toCameraModel64(allModels()[0].model);

  Eigen::Matrix2Xd P = Eigen::Matrix2Xd::Constant(2, 4, 100.0);
  Eigen::Matrix3Xd D = Eigen::Matrix3Xd::Constant(3, 5, 7.0);
  StatusCode statuses[4];
  EXPECT_EQ(pixelToRayBatch64(m64, P, D, statuses), -1);
  for (int i = 0; i < 4; ++i)
  {
    EXPECT_EQ(statuses[i], StatusCode::INVALID_INPUT) << "lane " << i;
  }
  EXPECT_TRUE(D.isZero());
}

TEST(BatchParity, PixelToRayBatch64MatchesScalar)
{
  // First-ever coverage of the double inverse batch: it had no caller in the
  // suite at all.
  const auto pixels = makeTestPixels();

  for (const NamedModel &nm : allModels())
  {
    const CameraModel64 m64 = toCameraModel64(nm.model);
    ASSERT_EQ(validateCameraModel64(m64), StatusCode::OK) << nm.name;

    double u_in[kCount];
    double v_in[kCount];
    for (int i = 0; i < kCount; ++i)
    {
      u_in[i] = static_cast<double>(pixels[static_cast<std::size_t>(i)].x());
      v_in[i] = static_cast<double>(pixels[static_cast<std::size_t>(i)].y());
    }
    double dirs[3 * kCount];
    std::vector<StatusCode> statuses(kCount);
    const int valid = pixelToRayBatch64(m64, u_in, v_in, kCount, dirs, statuses.data());

    int expected_valid = 0;
    for (int i = 0; i < kCount; ++i)
    {
      const RayResult64 scalar = pixelToRay64(m64, Pixel2d{u_in[i], v_in[i]});
      ASSERT_EQ(statuses[static_cast<std::size_t>(i)], scalar.status) << nm.name << " lane " << i;
      if (scalar.status != StatusCode::OK)
      {
        continue;
      }
      ++expected_valid;
      const Eigen::Vector3d dir(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
      EXPECT_LT((dir - scalar.ray.direction).norm(), 1e-9) << nm.name << " lane " << i;
    }
    EXPECT_EQ(valid, expected_valid) << nm.name;
  }
}

TEST(BatchParity, LargeBatchParallelChunking64MatchesScalar)
{
  // Double raw-pointer drivers above the OpenMP chunking threshold — in
  // particular the 2-wide NEON pinhole forward kernel, which no other test
  // runs across more than one chunk. Forward asserts exact per-lane pixel
  // agreement; the inverse (scalar Newton per lane in double) asserts
  // direction agreement against the scalar solve.
  constexpr int kBig = 20011;
  const auto seed_rays = makeTestRays();
  const auto seed_pixels = makeTestPixels();

  for (const NamedModel &nm : allModels())
  {
    const CameraModel64 m64 = toCameraModel64(nm.model);

    // Forward.
    std::vector<double> rays_xyz(3 * kBig);
    for (int i = 0; i < kBig; ++i)
    {
      const Eigen::Vector3d r =
        seed_rays[static_cast<std::size_t>(i) % seed_rays.size()].cast<double>();
      rays_xyz[3 * static_cast<std::size_t>(i) + 0] = r.x();
      rays_xyz[3 * static_cast<std::size_t>(i) + 1] = r.y();
      rays_xyz[3 * static_cast<std::size_t>(i) + 2] = r.z();
    }
    std::vector<double> u_out(kBig);
    std::vector<double> v_out(kBig);
    std::vector<StatusCode> fwd_statuses(kBig);
    const int fwd_valid = rayToPixelBatch64(
      m64, rays_xyz.data(), kBig, u_out.data(), v_out.data(), fwd_statuses.data()
    );

    int fwd_expected = 0;
    for (int i = 0; i < kBig; ++i)
    {
      const std::size_t s = static_cast<std::size_t>(i);
      const Eigen::Vector3d ray(rays_xyz[3 * s], rays_xyz[3 * s + 1], rays_xyz[3 * s + 2]);
      const PixelResult64 scalar = rayToPixel64(m64, ray);
      ASSERT_EQ(fwd_statuses[s], scalar.status) << nm.name << " lane " << i;
      if (scalar.status == StatusCode::OK)
      {
        ++fwd_expected;
        ASSERT_NEAR(u_out[s], scalar.pixel.u, 1e-6) << nm.name << " lane " << i;
        ASSERT_NEAR(v_out[s], scalar.pixel.v, 1e-6) << nm.name << " lane " << i;
      }
    }
    EXPECT_EQ(fwd_valid, fwd_expected) << nm.name;

    // Inverse.
    std::vector<double> u_in(kBig);
    std::vector<double> v_in(kBig);
    for (int i = 0; i < kBig; ++i)
    {
      const Eigen::Vector2f &p = seed_pixels[static_cast<std::size_t>(i) % seed_pixels.size()];
      u_in[static_cast<std::size_t>(i)] = static_cast<double>(p.x());
      v_in[static_cast<std::size_t>(i)] = static_cast<double>(p.y());
    }
    std::vector<double> dirs(3 * kBig);
    std::vector<StatusCode> inv_statuses(kBig);
    const int inv_valid =
      pixelToRayBatch64(m64, u_in.data(), v_in.data(), kBig, dirs.data(), inv_statuses.data());

    int inv_expected = 0;
    for (int i = 0; i < kBig; ++i)
    {
      const std::size_t s = static_cast<std::size_t>(i);
      const RayResult64 scalar = pixelToRay64(m64, Pixel2d{u_in[s], v_in[s]});
      ASSERT_EQ(inv_statuses[s], scalar.status) << nm.name << " lane " << i;
      if (scalar.status != StatusCode::OK)
      {
        continue;
      }
      ++inv_expected;
      const Eigen::Vector3d dir(dirs[3 * s], dirs[3 * s + 1], dirs[3 * s + 2]);
      ASSERT_LT((dir - scalar.ray.direction).norm(), 1e-9) << nm.name << " lane " << i;
    }
    EXPECT_EQ(inv_valid, inv_expected) << nm.name;
  }
}

TEST(BatchParity, RawPointerGuardsRejectBadInputs)
{
  // Defensive branches of all four raw-pointer drivers, previously untested:
  // count < 0 -> -1, count == 0 -> 0, and a required nullptr -> -1 with the
  // non-null outputs zeroed and every lane INVALID_INPUT.
  const CameraModel model = allModels()[0].model;
  const CameraModel64 m64 = toCameraModel64(model);

  float frays[12] = {};
  float fu[4] = {};
  float fv[4] = {};
  float fdirs[12] = {};
  double drays[12] = {};
  double du[4] = {};
  double dv[4] = {};
  double ddirs[12] = {};
  StatusCode st[4];

  EXPECT_EQ(rayToPixelBatch(model, frays, -1, fu, fv, st), -1);
  EXPECT_EQ(rayToPixelBatch(model, frays, 0, fu, fv, st), 0);
  EXPECT_EQ(pixelToRayBatch(model, fu, fv, -1, fdirs, st), -1);
  EXPECT_EQ(pixelToRayBatch(model, fu, fv, 0, fdirs, st), 0);
  EXPECT_EQ(rayToPixelBatch64(m64, drays, -1, du, dv, st), -1);
  EXPECT_EQ(rayToPixelBatch64(m64, drays, 0, du, dv, st), 0);
  EXPECT_EQ(pixelToRayBatch64(m64, du, dv, -1, ddirs, st), -1);
  EXPECT_EQ(pixelToRayBatch64(m64, du, dv, 0, ddirs, st), 0);

  for (StatusCode &s : st)
  {
    s = StatusCode::OK;
  }
  fu[0] = 7.0f;
  fv[0] = 7.0f;
  EXPECT_EQ(rayToPixelBatch(model, nullptr, 4, fu, fv, st), -1);
  EXPECT_EQ(fu[0], 0.0f);
  EXPECT_EQ(fv[0], 0.0f);
  for (const StatusCode s : st)
  {
    EXPECT_EQ(s, StatusCode::INVALID_INPUT);
  }
  fdirs[0] = 7.0f;
  EXPECT_EQ(pixelToRayBatch(model, nullptr, fv, 4, fdirs, st), -1);
  EXPECT_EQ(fdirs[0], 0.0f);

  du[0] = 7.0;
  EXPECT_EQ(rayToPixelBatch64(m64, nullptr, 4, du, dv, st), -1);
  EXPECT_EQ(du[0], 0.0);
  ddirs[0] = 7.0;
  EXPECT_EQ(pixelToRayBatch64(m64, nullptr, dv, 4, ddirs, st), -1);
  EXPECT_EQ(ddirs[0], 0.0);
}

TEST(BatchParity, EigenOverloadsRejectColumnMismatch)
{
  // Column-count mismatch between input and output: -1, zeroed output,
  // INVALID_INPUT per lane. Covers the float forward/inverse and double
  // forward variants (the double inverse variant is tested above).
  const CameraModel model = allModels()[0].model;
  const CameraModel64 m64 = toCameraModel64(model);
  StatusCode st[4];

  Eigen::Matrix3Xf R = Eigen::Matrix3Xf::Constant(3, 4, 0.5f);
  Eigen::Matrix2Xf P = Eigen::Matrix2Xf::Constant(2, 5, 7.0f);
  EXPECT_EQ(rayToPixelBatch(model, R, P, st), -1);
  EXPECT_TRUE(P.isZero());
  for (const StatusCode s : st)
  {
    EXPECT_EQ(s, StatusCode::INVALID_INPUT);
  }

  Eigen::Matrix2Xf Pin = Eigen::Matrix2Xf::Constant(2, 4, 100.0f);
  Eigen::Matrix3Xf D = Eigen::Matrix3Xf::Constant(3, 5, 7.0f);
  EXPECT_EQ(pixelToRayBatch(model, Pin, D, st), -1);
  EXPECT_TRUE(D.isZero());

  Eigen::Matrix3Xd Rd = Eigen::Matrix3Xd::Constant(3, 4, 0.5);
  Eigen::Matrix2Xd Pd = Eigen::Matrix2Xd::Constant(2, 5, 7.0);
  EXPECT_EQ(rayToPixelBatch64(m64, Rd, Pd, st), -1);
  EXPECT_TRUE(Pd.isZero());
}

// ---------------------------------------------------------------------------
// No-status fast path vs theta_max cap
// ---------------------------------------------------------------------------

TEST(BatchParity, NoStatusFastPathHonoursThetaMaxCap)
{
  // With statuses_out == nullptr and default SolverOptions the batch inverse
  // takes the no-verify fast path: it trusts the SIMD valid mask alone, with
  // no forward round-trip to catch discrepancies. The OMNI / DS / EUCM SIMD
  // kernels used to skip the scalar path's theta_max cap there, so a capped
  // model handed out beyond-FOV directions as "valid" — and with no status
  // array the caller had nothing to flag them with. Pixels are generated by
  // forward projection through an uncapped twin so each lane's expected side
  // of the cap is known and kept >= 0.15 rad clear of the boundary.
  std::vector<NamedModel> capped_models;
  for (const NamedModel &nm : allModels())
  {
    const ProjectionModelType t = nm.model.projection.type;
    if (t == ProjectionModelType::OMNIDIRECTIONAL || t == ProjectionModelType::DOUBLE_SPHERE || t == ProjectionModelType::EUCM)
    {
      capped_models.push_back(nm);
    }
  }
  ASSERT_EQ(capped_models.size(), 3U);

  constexpr float kThetaCap = 1.0f;
  const float inside_thetas[] = {0.2f, 0.4f, 0.6f, 0.8f};
  const float outside_thetas[] = {1.15f, 1.3f, 1.45f, 1.6f};
  const float phis[] = {0.3f, 1.87f, 3.44f, 5.01f};

  for (const NamedModel &nm : capped_models)
  {
    SCOPED_TRACE(nm.name);

    const CameraModel &wide = nm.model;  // theta_max = pi - 1e-4 (uncapped)
    CameraModel capped = nm.model;
    capped.projection.theta_max = kThetaCap;
    ASSERT_EQ(validateCameraModel(capped), StatusCode::OK);

    std::vector<float> us, vs;
    std::vector<bool> expect_inside;
    int inside_count = 0;
    int outside_count = 0;
    auto add_theta = [&](const float theta, const bool inside) {
      for (const float phi : phis)
      {
        const Eigen::Vector3f dir(
          std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)
        );
        const PixelResult fwd = rayToPixel(wide, dir);
        if (fwd.status != StatusCode::OK)
        {
          continue;  // outside the uncapped model's own domain — skip
        }
        us.push_back(fwd.pixel.u);
        vs.push_back(fwd.pixel.v);
        expect_inside.push_back(inside);
        (inside ? inside_count : outside_count) += 1;
      }
    };
    for (const float theta : inside_thetas) add_theta(theta, true);
    for (const float theta : outside_thetas) add_theta(theta, false);

    const int n = static_cast<int>(us.size());
    ASSERT_GE(inside_count, 4);
    ASSERT_GE(outside_count, 4) << "cap-exceeding pixels must actually exist";
    ASSERT_GE(n, 8);

    // Scalar ground truth on the capped model: inside lanes unproject OK,
    // outside lanes are OUT_OF_FOV via the scalar theta_max round-trip cap.
    std::vector<RayResult> scalar(n);
    int scalar_ok = 0;
    for (int i = 0; i < n; ++i)
    {
      scalar[i] = pixelToRay(capped, Pixel2{us[i], vs[i]});
      ASSERT_EQ(scalar[i].status == StatusCode::OK, expect_inside[i])
        << "scalar status disagrees with construction at lane " << i;
      if (scalar[i].status == StatusCode::OK) ++scalar_ok;
    }

    // Fast path: no statuses, default options.
    std::vector<float> dirs(
      3U * static_cast<std::size_t>(n), std::numeric_limits<float>::quiet_NaN()
    );
    const int valid = pixelToRayBatch(capped, us.data(), vs.data(), n, dirs.data());
    EXPECT_EQ(valid, scalar_ok);

    for (int i = 0; i < n; ++i)
    {
      const Eigen::Vector3f batch_dir(dirs[3 * i], dirs[3 * i + 1], dirs[3 * i + 2]);
      if (expect_inside[i])
      {
        EXPECT_LT((batch_dir - scalar[i].ray.direction).norm(), 1e-4f)
          << "direction mismatch at lane " << i;
      }
      else
      {
        // Invalid lanes must be zeroed, not populated with a beyond-FOV
        // direction the caller cannot distinguish from a valid one.
        EXPECT_TRUE(batch_dir.isZero()) << "beyond-cap lane " << i << " leaked a direction";
      }
    }
  }
}
