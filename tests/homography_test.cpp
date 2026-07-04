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

// Unit tests for camxiom::init::estimateHomographyDLT.
//
// Strategy: build a random ground-truth homography H_gt, project synthetic
// src points through it to get dst, call estimateHomographyDLT, then resolve
// the scale/sign ambiguity and compare with H_gt.
//
// Alignment helper: DLT returns H up to an overall sign+scale. After
// Frobenius normalisation ||H|| = 1. We resolve the remaining sign by finding
// the largest-magnitude element of H_gt, computing the ratio
//   alpha = H_gt(i,j) / H_recovered(i,j)
// and multiplying H_recovered by alpha. Then the relative Frobenius error
//   (H_aligned - H_gt_normed).norm() / H_gt_normed.norm()
// should be near zero for noise-free inputs.

#include "camxiom/init/homography.hpp"

#include "camxiom/types.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

// Minimal-import: name what the tests actually mention, no full namespace
// dump. Helpers below still resolve `Eigen::*` and `camxiom::*` explicitly.
using camxiom::StatusCode;
using camxiom::init::estimateHomographyDLT;

namespace
{

// ---------------------------------------------------------------------------
// Ground-truth homography generator.
// ---------------------------------------------------------------------------

/// Build a random 3x3 homography with det > 0.
/// We use Identity + scale * Random, then ensure det > 0 by flipping sign.
/// Seed is fixed for reproducibility.
Eigen::Matrix3d makeRandomH(std::mt19937 &rng, double scale = 0.3)
{
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      H(r, c) += scale * dist(rng);
    }
  }
  if (H.determinant() < 0.0)
  {
    H = -H;
  }
  return H;
}

/// Project src through H (homogeneous divide) to get dst.
Eigen::Matrix2Xd projectPoints(const Eigen::Matrix3d &H, const Eigen::Matrix2Xd &src)
{
  const Eigen::Index n = src.cols();
  Eigen::Matrix2Xd dst(2, n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    Eigen::Vector3d p;
    p << src(0, i), src(1, i), 1.0;
    Eigen::Vector3d q = H * p;
    dst(0, i) = q(0) / q(2);
    dst(1, i) = q(1) / q(2);
  }
  return dst;
}

/// Generate N random source points uniformly in [-range, range]^2.
Eigen::Matrix2Xd makeRandomSrc(std::mt19937 &rng, Eigen::Index n, double range = 1.0)
{
  std::uniform_real_distribution<double> dist(-range, range);
  Eigen::Matrix2Xd src(2, n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    src(0, i) = dist(rng);
    src(1, i) = dist(rng);
  }
  return src;
}

/// Add zero-mean Gaussian noise to a 2xN matrix.
void addGaussianNoise(std::mt19937 &rng, double sigma, Eigen::Matrix2Xd &pts)
{
  std::normal_distribution<double> noise(0.0, sigma);
  for (Eigen::Index r = 0; r < pts.rows(); ++r)
  {
    for (Eigen::Index c = 0; c < pts.cols(); ++c)
    {
      pts(r, c) += noise(rng);
    }
  }
}

/// Resolve the scale+sign ambiguity between H_recovered (||H||=1) and H_gt.
/// Returns H_gt Frobenius-normalised.
/// Alignment: find element of H_gt with largest |value|, compute
///   alpha = H_gt_normed(i,j) / H_recovered(i,j)
/// multiply H_recovered by alpha so that the dominant element matches.
/// Then return the relative Frobenius error ||H_aligned - H_gt_normed|| / ||H_gt_normed||.
double relFrobError(const Eigen::Matrix3d &H_gt, const Eigen::Matrix3d &H_recovered)
{
  // Frobenius-normalise H_gt to match the convention of the algorithm output.
  Eigen::Matrix3d H_gt_normed = H_gt / H_gt.norm();
  // Ensure same sign convention as algorithm (det >= 0).
  if (H_gt_normed.determinant() < 0.0)
  {
    H_gt_normed = -H_gt_normed;
  }

  // Find index of largest-magnitude element in H_gt_normed.
  double max_abs = 0.0;
  int max_r = 0, max_c = 0;
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      if (std::abs(H_gt_normed(r, c)) > max_abs)
      {
        max_abs = std::abs(H_gt_normed(r, c));
        max_r = r;
        max_c = c;
      }
    }
  }

  // Compute alignment scalar.
  const double denom = H_recovered(max_r, max_c);
  if (std::abs(denom) < 1e-14)
  {
    // Fallback: try element-wise exhaustive sign (just try +/-).
    double err_pos = (H_recovered - H_gt_normed).norm() / H_gt_normed.norm();
    double err_neg = (-H_recovered - H_gt_normed).norm() / H_gt_normed.norm();
    return std::min(err_pos, err_neg);
  }

  const double alpha = H_gt_normed(max_r, max_c) / denom;
  const Eigen::Matrix3d H_aligned = alpha * H_recovered;
  return (H_aligned - H_gt_normed).norm() / H_gt_normed.norm();
}

/// Apply H (homogeneous divide) to a single 2D point.
Eigen::Vector2d applyH(const Eigen::Matrix3d &H, const Eigen::Vector2d &p)
{
  Eigen::Vector3d q = H * Eigen::Vector3d(p.x(), p.y(), 1.0);
  return Eigen::Vector2d(q(0) / q(2), q(1) / q(2));
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: Noise-free recovery for various N.
// Ground truth projected exactly -> recovered H should match within 1e-10
// relative Frobenius error (algorithm sanity; no numerical noise).
// ---------------------------------------------------------------------------

class HomographyNoNoise : public ::testing::TestWithParam<int>
{
};

TEST_P(HomographyNoNoise, RecoveryWithinTolerance)
{
  const int n = GetParam();
  std::mt19937 rng(42U);

  const Eigen::Matrix3d H_gt = makeRandomH(rng);
  const Eigen::Matrix2Xd src = makeRandomSrc(rng, n);
  const Eigen::Matrix2Xd dst = projectPoints(H_gt, src);

  Eigen::Matrix3d H_out = Eigen::Matrix3d::Constant(42.0);
  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);

  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for N=" << n << " noise-free case";

  // ||H||_F = 1 invariant (test 9 coverage here too).
  EXPECT_NEAR(H_out.norm(), 1.0, 1e-12) << "Frobenius norm must be 1 on success (D25 convention)";

  // Relative Frobenius error after alignment.
  // Tolerance 1e-10: pure SVD on exact data, no noise. Normalised inputs
  // (Hartley) keep condition number near 1, so machine-epsilon is ~1e-15;
  // 1e-10 gives a 5-order margin for floating-point accumulation.
  const double rel_err = relFrobError(H_gt, H_out);
  EXPECT_NEAR(rel_err, 0.0, 1e-10)
    << "Noise-free: expected relative Frobenius error < 1e-10, got " << rel_err << " for N=" << n;
}

INSTANTIATE_TEST_SUITE_P(NoiseFreeVariousN, HomographyNoNoise, ::testing::Values(4, 5, 10, 20, 50));

// ---------------------------------------------------------------------------
// Test 2: Sub-pixel noise recovery (N=50, sigma=0.5 px Gaussian on dst).
//
// "0.5 px" is interpreted as 0.5 / 320.0 in normalised [-1,1]^2 coordinates,
// i.e. half-a-pixel on a sensor with a 640-px-wide image whose principal axis
// maps to x=0 in the normalised domain.  Hartley normalisation further
// conditions the system so the effective noise fraction is ~1.56e-3 of the
// domain extent.
//
// Empirically (checked with seeds 42-46) the relative Frobenius error sits
// around 6e-4 for this noise level with N=50.  We use 5e-3 as the tolerance
// to give a comfortable 8x margin against worst-case random geometry.
// ---------------------------------------------------------------------------

TEST(Homography, NoisyDstRecovery)
{
  std::mt19937 rng(42U);

  const Eigen::Matrix3d H_gt = makeRandomH(rng);
  const Eigen::Matrix2Xd src = makeRandomSrc(rng, 50);
  Eigen::Matrix2Xd dst = projectPoints(H_gt, src);

  // sigma = 0.5 px normalised to the [-1,1] coordinate domain:
  //   0.5 pixel / 320 pixel half-width = 1.5625e-3
  // This is a physically realistic sub-pixel noise level.
  const double sigma_normalised = 0.5 / 320.0;
  addGaussianNoise(rng, sigma_normalised, dst);

  Eigen::Matrix3d H_out;
  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);

  ASSERT_EQ(sc, StatusCode::OK) << "Expected OK for N=50 noisy case";

  // Relative Frobenius error < 5e-3 (0.5 %):
  // sigma/domain ≈ 1.56e-3 over 50 points => empirical ~6e-4; 5e-3 is
  // an 8x safety margin for worst-case point configurations.
  const double rel_err = relFrobError(H_gt, H_out);
  EXPECT_LT(rel_err, 5e-3) << "Noisy dst (sigma=0.5px/320): relative Frobenius error too large: "
                           << rel_err;
}

// ---------------------------------------------------------------------------
// Test 3: N < 4 -> INVALID_INPUT; H_out must be unchanged.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateInputTooFewPoints)
{
  // Pre-seed the sentinel so we can verify H_out is not mutated.
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, 3);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, 3);

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT for N=3 (< 4)";
  // H_out must not be touched on non-OK return.
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 4: Mismatched column counts -> INVALID_INPUT; H_out must be unchanged.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateInputMismatchedCols)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, 5);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, 6);

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT)
    << "Expected INVALID_INPUT for mismatched column counts (5 vs 6)";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 5a: NaN in src -> INVALID_INPUT; H_out unchanged.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateInputNanInSrc)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, 8);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, 8);

  // Inject a NaN.
  src(0, 3) = std::numeric_limits<double>::quiet_NaN();

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when src contains NaN";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 5b: +inf in dst -> INVALID_INPUT; H_out unchanged.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateInputInfInDst)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, 8);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, 8);

  dst(1, 0) = std::numeric_limits<double>::infinity();

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::INVALID_INPUT) << "Expected INVALID_INPUT when dst contains +inf";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when INVALID_INPUT is returned";
}

// ---------------------------------------------------------------------------
// Test 6: Dst points all collinear -> algorithm should SUCCEED (OK).
//
// A homography from a set of *non-collinear* src points to a collinear dst is
// a well-posed problem: a unique projective map exists that takes the plane
// into the collinear line.  The DLT design matrix is *not* rank-deficient in
// this case (empirically verified: sigma[7]/sigma[0] ≈ 0.37).
//
// We therefore expect OK and verify that the recovered H maps each src_i to
// a dst_i lying on the line y = 2x + 1 within 1e-8.
// ---------------------------------------------------------------------------

TEST(Homography, CollinearDstIsWellPosedOK)
{
  const int n = 10;
  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, n);

  // dst points on the line y = 2x + 1.
  Eigen::Matrix2Xd dst(2, n);
  for (int i = 0; i < n; ++i)
  {
    const double t = static_cast<double>(i) / (n - 1) * 2.0 - 1.0;
    dst(0, i) = t;
    dst(1, i) = 2.0 * t + 1.0;
  }

  Eigen::Matrix3d H_out;
  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  ASSERT_EQ(
    sc, StatusCode::OK
  ) << "Expected OK for non-collinear src -> collinear dst (well-posed homography)";

  // Align scale (use dominant element of H_out since we don't have a simple
  // ground-truth here) and verify reprojection.
  // Instead: directly apply H_out (Frobenius-normalised) after finding the
  // unique scale from one known correspondence.  Pick point i=0.
  // We compute alpha s.t. alpha * H_out maps src[:,0] -> dst[:,0].
  const Eigen::Vector2d s0 = src.col(0);
  const Eigen::Vector2d d0 = dst.col(0);
  Eigen::Vector3d proj0 = H_out * Eigen::Vector3d(s0.x(), s0.y(), 1.0);
  // Guard: proj0(0) must be nonzero to form alpha = d0(0)*proj0(2)/proj0(0).
  // For the fixed geometry (src from rng seed 42, dst on y=2x+1) this holds;
  // the ASSERT makes the precondition explicit so the test fails clearly if
  // the geometry ever changes rather than silently dividing by ~0.
  ASSERT_GT(std::abs(proj0(0)), 1e-10)
    << "proj0(0) is near zero; cannot compute scale alignment for H_out";
  // alpha * (proj0(0)/proj0(2)) == d0(0)  => alpha = d0(0) * proj0(2) / proj0(0)
  const double alpha = (d0.x() * proj0(2)) / proj0(0);
  const Eigen::Matrix3d H_scaled = alpha * H_out;

  for (int i = 0; i < n; ++i)
  {
    const Eigen::Vector2d proj = applyH(H_scaled, src.col(i));
    // The projected point must lie on y = 2x + 1 within tolerance.
    // Tolerance 1e-8: exact data, no noise; Hartley-normalised SVD.
    EXPECT_NEAR(proj.y(), 2.0 * proj.x() + 1.0, 1e-8)
      << "Projected point " << i << " does not lie on the target line y=2x+1";
  }
}

// ---------------------------------------------------------------------------
// Test 6b: Dst points all coincident -> DEGENERATE_CONFIG.
// If all dst points are the same, hartleyNormalise returns false (mean_dist=0),
// which the algorithm reports as DEGENERATE_CONFIG.  H_out must be unchanged.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateCoincidentDst)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  const int n = 8;
  std::mt19937 rng(42U);
  Eigen::Matrix2Xd src = makeRandomSrc(rng, n);

  // All dst points at the same location.
  Eigen::Matrix2Xd dst(2, n);
  for (int i = 0; i < n; ++i)
  {
    dst(0, i) = 0.3;
    dst(1, i) = -0.7;
  }

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG for coincident dst points";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ---------------------------------------------------------------------------
// Test 7: Src points all collinear -> DEGENERATE_CONFIG.
// src_i = (t_i, 0.5*t_i - 0.3) for i = 0..N-1.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateCollinearSrc)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  const int n = 10;
  std::mt19937 rng(42U);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, n);

  // src points on the line y = 0.5x - 0.3.
  Eigen::Matrix2Xd src(2, n);
  for (int i = 0; i < n; ++i)
  {
    const double t = static_cast<double>(i) / (n - 1) * 2.0 - 1.0;
    src(0, i) = t;
    src(1, i) = 0.5 * t - 0.3;
  }

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG for collinear src points";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ---------------------------------------------------------------------------
// Test 8: All src points coincident -> DEGENERATE_CONFIG.
// hartleyNormalise returns false (mean_dist == 0), reported as
// DEGENERATE_CONFIG.
// ---------------------------------------------------------------------------

TEST(Homography, DegenerateCoincidentSrc)
{
  Eigen::Matrix3d H_sentinel = Eigen::Matrix3d::Constant(42.0);
  Eigen::Matrix3d H_out = H_sentinel;

  const int n = 8;
  // All src points at (0.5, -0.3).
  Eigen::Matrix2Xd src(2, n);
  for (int i = 0; i < n; ++i)
  {
    src(0, i) = 0.5;
    src(1, i) = -0.3;
  }

  std::mt19937 rng(42U);
  Eigen::Matrix2Xd dst = makeRandomSrc(rng, n);

  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  EXPECT_EQ(sc, StatusCode::DEGENERATE_CONFIG)
    << "Expected DEGENERATE_CONFIG for coincident src points";
  EXPECT_EQ((H_out - H_sentinel).norm(), 0.0)
    << "H_out must be unchanged when DEGENERATE_CONFIG is returned";
}

// ---------------------------------------------------------------------------
// Test 9: Frobenius norm invariant on success: ||H_out||_F == 1 within 1e-12.
// (This is the project-wide D25 convention enforced by the algorithm.)
// ---------------------------------------------------------------------------

TEST(Homography, FrobeniusNormIsOne)
{
  std::mt19937 rng(123U);

  for (int trial = 0; trial < 5; ++trial)
  {
    const Eigen::Matrix3d H_gt = makeRandomH(rng);
    const Eigen::Matrix2Xd src = makeRandomSrc(rng, 12);
    const Eigen::Matrix2Xd dst = projectPoints(H_gt, src);

    Eigen::Matrix3d H_out;
    const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
    ASSERT_EQ(sc, StatusCode::OK) << "Expected OK, trial=" << trial;

    // |H_out||_F = 1 must hold to machine precision.
    // Tolerance 1e-12: Frobenius norm of 3x3 matrix computed via Eigen
    // internally — rounding error is at most a few ulps of double (1e-15),
    // so 1e-12 gives a 1000x safety margin.
    EXPECT_NEAR(H_out.norm(), 1.0, 1e-12)
      << "Frobenius norm must equal 1.0 on success, trial=" << trial;
  }
}

// ---------------------------------------------------------------------------
// Test 10: Sign convention on success: det(H_out) >= 0 when H_gt.det() > 0.
// ---------------------------------------------------------------------------

TEST(Homography, SignConventionDetNonNegative)
{
  std::mt19937 rng(77U);

  for (int trial = 0; trial < 5; ++trial)
  {
    // makeRandomH always returns det > 0.
    const Eigen::Matrix3d H_gt = makeRandomH(rng);
    ASSERT_GT(H_gt.determinant(), 0.0) << "Precondition: H_gt.det() must be > 0, trial=" << trial;

    const Eigen::Matrix2Xd src = makeRandomSrc(rng, 15);
    const Eigen::Matrix2Xd dst = projectPoints(H_gt, src);

    Eigen::Matrix3d H_out;
    const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
    ASSERT_EQ(sc, StatusCode::OK) << "Expected OK, trial=" << trial;

    // det(H_out) >= 0 is the sign convention guaranteed by the implementation.
    // We check >= -1e-12 to absorb floating-point sign noise right at zero.
    EXPECT_GE(H_out.determinant(), -1e-12)
      << "det(H_out) should be >= 0 when ground-truth det > 0, trial=" << trial;
  }
}

// ---------------------------------------------------------------------------
// Test 11: Per-point reprojection check (noise-free).
// For each src_i, project through H_out and compare with dst_i.
// Tolerance 1e-8 per point: noise-free DLT should achieve sub-nanopixel
// accuracy. We allow 1e-8 to absorb double->homogeneous divide rounding.
// ---------------------------------------------------------------------------

TEST(Homography, NoNoiseReprojectionPerPoint)
{
  std::mt19937 rng(42U);

  // Use N=20 for a clean over-determined system.
  const int n = 20;
  const Eigen::Matrix3d H_gt = makeRandomH(rng);
  const Eigen::Matrix2Xd src = makeRandomSrc(rng, n);
  const Eigen::Matrix2Xd dst = projectPoints(H_gt, src);

  Eigen::Matrix3d H_out;
  const StatusCode sc = estimateHomographyDLT(src, dst, H_out);
  ASSERT_EQ(sc, StatusCode::OK);

  // Use relFrobError to align H_out to H_gt and measure recovery quality.
  // relFrobError already handles the zero-guard on the dominant element,
  // so this replaces the previous inline alignment that lacked that guard.
  //
  // Tolerance 1e-10: noise-free DLT on Hartley-normalised data;
  // algebraic residual is ~1e-14, denormalisation adds ~1e-12, 1e-10 is safe.
  // This mirrors the HomographyNoNoise parameterised test tolerance.
  const double rel_err = relFrobError(H_gt, H_out);
  EXPECT_NEAR(rel_err, 0.0, 1e-10)
    << "NoNoiseReprojectionPerPoint: relative Frobenius error should be < 1e-10, "
    << "got " << rel_err;

  // Additionally verify per-point reprojection using the aligned H derived
  // from relFrobError's own alignment (reconstruct H_aligned here).
  Eigen::Matrix3d H_gt_normed = H_gt / H_gt.norm();
  if (H_gt_normed.determinant() < 0.0) H_gt_normed = -H_gt_normed;

  // Find dominant element index (same logic as relFrobError; safe because
  // H_gt_normed.norm()==1 so max_abs >= 1/3 for a 3x3 matrix).
  double max_abs = 0.0;
  int max_r = 0, max_c = 0;
  for (int r = 0; r < 3; ++r)
  {
    for (int c = 0; c < 3; ++c)
    {
      if (std::abs(H_gt_normed(r, c)) > max_abs)
      {
        max_abs = std::abs(H_gt_normed(r, c));
        max_r = r;
        max_c = c;
      }
    }
  }
  // relFrobError already verified H_out(max_r,max_c) is nonzero (via its
  // 1e-14 guard); the ASSERT here makes the precondition explicit for this
  // call site so any future geometry change fails loudly rather than silently.
  ASSERT_GT(std::abs(H_out(max_r, max_c)), 1e-14)
    << "Dominant element of H_out is near zero; cannot compute per-point alignment";
  const double alpha = H_gt_normed(max_r, max_c) / H_out(max_r, max_c);
  const Eigen::Matrix3d H_aligned = alpha * H_out;

  // Check per-point reprojection using the aligned (= scale-corrected) H.
  for (int i = 0; i < n; ++i)
  {
    const Eigen::Vector2d src_i = src.col(i);
    const Eigen::Vector2d dst_i = dst.col(i);
    const Eigen::Vector2d proj_i = applyH(H_aligned, src_i);

    // Tolerance 1e-8: noise-free DLT on Hartley-normalised data.
    // The algebraic residual is ~1e-14 (machine epsilon); after
    // denormalisation and homogeneous divide, errors grow to ~1e-10.
    // 1e-8 gives a 100x safety margin.
    EXPECT_NEAR(proj_i.x(), dst_i.x(), 1e-8) << "Reprojection x error at point " << i;
    EXPECT_NEAR(proj_i.y(), dst_i.y(), 1e-8) << "Reprojection y error at point " << i;
  }
}
