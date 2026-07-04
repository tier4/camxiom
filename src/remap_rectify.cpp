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

#include "remap_rectify.hpp"

#include "camxiom/projection.hpp"
#include "camxiom/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

// Rectify-target focal computation (LOADMAP R1), extracted verbatim from
// remap.cpp. The three file-local helpers below are used only by
// computeRectifyFocalPairs; they were previously in remap.cpp's anonymous
// namespace. Numerics are unchanged.

namespace camxiom::detail
{
namespace
{

// Find the outermost source-pixel along the line from (cx_src, cy_src) to
// (u_outer, v_outer) for which the validity predicate holds.
//
// Validity predicate: pixelToRay returns OK AND, after rotating the ray
// into the output frame, the z component is at least `z_clamp_threshold`
// (i.e. the ray is forward-facing relative to the output projection).
//
// For pinhole sources, the raster boundary itself is typically valid ->
// returns {u_outer, v_outer} on the first try.  For fisheye / omni / DS /
// EUCM sources whose raster boundary lies outside the projection's valid
// theta region, OR whose corresponding output-frame ray is back-facing
// (z < z_clamp_threshold, as happens for hemispherical+ omni sources),
// this walks inward by bisection (up to 24 iterations, sub-pixel
// resolution) and returns the boundary of the valid region in the
// direction of (u_outer, v_outer).
//
// Returns std::nullopt only if even the source-pixel centre is invalid
// (extremely degenerate).
std::optional<Pixel2> findOutermostValidAlongLine(
  const CameraModel &src_model, const Eigen::Matrix3f &R_out_from_src, float z_clamp_threshold,
  float u_outer, float v_outer
)
{
  auto isValid = [&](float u, float v) -> bool {
    const RayResult rr = pixelToRay(src_model, Pixel2{u, v});
    if (rr.status != StatusCode::OK) return false;
    const Eigen::Vector3f ray_out = R_out_from_src * rr.ray.direction;
    return ray_out.z() >= z_clamp_threshold;
  };

  // Quick check: raster boundary itself is valid.
  if (isValid(u_outer, v_outer))
  {
    return Pixel2{u_outer, v_outer};
  }

  const float cx_src = src_model.intrinsics.cx;
  const float cy_src = src_model.intrinsics.cy;

  // Confirm centre is valid; if not, no useful boundary exists along this line.
  if (!isValid(cx_src, cy_src))
  {
    return std::nullopt;
  }

  // Bisection: p_inner is always valid; p is always invalid.
  float u_inner = cx_src, v_inner = cy_src;
  float u = u_outer, v = v_outer;
  for (int k = 0; k < 24; ++k)
  {
    const float u_mid = 0.5f * (u_inner + u);
    const float v_mid = 0.5f * (v_inner + v);
    if (isValid(u_mid, v_mid))
    {
      u_inner = u_mid;
      v_inner = v_mid;
    }
    else
    {
      u = u_mid;
      v = v_mid;
    }
  }
  return Pixel2{u_inner, v_inner};
}

// Forward, fold-over-robust inscribed half-extent along one output axis.
//
// The inscribed rectangle must be computed WITHOUT inverting the source
// distortion: a polynomial (plumb_bob / RADTAN) distortion is non-monotonic
// past its fold-over radius, and pixelToRay near that radius returns a
// converged-but-wrong-branch ray (e.g. a left-edge pixel mapping to a positive
// x_n), which silently corrupts the inscribed bounds. Instead we drive from the
// OUTPUT side and only ever FORWARD-project (rayToPixel), which every model can
// always evaluate.
//
// Marches the output normalised coordinate t outward from 0 along one axis
// (`axis` 0 => vary x at y=0; 1 => vary y at x=0) in the output frame. Each
// output ray (xn, yn, 1) is rotated into the source frame via
// `src_from_output` and forward-projected. The march stops at the source raster
// edge (`edge_target`; the output edge then sits on the source boundary -- the
// inscribed limit) or where the forward-projected coordinate stops moving
// outward (the source distortion has folded and can represent nothing further).
// Returns the signed normalised extent at the stop. x is taken at y=0 and y at
// x=0, so fx and fy stay independent (D45).
float forwardInscribedExtent(
  const CameraModel &src_model, const Eigen::Matrix3f &src_from_output, int axis, float sign,
  float edge_target, float center_coord
)
{
  constexpr float kStep = 0.0005f;    // normalised-coordinate march step.
  constexpr float kMaxExtent = 4.0f;  // generous cap (> any sane pinhole FOV).
  float prev_coord = center_coord;
  float best_t = 0.0f;
  for (float a = kStep; a <= kMaxExtent; a += kStep)
  {
    const float xn = (axis == 0) ? sign * a : 0.0f;
    const float yn = (axis == 1) ? sign * a : 0.0f;
    Eigen::Vector3f ray_out(xn, yn, 1.0f);
    ray_out.normalize();
    const Eigen::Vector3f ray_src = src_from_output * ray_out;
    const PixelResult px = rayToPixel(src_model, ray_src);
    if (px.status != StatusCode::OK) break;  // source cannot represent this ray.
    const float coord = (axis == 0) ? px.pixel.u : px.pixel.v;
    // Require the forward coordinate to keep moving outward; if it reverses the
    // distortion has folded over and nothing beyond is representable.
    if (sign > 0.0f && coord < prev_coord) break;
    if (sign < 0.0f && coord > prev_coord) break;
    best_t = sign * a;
    if (sign > 0.0f && coord >= edge_target) break;  // reached the source edge.
    if (sign < 0.0f && coord <= edge_target) break;
    prev_coord = coord;
  }
  return best_t;
}

// Whether a centred output pinhole (fx, fy) maps its ENTIRE boundary -- the 4
// corners and edge samples -- inside the source raster [margin, w-1-margin]
// under forward projection. The per-axis centre-line extents bound the output
// edges but NOT the corners (a corner ray has a larger radius than either
// centre-line and, for a wide source, can exceed the source's valid region), so
// the inscribed FOV is shrunk until this returns true. Forward only (the output
// pinhole inverse is the closed-form (u-cx)/fx), so a folding source cannot
// corrupt it.
bool outputBoundaryInsideSource(
  const CameraModel &src_model, const Eigen::Matrix3f &src_from_output, const ImageSize &src_size,
  const ImageSize &out_size, float margin, float fx, float fy
)
{
  const float u_lo = margin;
  const float u_hi = static_cast<float>(src_size.width) - 1.0f - margin;
  const float v_lo = margin;
  const float v_hi = static_cast<float>(src_size.height) - 1.0f - margin;
  const float w = static_cast<float>(out_size.width);
  const float h = static_cast<float>(out_size.height);
  const float cx = (w - 1.0f) * 0.5f;
  const float cy = (h - 1.0f) * 0.5f;
  constexpr int kPerEdge = 32;

  auto check = [&](float u, float v) -> bool {
    Eigen::Vector3f ray_out((u - cx) / fx, (v - cy) / fy, 1.0f);
    ray_out.normalize();
    const PixelResult px = rayToPixel(src_model, src_from_output * ray_out);
    if (px.status != StatusCode::OK) return false;
    return px.pixel.u >= u_lo && px.pixel.u <= u_hi && px.pixel.v >= v_lo && px.pixel.v <= v_hi;
  };

  for (int i = 0; i <= kPerEdge; ++i)
  {
    const float u = (w - 1.0f) * static_cast<float>(i) / static_cast<float>(kPerEdge);
    if (!check(u, 0.0f) || !check(u, h - 1.0f)) return false;
  }
  for (int i = 1; i < kPerEdge; ++i)
  {
    const float v = (h - 1.0f) * static_cast<float>(i) / static_cast<float>(kPerEdge);
    if (!check(0.0f, v) || !check(w - 1.0f, v)) return false;
  }
  return true;
}

}  // namespace

// Single source-boundary pass producing both inscribed and circumscribed
// per-axis focal estimates.
//
// For each sampled source boundary pixel p (on top / bottom / left / right
// edge of the source raster):
//   ray_src = pixelToRay(src_model, p)
//   ray_out = R_out_from_src * ray_src.direction
//   skip   if ray_out.z() < sin(theta_clamp_margin)  (rays >= ~pi/2)
//   x_n    = ray_out.x() / ray_out.z()
//   y_n    = ray_out.y() / ray_out.z()
//
// Inscribed (per edge of the source boundary):
//   LEFT  -> x_n is on the negative side; tighten x_left_max  (largest x_n)
//   RIGHT -> x_n is on the positive side; tighten x_right_min (smallest x_n)
//   TOP   -> y_n is on the negative side; tighten y_top_max
//   BOTTOM-> y_n is on the positive side; tighten y_bot_min
//   fx_inscribed = (out_w - 1) / (x_right_min - x_left_max)
//   fy_inscribed = (out_h - 1) / (y_bot_min   - y_top_max)
//
// Circumscribed (overall extremes across all edges):
//   fx_circumscribed = (out_w - 1) / (x_max - x_min)
//   fy_circumscribed = (out_h - 1) / (y_max - y_min)
//
// Returns a struct with `_valid` flags set when the corresponding bound is
// well-defined. Each caller checks the flag for the policy it cares about.
RectifyFocalPairs computeRectifyFocalPairs(
  const CameraModel &src_model, const ImageSize &src_size, const ImageSize &out_size,
  const Eigen::Matrix3f &src_from_output_rotation, int boundary_sample_count, float source_margin_px
)
{
  constexpr float kThetaClampMargin = 0.05f;  // ~2.86 deg
  const float kZClampThreshold = std::sin(kThetaClampMargin);

  // R_out_from_src is the inverse of src_from_output (orthogonal => transpose).
  const Eigen::Matrix3f R_out_from_src = src_from_output_rotation.transpose();

  const float sw = static_cast<float>(src_size.width);
  const float sh = static_cast<float>(src_size.height);
  const float out_w = static_cast<float>(out_size.width);
  const float out_h = static_cast<float>(out_size.height);

  const float margin = source_margin_px;
  const float u_lo = margin;
  const float u_hi = sw - 1.0f - margin;
  const float v_lo = margin;
  const float v_hi = sh - 1.0f - margin;

  const int n = std::max(boundary_sample_count, 8);
  const int per_edge = n / 4;

  // Circumscribed extremes across all edges.
  float x_min = +std::numeric_limits<float>::infinity();
  float x_max = -std::numeric_limits<float>::infinity();
  float y_min = +std::numeric_limits<float>::infinity();
  float y_max = -std::numeric_limits<float>::infinity();

  int circumscribed_sample_count = 0;

  // Circumscribed (alpha = 1 / ALL_SOURCE_CONTAINED): overall extent of the
  // source boundary in the output normalised plane. This still lifts the source
  // boundary via the inverse (pixelToRay), bisecting inward to the outermost
  // valid sample; it is correct for non-folding sources and legitimately
  // unavailable for a source whose boundary folds over (the far corners are not
  // representable). The inscribed bound below is computed FORWARD, so the
  // alpha = 0 path never depends on this inverse pass.
  auto considerPixel = [&](float u_src, float v_src) {
    const auto sample_opt =
      findOutermostValidAlongLine(src_model, R_out_from_src, kZClampThreshold, u_src, v_src);
    if (!sample_opt) return;  // No valid boundary along this line.
    const RayResult ray = pixelToRay(src_model, sample_opt.value());
    if (ray.status != StatusCode::OK) return;  // Defensive (should not happen).

    const Eigen::Vector3f ray_out = R_out_from_src * ray.ray.direction;
    if (ray_out.z() < kZClampThreshold) return;  // Defensive (helper guarantees this).

    const float x_n = ray_out.x() / ray_out.z();
    const float y_n = ray_out.y() / ray_out.z();

    if (x_n < x_min) x_min = x_n;
    if (x_n > x_max) x_max = x_n;
    if (y_n < y_min) y_min = y_n;
    if (y_n > y_max) y_max = y_n;
    ++circumscribed_sample_count;
  };

  // Top + bottom edges (full u-range), then left + right edges (corners already
  // covered by the top / bottom passes).
  for (int i = 0; i <= per_edge; ++i)
  {
    const float u =
      (per_edge > 0) ? (u_lo + (u_hi - u_lo) * static_cast<float>(i) / static_cast<float>(per_edge))
                     : u_lo;
    considerPixel(u, v_lo);
    considerPixel(u, v_hi);
  }
  for (int i = 1; i < per_edge; ++i)
  {
    const float v = v_lo + (v_hi - v_lo) * static_cast<float>(i) / static_cast<float>(per_edge);
    considerPixel(u_lo, v);
    considerPixel(u_hi, v);
  }

  RectifyFocalPairs result{};

  // Circumscribed validity: at least one ray survived; width/height positive.
  if (circumscribed_sample_count > 0 && std::isfinite(x_min) && std::isfinite(x_max) && std::isfinite(y_min) && std::isfinite(y_max) && (x_max > x_min) && (y_max > y_min))
  {
    result.fx_circumscribed = (out_w - 1.0f) / (x_max - x_min);
    result.fy_circumscribed = (out_h - 1.0f) / (y_max - y_min);
    result.circumscribed_valid = true;
  }

  // Inscribed (alpha = 0): FORWARD, fold-over robust, fx/fy independent. Each
  // output axis is marched outward and forward-projected through the source (no
  // source inversion, so a folding distortion cannot corrupt the bound),
  // stopping at the source raster edge or the fold. fx comes from the
  // horizontal span (y = 0), fy from the vertical span (x = 0).
  const float cx_src = src_model.intrinsics.cx;
  const float cy_src = src_model.intrinsics.cy;
  const float x_left =
    forwardInscribedExtent(src_model, src_from_output_rotation, 0, -1.0f, u_lo, cx_src);
  const float x_right =
    forwardInscribedExtent(src_model, src_from_output_rotation, 0, +1.0f, u_hi, cx_src);
  const float y_top =
    forwardInscribedExtent(src_model, src_from_output_rotation, 1, -1.0f, v_lo, cy_src);
  const float y_bot =
    forwardInscribedExtent(src_model, src_from_output_rotation, 1, +1.0f, v_hi, cy_src);

  if (std::isfinite(x_left) && std::isfinite(x_right) && std::isfinite(y_top) && std::isfinite(y_bot) && (x_right > x_left) && (y_bot > y_top))
  {
    // Per-axis centre-line focals set the fx/fy aspect; they bound the output
    // edges but not the corners. Shrink the FOV uniformly (preserving the
    // aspect, so fx/fy stay independent) until the full output boundary --
    // corners included -- maps inside the source.
    const float fx0 = (out_w - 1.0f) / (x_right - x_left);
    const float fy0 = (out_h - 1.0f) / (y_bot - y_top);

    float s_hi = 1.0f;
    bool fits = outputBoundaryInsideSource(
      src_model, src_from_output_rotation, src_size, out_size, margin, fx0, fy0
    );
    if (!fits)
    {
      for (int k = 0; k < 24; ++k)
      {
        s_hi *= 1.5f;
        if (outputBoundaryInsideSource(
              src_model, src_from_output_rotation, src_size, out_size, margin, fx0 * s_hi,
              fy0 * s_hi
            ))
        {
          fits = true;
          break;
        }
      }
    }
    if (fits)
    {
      // Binary search the smallest scale (the widest corner-valid FOV).
      float lo = (s_hi == 1.0f) ? 1.0f : s_hi / 1.5f;
      float hi = s_hi;
      for (int k = 0; k < 28; ++k)
      {
        const float mid = 0.5f * (lo + hi);
        if (outputBoundaryInsideSource(
              src_model, src_from_output_rotation, src_size, out_size, margin, fx0 * mid, fy0 * mid
            ))
        {
          hi = mid;
        }
        else
        {
          lo = mid;
        }
      }
      // Extra 0.1% FOV shrink for FP-rounding headroom at the corners.
      constexpr float kInscribedFocalSafetyMargin = 1.001f;
      result.fx_inscribed = fx0 * hi * kInscribedFocalSafetyMargin;
      result.fy_inscribed = fy0 * hi * kInscribedFocalSafetyMargin;
      result.inscribed_valid = true;
    }
  }

  return result;
}

}  // namespace camxiom::detail
