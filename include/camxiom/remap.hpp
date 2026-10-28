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

#ifndef CAMXIOM__REMAP_HPP
#define CAMXIOM__REMAP_HPP

#include "camxiom/types.hpp"

namespace camxiom
{

/// Plain aggregate like every other result struct (types.hpp policy); build
/// with brace init: RemapResult{status, valid_count, total_count}.
struct [[nodiscard]] RemapResult
{
  StatusCode status{StatusCode::OK};
  int valid_count{0};
  int total_count{0};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

/// Build remap maps for image undistortion / re-projection.
///
/// For each pixel (u_dst, v_dst) in the destination image, compute the
/// corresponding source pixel (u_src, v_src) such that:
///   dst(v_dst, u_dst) = src(map_y[v_dst * width + u_dst], map_x[v_dst * width + u_dst])
///
/// The mapping is: dst_pixel → pixelToRay(dst_model) → rayToPixel(src_model).
///
/// @param src_model   Camera model of the source (distorted) image.
/// @param dst_model   Camera model of the destination (rectified) image.
///                    Use a distortion-free copy of src_model for simple undistortion.
/// @param width       Destination image width in pixels.
/// @param height      Destination image height in pixels.
/// @param map_x       Output buffer for x-coordinates, size = width * height.
/// @param map_y       Output buffer for y-coordinates, size = width * height.
/// @param solver_options  Solver options for pixelToRay inverse projection.
/// @return RemapResult with status + valid/total pixel counts.
///         `status == OK` means the map build itself succeeded (invalid destination pixels are
///         represented by -1 in map_x/map_y). `status != OK` means invalid input/model and output
///         should be treated as unusable.
RemapResult buildRemapMap(
  const CameraModel &src_model, const CameraModel &dst_model, int width, int height, float *map_x,
  float *map_y, const SolverOptions &solver_options = SolverOptions{}
);

/// Same-projection undistort: removes distortion but keeps the projection model.
///
/// For non-pinhole models (fisheye, omnidirectional, DS, EUCM), this does NOT
/// produce a rectilinear (perspective) image. Use buildRectifyRemapMap() with a
/// pinhole output template for that.
RemapResult buildUndistortRemapMap(
  const CameraModel &src_model, int width, int height, float *map_x, float *map_y,
  const SolverOptions &solver_options = SolverOptions{}
);

// ===========================================================================
// Image-aware remap
// ===========================================================================

enum class InvalidPixelPolicy : std::uint8_t { WRITE_NEGATIVE_ONE = 0U, WRITE_RAW_COORDINATE };

struct ImageRemapOptions
{
  Eigen::Matrix3f src_from_dst_rotation{Eigen::Matrix3f::Identity()};
  bool require_source_in_bounds{true};
  InvalidPixelPolicy invalid_pixel_policy{InvalidPixelPolicy::WRITE_NEGATIVE_ONE};
  SolverOptions solver_options{};
};

struct [[nodiscard]] ImageRemapResult
{
  StatusCode status{StatusCode::OK};
  int model_valid_count{0};
  int source_in_bounds_count{0};
  int total_count{0};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

/// Image-aware remap: dst_pixel → ray → rotate → src_pixel, with source bounds checking.
///
/// Unlike buildRemapMap(), this function:
///   - Applies an optional rotation between src and dst coordinate frames.
///   - Checks whether the projected source pixel falls within [0, src_width) × [0, src_height).
///   - Reports model_valid_count and source_in_bounds_count separately.
///
/// @param src_model   Camera model of the source image.
/// @param src_size    Source image dimensions (for bounds checking). Must be
///                    valid (positive) when require_source_in_bounds is set:
///                    an invalid size is rejected with INVALID_INPUT rather
///                    than silently skipping the advertised check.
/// @param dst_model   Camera model of the destination image.
/// @param dst_size    Destination image dimensions (output map size).
/// @param map_x       Output buffer for x-coordinates, size = dst_size.width * dst_size.height.
/// @param map_y       Output buffer for y-coordinates, size = dst_size.width * dst_size.height.
/// @param options     Remap options (rotation, bounds policy, solver).
ImageRemapResult buildImageRemapMap(
  const CameraModel &src_model, ImageSize src_size, const CameraModel &dst_model,
  ImageSize dst_size, float *map_x, float *map_y,
  const ImageRemapOptions &options = ImageRemapOptions{}
);

/// Same-projection undistort (image-aware variant).
///
/// dst_model = makeDistortionFree(src_model), rotation = I, src_size = dst_size = image_size.
ImageRemapResult buildUndistortImageRemapMap(
  const CameraModel &src_model, ImageSize image_size, float *map_x, float *map_y,
  const SolverOptions &solver_options = SolverOptions{}
);

// ===========================================================================
// Field of view
// ===========================================================================

struct [[nodiscard]] FovResult
{
  StatusCode status{StatusCode::OK};
  float horizontal_fov_deg{0.0f};
  float vertical_fov_deg{0.0f};
  float diagonal_fov_deg{0.0f};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

/// Field of view of `model` over an image of `image_size`, in degrees.
///
/// Works for ANY camera model. For pinhole, the fisheye laws,
/// omnidirectional, double sphere and EUCM, each angle is measured between
/// the unprojected rays of two boundary pixels (pixelToRay), the same
/// computation the rectify APIs use to fill their *_fov_deg fields.
///   * horizontal: (0, cy) .. (width-1, cy) along the principal-point row
///   * vertical:   (cx, 0) .. (cx, height-1) along the principal-point column
///   * diagonal:   (0, 0) .. (width-1, height-1)
///
/// For the lat-long projections (EQUIRECTANGULAR, CYLINDRICAL), whose field
/// can exceed 180 degrees, the values are angular spans of valid pixels,
/// edge to edge, clipped to the sphere:
///   * horizontal: the longitude span of the principal-point row, from the
///     left edge of column 0 (u = -0.5) to the right edge of the last column
///   * vertical:   the latitude span of the principal-point column (u = cx),
///     top edge to bottom edge. With skew that column's longitude changes
///     from row to row; rows where it passes the seam are not counted.
/// A full equirectangular image reports 360 x 180. A lat-long model has no
/// diagonal angle; diagonal_fov_deg is 0.
///
/// Returns INVALID_INPUT for an invalid size and the validation status for an
/// invalid model. When a boundary pixel cannot be unprojected (e.g. outside
/// the model's theta_max), that single component is reported as 0.0 while the
/// overall status stays OK — matching the rectify FOV field behaviour.
FovResult computeFov(
  const CameraModel &model, ImageSize image_size,
  const SolverOptions &solver_options = SolverOptions{}
);

// ===========================================================================
// Rectification: fit an output camera to the source + remap
// ===========================================================================

/// How to build the output camera of a rectification.
///
/// The output camera is `output_template` scaled to fit the source: the
/// template's projection (type, xi / alpha / beta, theta_max) and pixel shape
/// (the ratios fx : fy : skew) are kept, one common scale s multiplies fx, fy
/// and skew, and the principal point is put at the image centre
/// ((W - 1) / 2, (H - 1) / 2). The template's cx and cy are not used, and its
/// distortion must be NONE. Any projection type works, the latitude-longitude
/// ones included; a fisheye-law template keeps its theta_max, which bounds the
/// field it can show (the ProjectionModel default is pi/2).
///
/// `alpha` chooses s:
///   * 0: the widest view in which every output pixel shows source content
///     (no invalid border): every ray the output shows between its outermost
///     pixel centres is a valid source direction (below) and lands at least
///     `source_margin_px` inside the source image edge. For an
///     equirectangular source that wraps all the way around, the view also
///     stays off the source's seam (the image edge where longitude wraps);
///   * 1: the narrowest view in which every valid source ray that lands
///     between the source's outermost pixel centres, and that the output can
///     represent, lands inside the output image, with an invalid border
///     where the source does not reach;
///   * in between: s interpolates linearly between the two.
/// Both are decided over the continuous sets of rays, not on samples: the
/// fit bounds the projections over patches of directions (interval
/// arithmetic) and refines them until its bound is within 2e-4 of a value an
/// actual ray attains, then keeps 1e-4 more of headroom. Where it cannot
/// settle a patch (a root of the fold conditions it cannot bracket, its work
/// budget), it keeps the bound it has: alpha = 0 narrower, alpha = 1 wider.
///
/// Scope: this holds for the model formulas evaluated in double precision,
/// up to floating-point rounding. The 1e-4 of headroom covers that rounding,
/// and the float evaluation of the public projections, for ordinary models;
/// models near singular parameters (e.g. a Mei model with xi close to -1,
/// where the float and double evaluations of its projection differ greatly)
/// are outside it. Images need at least two pixels along each axis.
///
/// A source ray is valid when it lies before the source distortion's first
/// fold: along its direction out from the source axis, the forward
/// projection must still succeed and keep the orientation it has near the
/// axis. Past a fold (e.g. a strong negative radtan k1 in the image corners)
/// a pixel unprojects onto the folded branch, to a ray the lens does not see
/// there. The fold is found from the distortion polynomials (certified
/// Bernstein bounds), so even a fold that turns back for a fraction of a
/// degree counts. A root where a condition only touches zero (a double
/// root) cannot be told from a crossing one, so the rays past it count as
/// possibly invalid: alpha = 1 keeps them and alpha = 0 stops before them.
///
/// The horizon margin, 0.05 rad (about 3 degrees) from where the output
/// projection's coordinates grow without bound (the pinhole 90-degree
/// horizon, the stereographic antipode, the cylinder's poles, the end of the
/// sphere models' domain) or become singular (the antipode of the
/// equidistant and equisolid laws: a finite circle, but every direction
/// lands on it, so a pixel there has no single ray):
///   * alpha = 1 leaves the source rays inside that margin out (near an
///     unbounded edge they would push s to zero); those pixels may fall
///     outside the output.
///   * alpha = 0 ignores it as long as the source limits the view. Only when
///     the source covers the output all the way to that edge (e.g. a fisheye
///     whose whole image circle lies inside the frame, rectified to pinhole)
///     is there no widest view; the view then stops at the margin.
struct RectifiedOutputModelOptions
{
  /// Projection and pixel shape of the output. Default: a square-pixel
  /// pinhole.
  CameraModel output_template{
    IntrinsicsModel{}, ProjectionModel{ProjectionModelType::PINHOLE}, DistortionModel{}};
  /// Output image size; an invalid (default) size means the source size.
  ImageSize output_size{};
  /// 0 = no invalid output pixels, 1 = the whole source kept. In [0, 1].
  float alpha{0.0f};
  Eigen::Matrix3f src_from_output_rotation{Eigen::Matrix3f::Identity()};
  /// Margin in source pixels that the alpha = 0 view keeps clear of the
  /// source image edge. alpha = 1 keeps the source out to its outermost
  /// pixel centres regardless.
  float source_margin_px{1.0f};
  /// Solver options for the pixelToRay calls of the fit and the map build.
  SolverOptions solver_options{};
};

struct [[nodiscard]] RectifiedOutputModelResult
{
  StatusCode status{StatusCode::OK};
  CameraModel output_model{};
  float horizontal_fov_deg{0.0f};
  float vertical_fov_deg{0.0f};
  float diagonal_fov_deg{0.0f};
  int estimated_source_in_bounds_count{0};
  int estimated_total_count{0};

  constexpr bool ok() const { return status == StatusCode::OK; }
  constexpr explicit operator bool() const { return ok(); }
};

/// Fit the output camera of a rectification (see RectifiedOutputModelOptions).
///
/// On OK, `output_model` is a valid CameraModel of the template's projection
/// and the *_fov_deg fields are computeFov(output_model, output size). The
/// estimated_* counts come from a coarse grid over the output image.
///
/// INVALID_MODEL for an invalid source model; INVALID_INPUT for an invalid
/// source size or a source or output image less than two pixels wide or
/// high, alpha outside [0, 1], or a template that is not a valid
/// model with distortion NONE; DOMAIN_ERROR when the fit has no solution:
/// for alpha < 1, no output view shows the source everywhere (e.g. the
/// output's centre ray misses the source, or the output is centred on the
/// seam of an equirectangular source that wraps around); for alpha > 0, no
/// source ray is representable.
RectifiedOutputModelResult makeRectifiedOutputModel(
  const CameraModel &src_model, ImageSize src_size,
  const RectifiedOutputModelOptions &options = RectifiedOutputModelOptions{}
);

struct [[nodiscard]] RectifyRemapResult
{
  ImageRemapResult remap_result{};
  CameraModel output_model{};
  ImageSize output_size{};
  float horizontal_fov_deg{0.0f};
  float vertical_fov_deg{0.0f};
  float diagonal_fov_deg{0.0f};

  constexpr bool ok() const { return remap_result.ok(); }
  constexpr explicit operator bool() const { return ok(); }
};

/// Fit the output camera (makeRectifiedOutputModel) and build its remap map
/// (buildImageRemapMap with options.src_from_output_rotation, source-bounds
/// checking on). The map is exactly the one buildImageRemapMap builds from
/// the returned output_model, so the model can be stored and the map
/// rebuilt later.
///
/// @param map_x  Output x-coordinates, size = output width * height.
/// @param map_y  Output y-coordinates, same size.
RectifyRemapResult buildRectifyRemapMap(
  const CameraModel &src_model, ImageSize src_size, const RectifiedOutputModelOptions &options,
  float *map_x, float *map_y
);

const char *toString(InvalidPixelPolicy pixel_policy);

}  // namespace camxiom

#endif  // CAMXIOM__REMAP_HPP
