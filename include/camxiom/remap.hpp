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
/// produce a rectilinear (perspective) image. Use buildRectifyRemapMap() for that.
RemapResult buildUndistortRemapMap(
  const CameraModel &src_model, int width, int height, float *map_x, float *map_y,
  const SolverOptions &solver_options = SolverOptions{}
);

// ===========================================================================
// Image-aware remap
// ===========================================================================

struct ImageSize
{
  int width{0};
  int height{0};

  constexpr bool isValid() const { return width > 0 && height > 0; }
};

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
/// Works for ANY camera model (pinhole, fisheye, omnidirectional, double
/// sphere, EUCM): each angle is measured between the unprojected rays of two
/// boundary pixels (pixelToRay), the same computation the rectify APIs use to
/// fill their *_fov_deg fields.
///   * horizontal: (0, cy) .. (width-1, cy) along the principal-point row
///   * vertical:   (cx, 0) .. (cx, height-1) along the principal-point column
///   * diagonal:   (0, 0) .. (width-1, height-1)
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
// Rectification: virtual output camera generation + remap
// ===========================================================================

enum class RectifiedProjectionType : std::uint8_t {
  PINHOLE = 0U,
  CYLINDRICAL,
  STEREOGRAPHIC,
  LONGITUDE_LATITUDE
};

enum class RectifyFitPolicy : std::uint8_t {
  MAX_INSCRIBED_VALID = 0U,
  ALLOW_INVALID_BORDER,
  // ALL_SOURCE_CONTAINED: alpha=1-style widest-FOV fit.
  // Finds the smallest output focal such that every source pixel's ray projects
  // into the output bounds; rays at angles >= (pi/2 - clamp_margin) are skipped
  // because a rectilinear pinhole target cannot represent them stably.
  ALL_SOURCE_CONTAINED
};

struct RectifiedOutputModelOptions
{
  RectifiedProjectionType projection_type{RectifiedProjectionType::PINHOLE};
  ImageSize output_size{};
  float focal_scale{1.0f};
  RectifyFitPolicy fit_policy{RectifyFitPolicy::MAX_INSCRIBED_VALID};
  Eigen::Matrix3f src_from_output_rotation{Eigen::Matrix3f::Identity()};
  int boundary_sample_count{2048};
  float source_margin_px{1.0f};
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

/// Generate a virtual rectified output camera model from a source camera model.
///
/// Currently only supports RectifiedProjectionType::PINHOLE.
/// For non-pinhole rectification types (CYLINDRICAL, STEREOGRAPHIC, LONGITUDE_LATITUDE),
/// use buildRectifyRemapMap() directly — the output_model field of RectifyRemapResult
/// will be default-constructed (UNKNOWN) because these projections are not representable
/// as a CameraModel.
///
/// For PINHOLE, the generated model has:
///   - projection.type = PINHOLE
///   - distortion.type = NONE
///   - skew = 0, fx/fy fitted independently to the selected policy
///   - cx = (width - 1) / 2, cy = (height - 1) / 2
///
/// @param src_model  Source camera model.
/// @param src_size   Source image dimensions.
/// @param options    Rectification options (output size, focal_scale, fit policy, etc.).
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

/// Convenience: generate rectified output model + build remap map in one call.
///
/// Internally calls makeRectifiedOutputModel() then buildImageRemapMap().
///
/// @param src_model  Source camera model.
/// @param src_size   Source image dimensions.
/// @param options    Rectification options.
/// @param map_x      Output buffer for x-coordinates.
/// @param map_y      Output buffer for y-coordinates.
RectifyRemapResult buildRectifyRemapMap(
  const CameraModel &src_model, ImageSize src_size, const RectifiedOutputModelOptions &options,
  float *map_x, float *map_y
);

/// Options for buildRectifyMap (alpha API).
struct BuildRectifyMapOptions
{
  /// Rotation from output frame to source frame.  Default identity.
  /// Use a non-identity rotation for stereo rectify pre-processing.
  Eigen::Matrix3f src_from_output_rotation{Eigen::Matrix3f::Identity()};
  /// Number of boundary samples for inscribed binary search and
  /// circumscribed direct computation.  Default 2048 matches the
  /// existing RectifiedOutputModelOptions.
  int boundary_sample_count{2048};
  /// Pixel margin applied to source boundary (matches existing
  /// RectifiedOutputModelOptions::source_margin_px).
  float source_margin_px{1.0f};
  /// Solver options forwarded to pixelToRay calls during remap-map build.
  SolverOptions solver_options{};
};

/// Result of buildRectifyMap.
struct BuildRectifyMapResult
{
  /// Status + per-pixel coverage counts from the underlying buildImageRemapMap.
  ImageRemapResult remap_result{};
  /// Generated pinhole output model (K_new).  Valid only when
  /// remap_result.status == OK.
  CameraModel output_model{};
  /// Echo of the input alpha after clamping to [0, 1].
  float alpha_used{0.0f};
  /// Horizontal field of view of output_model, degrees.
  float horizontal_fov_deg{0.0f};
  /// Vertical field of view of output_model, degrees.
  float vertical_fov_deg{0.0f};
  /// Diagonal field of view of output_model, degrees.
  float diagonal_fov_deg{0.0f};

  constexpr bool ok() const { return remap_result.ok(); }
  constexpr explicit operator bool() const { return ok(); }
};

/// Alpha-blended rectification map builder.
///
/// alpha = 0: rectified image contains only valid pixels (inscribed).
/// alpha = 1: rectified image keeps as much source FOV as the virtual pinhole
///            target can represent, with black borders where needed;
///            near-90-degree rays are skipped because a rectilinear pinhole
///            target cannot represent them stably.
/// Intermediate alpha: focal is linearly interpolated between the two extremes.
///
/// @param src_model  Source camera model (any of pinhole / fisheye / omni /
///                   double sphere / EUCM).
/// @param src_size   Source image dimensions.
/// @param dst_size   Output image dimensions.
/// @param alpha      Mixing parameter; clamped to [0, 1] if outside.
/// @param map_x      Output x-coordinates buffer, size = dst_size.width * dst_size.height.
/// @param map_y      Output y-coordinates buffer, same size.
/// @param options    Optional tuning (rotation, sample count, margin, solver).
BuildRectifyMapResult buildRectifyMap(
  const CameraModel &src_model, ImageSize src_size, ImageSize dst_size, float alpha, float *map_x,
  float *map_y, const BuildRectifyMapOptions &options = BuildRectifyMapOptions{}
);

const char *toString(RectifiedProjectionType projection_type);
const char *toString(RectifyFitPolicy fit_policy);
const char *toString(InvalidPixelPolicy pixel_policy);

}  // namespace camxiom

#endif  // CAMXIOM__REMAP_HPP
