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

#ifndef CAMXIOM__CALIB__CONVERT_HPP
#define CAMXIOM__CALIB__CONVERT_HPP

#include "camxiom/types.hpp"

#include <limits>

namespace camxiom::calib
{

/// Options for convertCameraModel().
struct ModelConversionOptions
{
  /// Synthetic-correspondence grid over the source image. Source pixels that
  /// do not unproject (outside the source FOV) are skipped automatically.
  int grid_cols{24};
  int grid_rows{18};
  /// Iteration cap for the non-linear fit.
  int max_iterations{200};
};

/// Result of convertCameraModel().
///
/// `status == OK` means the fit CONVERGED — it does not certify quality.
/// Whether the destination family can actually represent the source
/// geometry is reported honestly through rms/max_fit_error_px and
/// representable_point_count: converting a 190-degree fisheye to a pinhole
/// converges to the best pinhole there is, with a large residual.
struct [[nodiscard]] ModelConversionResult
{
  StatusCode status{StatusCode::INVALID_INPUT};
  /// The fitted model (same family as the seed passed in).
  CameraModel camera_model{};
  /// Reprojection residual of the fitted model against the source pixels,
  /// over the grid correspondences the fitted model can represent.
  double rms_fit_error_px{std::numeric_limits<double>::infinity()};
  double max_fit_error_px{std::numeric_limits<double>::infinity()};
  /// Grid correspondences used for the fit (source unprojection succeeded).
  int used_point_count{0};
  /// Of those, how many the FITTED model can reproject at all. A large gap
  /// to used_point_count means part of the source FOV is outside what the
  /// destination family can express.
  int representable_point_count{0};

  constexpr bool ok() const { return status == StatusCode::OK; }
  explicit operator bool() const { return ok(); }
};

/// Fit a camera model of a (typically different) family so it reproduces
/// `src_model`'s pixel<->ray geometry: e.g. KB4 fisheye -> pinhole+radtan5,
/// pinhole -> double sphere, MEI -> EUCM.
///
/// How: a grid of source pixels is unprojected through `src_model` into unit
/// rays, and the destination model is fitted to project those rays back onto
/// the same pixels (single synthetic view, pose fixed to identity, ordinary
/// least squares — the correspondences are synthetic, so there are no
/// outliers to robustify against).
///
/// `dst_seed` selects the destination family AND provides the initial guess:
/// pass getDefaultSeed(type, width, height) for the standard warm start, or
/// a hand-tuned model when converting between wildly different geometries.
/// The fitted model's theta_max is refreshed from its final coefficients.
///
/// @return status OK on a CONVERGED fit: the optimizer met its tolerances,
///         or an iteration-capped fit already measures at numerical
///         precision (rms <= 1e-3 px — exact model pairs drive the cost to
///         machine epsilon, where relative tolerances cannot fire). Check
///         rms_fit_error_px for quality. NON_CONVERGED with the best-effort
///         model otherwise, including ordinary iteration-capped fits;
///         INVALID_INPUT / INVALID_MODEL / DEGENERATE_CONFIG on bad inputs
///         (fewer than 8 usable grid correspondences).
[[nodiscard]] ModelConversionResult convertCameraModel(
  const CameraModel &src_model, int image_width, int image_height, const CameraModel &dst_seed,
  const ModelConversionOptions &options = ModelConversionOptions{}
);

}  // namespace camxiom::calib

#endif  // CAMXIOM__CALIB__CONVERT_HPP
