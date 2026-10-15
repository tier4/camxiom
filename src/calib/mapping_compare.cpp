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

#include "camxiom/model.hpp"
#include "camxiom/projection64.hpp"
#include "camxiom/types64.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace camxiom::calib
{

namespace
{

bool specValid(const MappingDomainSpec &spec)
{
  return spec.image_width > 0 && spec.image_height > 0 && spec.lattice_cols >= 2 &&
         spec.lattice_rows >= 2;
}

double latticeCoord(int index, int count, int extent)
{
  // Uniform lattice including both borders: index in [0, count-1] maps onto
  // pixel coordinates [0, extent-1].
  return static_cast<double>(index) * static_cast<double>(extent - 1) /
         static_cast<double>(count - 1);
}

}  // namespace

MappingComparison compareProjectionMappings(
  const CameraModel &model_a, const CameraModel &model_b, const MappingDomainSpec &spec
)
{
  MappingComparison out;
  if (!specValid(spec) || validateCameraModel(model_a) != StatusCode::OK ||
      validateCameraModel(model_b) != StatusCode::OK)
  {
    out.status = StatusCode::INVALID_INPUT;
    return out;
  }

  const CameraModel64 a64 = toCameraModel64(model_a);
  const CameraModel64 b64 = toCameraModel64(model_b);

  std::vector<double> deltas;
  deltas.reserve(static_cast<std::size_t>(spec.lattice_cols) * spec.lattice_rows);
  double sum_sq = 0.0;
  double max_delta = 0.0;

  for (int row = 0; row < spec.lattice_rows; ++row)
  {
    for (int col = 0; col < spec.lattice_cols; ++col)
    {
      ++out.lattice_points;
      const Pixel2d pixel{
        latticeCoord(col, spec.lattice_cols, spec.image_width),
        latticeCoord(row, spec.lattice_rows, spec.image_height)};
      const RayResult64 ray_a = pixelToRay64(a64, pixel);
      const RayResult64 ray_b = pixelToRay64(b64, pixel);
      const bool a_valid = ray_a.status == StatusCode::OK && ray_a.ray.direction.norm() > 0.0;
      const bool b_valid = ray_b.status == StatusCode::OK && ray_b.ray.direction.norm() > 0.0;
      if (a_valid && b_valid)
      {
        ++out.both_valid;
        const double cosine = std::clamp(
          ray_a.ray.direction.normalized().dot(ray_b.ray.direction.normalized()), -1.0, 1.0
        );
        const double delta = std::acos(cosine);
        deltas.push_back(delta);
        sum_sq += delta * delta;
        max_delta = std::max(max_delta, delta);
      }
      else if (a_valid)
      {
        ++out.only_a_valid;
      }
      else if (b_valid)
      {
        ++out.only_b_valid;
      }
      else
      {
        ++out.neither_valid;
      }
    }
  }

  if (!deltas.empty())
  {
    out.rms_angular_delta_rad = std::sqrt(sum_sq / static_cast<double>(deltas.size()));
    out.max_angular_delta_rad = max_delta;
    std::sort(deltas.begin(), deltas.end());
    const std::size_t p95_index = std::min(
      deltas.size() - 1, static_cast<std::size_t>(0.95 * static_cast<double>(deltas.size()))
    );
    out.p95_angular_delta_rad = deltas[p95_index];
  }
  out.status = StatusCode::OK;
  return out;
}

MappingDomainReport validateMappingDomain(const CameraModel &model, const MappingDomainSpec &spec)
{
  MappingDomainReport out;
  if (!specValid(spec) || validateCameraModel(model) != StatusCode::OK)
  {
    out.status = StatusCode::INVALID_INPUT;
    return out;
  }
  const CameraModel64 model64 = toCameraModel64(model);
  for (int row = 0; row < spec.lattice_rows; ++row)
  {
    for (int col = 0; col < spec.lattice_cols; ++col)
    {
      ++out.lattice_points;
      const Pixel2d pixel{
        latticeCoord(col, spec.lattice_cols, spec.image_width),
        latticeCoord(row, spec.lattice_rows, spec.image_height)};
      const RayResult64 ray = pixelToRay64(model64, pixel);
      if (ray.status == StatusCode::OK && ray.ray.direction.norm() > 0.0)
      {
        ++out.valid_points;
      }
    }
  }
  out.status = StatusCode::OK;
  return out;
}

}  // namespace camxiom::calib
