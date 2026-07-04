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

#include "detail/simd_pinhole.hpp"

#include "camxiom/projection.hpp"
#include "detail/projection_models.hpp"

namespace camxiom::detail
{

#ifdef CAMXIOM_HAS_SSE2

int rayToPixelBatchPinholeSse(
  const CameraModel &model, const float *rays_xyz, const int count, float *u_out, float *v_out,
  StatusCode *statuses_out
)
{
  if (model.distortion.has_tilt)
  {
    int valid_count = 0;
    for (int i = 0; i < count; ++i)
    {
      const int off = i * 3;
      const Eigen::Vector3f ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
      const PixelResult result = pinhole::rayToPixel(model, ray);
      u_out[i] = result.pixel.u;
      v_out[i] = result.pixel.v;
      if (statuses_out != nullptr)
      {
        statuses_out[i] = result.status;
      }
      if (result.status == StatusCode::OK)
      {
        ++valid_count;
      }
    }
    return valid_count;
  }

  const int simd_count = count & ~3;
  int valid_count = 0;

  for (int i = 0; i < simd_count; i += 4)
  {
    float u4[4];
    float v4[4];
    const int mask = rayToPixelPinholeSse4(model, rays_xyz + i * 3, u4, v4);

    for (int j = 0; j < 4; ++j)
    {
      const int index = i + j;
      const bool simd_ok = ((mask >> j) & 1) != 0;
      if (simd_ok)
      {
        u_out[index] = u4[j];
        v_out[index] = v4[j];
        if (statuses_out != nullptr)
        {
          statuses_out[index] = StatusCode::OK;
        }
        ++valid_count;
      }
      else
      {
        const int off = index * 3;
        const Eigen::Vector3f ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
        const PixelResult result = pinhole::rayToPixel(model, ray);
        u_out[index] = result.pixel.u;
        v_out[index] = result.pixel.v;
        if (statuses_out != nullptr)
        {
          statuses_out[index] = result.status;
        }
        if (result.status == StatusCode::OK)
        {
          ++valid_count;
        }
      }
    }
  }

  for (int i = simd_count; i < count; ++i)
  {
    const int off = i * 3;
    const Eigen::Vector3f ray(rays_xyz[off], rays_xyz[off + 1], rays_xyz[off + 2]);
    const PixelResult result = pinhole::rayToPixel(model, ray);
    u_out[i] = result.pixel.u;
    v_out[i] = result.pixel.v;
    if (statuses_out != nullptr)
    {
      statuses_out[i] = result.status;
    }
    if (result.status == StatusCode::OK)
    {
      ++valid_count;
    }
  }

  return valid_count;
}

#endif  // CAMXIOM_HAS_SSE2

}  // namespace camxiom::detail
