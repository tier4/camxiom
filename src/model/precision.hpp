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

#ifndef CAMXIOM__MODEL__PRECISION_HPP
#define CAMXIOM__MODEL__PRECISION_HPP

#include "camxiom/model.hpp"
#include "camxiom/types64.hpp"

#include <type_traits>

namespace camxiom::detail
{
template <typename T>
CameraModel64 asModel64(const CameraModelT<T> &model)
{
  if constexpr (std::is_same_v<T, double>)
    return model;
  else
    return toCameraModel64(model);
}

template <typename T>
StatusCode validateModel(const CameraModelT<T> &model)
{
  if constexpr (std::is_same_v<T, double>)
    return validateCameraModel64(model);
  else
    return validateCameraModel(model);
}

template <typename T>
void updateModelThetaMax(CameraModelT<T> &model)
{
  if constexpr (std::is_same_v<T, double>)
    updateThetaMax64(model);
  else
    updateThetaMax(model);
}
}  // namespace camxiom::detail

#endif  // CAMXIOM__MODEL__PRECISION_HPP
