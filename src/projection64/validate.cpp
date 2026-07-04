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

#include "camxiom/projection64.hpp"
#include "model/validate_impl.hpp"

// Double validator: a thin <double> instantiation of the scalar-templated core
// in model/validate_impl.hpp (#1 step 4). The float counterpart is the <float>
// instantiation in src/model/validate.cpp; both share one source, so the D47
// pi-boundary reconciliation can no longer drift between precisions.

namespace camxiom
{

namespace detail64
{

StatusCode validateCameraModelQuery64(const CameraModel64 &model)
{
  return detail_impl::validateCameraModelQueryImpl<double>(model);
}

}  // namespace detail64

StatusCode validateCameraModel64(const CameraModel64 &model)
{
  return detail_impl::validateCameraModelImpl<double>(model);
}

}  // namespace camxiom
