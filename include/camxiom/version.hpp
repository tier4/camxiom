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

#ifndef CAMXIOM__VERSION_HPP
#define CAMXIOM__VERSION_HPP

// camxiom release version.
//
// Kept in sync with the CMake project() VERSION and package.xml by a
// configure-time check in CMakeLists.txt; bump all three together.
//
// CAMXIOM_VERSION_CODE supports numeric comparison in consumer code:
//   #if CAMXIOM_VERSION_CODE >= CAMXIOM_MAKE_VERSION(0, 2, 0)

#define CAMXIOM_VERSION_MAJOR 0
#define CAMXIOM_VERSION_MINOR 1
#define CAMXIOM_VERSION_PATCH 0
#define CAMXIOM_VERSION_STRING "0.1.0"

#define CAMXIOM_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + (patch))
#define CAMXIOM_VERSION_CODE \
  CAMXIOM_MAKE_VERSION(CAMXIOM_VERSION_MAJOR, CAMXIOM_VERSION_MINOR, CAMXIOM_VERSION_PATCH)

#endif  // CAMXIOM__VERSION_HPP
