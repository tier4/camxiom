cmake_minimum_required(VERSION 3.14)
project(camxiom VERSION 0.1.0 LANGUAGES CXX)

# include/camxiom/version.hpp and package.xml carry the same version as
# project(); fail the configure if any of them drift apart.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/camxiom/version.hpp" _camxiom_version_hpp)
string(REGEX MATCH "#define CAMXIOM_VERSION_STRING \"([0-9.]+)\"" _camxiom_version_match "${_camxiom_version_hpp}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
  message(FATAL_ERROR
    "include/camxiom/version.hpp declares version '${CMAKE_MATCH_1}' but "
    "project() declares '${PROJECT_VERSION}'; bump both (and package.xml) together.")
endif()
# The numeric macros feed CAMXIOM_VERSION_CODE; keep them in lock-step with
# the string (the snapshot test strips preprocessor lines, so nothing else
# guards them).
string(REGEX MATCH "#define CAMXIOM_VERSION_MAJOR ([0-9]+)" _camxiom_major_match "${_camxiom_version_hpp}")
set(_camxiom_macro_version "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define CAMXIOM_VERSION_MINOR ([0-9]+)" _camxiom_minor_match "${_camxiom_version_hpp}")
string(APPEND _camxiom_macro_version ".${CMAKE_MATCH_1}")
string(REGEX MATCH "#define CAMXIOM_VERSION_PATCH ([0-9]+)" _camxiom_patch_match "${_camxiom_version_hpp}")
string(APPEND _camxiom_macro_version ".${CMAKE_MATCH_1}")
if(NOT _camxiom_macro_version STREQUAL PROJECT_VERSION)
  message(FATAL_ERROR
    "include/camxiom/version.hpp numeric macros declare '${_camxiom_macro_version}' "
    "but project() declares '${PROJECT_VERSION}'; bump MAJOR/MINOR/PATCH with the string.")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/package.xml" _camxiom_package_xml)
string(REGEX MATCH "<version>([0-9.]+)</version>" _camxiom_pkg_match "${_camxiom_package_xml}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
  message(FATAL_ERROR
    "package.xml declares version '${CMAKE_MATCH_1}' but project() declares "
    "'${PROJECT_VERSION}'; bump both together.")
endif()
unset(_camxiom_version_hpp)
unset(_camxiom_version_match)
unset(_camxiom_major_match)
unset(_camxiom_minor_match)
unset(_camxiom_patch_match)
unset(_camxiom_macro_version)
unset(_camxiom_package_xml)
unset(_camxiom_pkg_match)

# ---------------------------------------------------------------------------
# camxiom — OpenCV-free, ROS-free camera geometry library
#
# Split along the runtime-vs-calibration axis so consumers only pay for what
# they use:
#   * camxiom_core   Eigen-only RUNTIME geometry (ray<->pixel projection /
#                    undistort / remap / jacobian / lut / batch / SIMD /
#                    compat). NO Ceres, NO calibration machinery.
#   * camxiom_calib  Everything CALIBRATION: the data-independent default seed,
#                    all initial-guess estimators (linear + Ceres-polished) and
#                    the Ceres-based PnP solver / single-pass intrinsics
#                    calibrator. Links camxiom_core and Ceres. Built only when
#                    CAMXIOM_WITH_CERES is ON.
#   * camxiom        Umbrella INTERFACE target that links every layer that was
#                    built, so existing consumers keep linking
#                    `camxiom::camxiom` for the full feature set.
#
# Ceres is OPTIONAL (#6): a consumer that only needs projection / undistort /
# remap links `camxiom::core` and never pulls in Ceres. CAMXIOM_WITH_CERES
# auto-detects Ceres but can be forced off for a core-only build.
#
# Optional interop layers are compiled INTO camxiom_core when their toolkit is
# available at configure time (Ceres-free, so they stay in the runtime core):
#   * ROS sensor_msgs found  -> src/compat/ros.cpp   (+ CAMXIOM_HAS_ROS)
#   * OpenCV found           -> src/opencv/*.cpp      (+ CAMXIOM_HAS_OPENCV)
# When neither is present, only the dependency-free core is built. ament is
# itself optional: in a ROS/colcon workspace the ament install/export flow is
# used, otherwise a plain CMake install/export is emitted.
# ---------------------------------------------------------------------------

find_package(ament_cmake QUIET)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
# Normalised boolean for the package config template (find_dependency gating).
if(OpenMP_CXX_FOUND)
  set(CAMXIOM_WITH_OPENMP TRUE)
else()
  set(CAMXIOM_WITH_OPENMP FALSE)
endif()

# Ceres is optional. The Ceres-based calibration layer (camxiom_calib) is only
# built when Ceres (>=2.0) is available. Auto-detect, but let callers force off.
# 2.0 is the floor because that is what Ubuntu 22.04 / JetPack apt repos ship;
# the single Manifold-API call site (pnp_solver.cpp) branches on CERES_VERSION.
find_package(Ceres 2.0 QUIET)
option(CAMXIOM_WITH_CERES
  "Build the Ceres-based calibration layer (camxiom_calib)"
  ${Ceres_FOUND})
if(CAMXIOM_WITH_CERES AND NOT Ceres_FOUND)
  message(FATAL_ERROR
    "camxiom: CAMXIOM_WITH_CERES=ON but Ceres (>=2.0) was not found. "
    "Install Ceres, or configure with -DCAMXIOM_WITH_CERES=OFF for a "
    "core-only (projection/remap) build.")
endif()

# Optional interop toolkits (presence decides what compat is built in).
find_package(sensor_msgs QUIET)
find_package(OpenCV QUIET COMPONENTS core imgproc calib3d)

# ---------------------------------------------------------------------------
# Warnings (library targets only, never exported to consumers).
# -Werror=switch is load-bearing, not cosmetic: the per-model dispatch is an
# enum switch replicated across the scalar / double / batch / Jacobian /
# autodiff-template layers, and those switches deliberately carry no
# `default:` — adding a ProjectionModelType / DistortionModelType enumerator
# must turn every not-yet-updated dispatch site into a compile error here,
# instead of a silent runtime INVALID_MODEL or scalar fallback.
# ---------------------------------------------------------------------------
set(CAMXIOM_WARNING_OPTIONS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CAMXIOM_WARNING_OPTIONS -Wall -Wextra -Werror=switch)
endif()

# ---------------------------------------------------------------------------
# camxiom_core — Eigen-only RUNTIME geometry (no Ceres, no calibration).
# ---------------------------------------------------------------------------
add_library(camxiom_core
  src/types.cpp
  src/distortion/plane.cpp
  src/distortion/angle.cpp
  src/model/parse.cpp
  src/model/validate.cpp
  src/model/validated_model.cpp
  src/model/factory.cpp
  src/dispatch.cpp
  src/projection64/dispatch.cpp
  src/projection64/validate.cpp
  src/detail/projection_solve.cpp
  src/pinhole/projection.cpp
  src/pinhole/projection64.cpp
  src/fisheye/projection.cpp
  src/fisheye/projection64.cpp
  src/omnidirectional/projection.cpp
  src/omnidirectional/projection64.cpp
  src/double_sphere/projection.cpp
  src/double_sphere/projection64.cpp
  src/eucm/projection.cpp
  src/eucm/projection64.cpp
  src/jacobian/float.cpp
  src/jacobian/double.cpp
  src/jacobian/batch.cpp
  src/jacobian/with_distortion_deriv_double.cpp
  src/lut/lut.cpp
  src/batch/float.cpp
  src/batch/double.cpp
  # SIMD kernels — SSE2/AVX2 on x86 via __SSE2__/__AVX2__ macros. On aarch64
  # both directions run natively through the NEON compat mapping
  # (src/detail/simd_neon_compat.hpp); only the KB4-family fisheye inverse
  # stays pinned to the scalar path there (measured slower — see
  # ARCHITECTURE.md §5).
  src/detail/simd_pinhole.cpp
  src/detail/simd_fisheye.cpp
  src/detail/simd_ds_eucm.cpp
  src/detail/simd_inverse.cpp
  src/detail/simd_avx2.cpp
  # compat/ — ROS-free, OpenCV-free K/D <-> CameraModel conversions
  # (canonical interchange through the camxiom::CameraInfo POD).
  src/pinhole/compat.cpp
  src/fisheye/compat.cpp
  src/omnidirectional/compat.cpp
  src/double_sphere/compat.cpp
  src/eucm/compat.cpp
  src/compat/factory.cpp
  src/compat/camera_info_export.cpp
  src/compat/camera_info_yaml.cpp
  src/remap.cpp
  src/remap_rectify.cpp
  src/remap_kernel.cpp
)
add_library(camxiom::core ALIAS camxiom_core)

target_compile_features(camxiom_core PUBLIC cxx_std_17)
target_compile_options(camxiom_core PRIVATE ${CAMXIOM_WARNING_OPTIONS})
target_include_directories(camxiom_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_include_directories(camxiom_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(camxiom_core PUBLIC
  Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(camxiom_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(camxiom_core PRIVATE CAMXIOM_HAS_OPENMP)
endif()

# ---------------------------------------------------------------------------
# Optional ROS interop layer (sensor_msgs <-> camxiom::CameraInfo POD).
# Ceres-free, so it is compiled into camxiom_core.
# ---------------------------------------------------------------------------
set(CAMXIOM_WITH_ROS FALSE)
if(sensor_msgs_FOUND)
  set(CAMXIOM_WITH_ROS TRUE)
  target_sources(camxiom_core PRIVATE src/compat/ros.cpp)
  target_compile_definitions(camxiom_core PUBLIC CAMXIOM_HAS_ROS)
  if(COMMAND ament_target_dependencies)
    ament_target_dependencies(camxiom_core PUBLIC sensor_msgs)
  else()
    target_link_libraries(camxiom_core PUBLIC ${sensor_msgs_TARGETS})
  endif()
  message(STATUS "camxiom: ROS sensor_msgs found -> building ROS compat layer")
else()
  message(STATUS "camxiom: ROS sensor_msgs not found -> skipping ROS compat layer")
endif()

# ---------------------------------------------------------------------------
# Optional OpenCV interop layer (cv::Mat drop-in API + CameraInfo bridge).
# Ceres-free, so it is compiled into camxiom_core.
# ---------------------------------------------------------------------------
set(CAMXIOM_WITH_OPENCV FALSE)
if(OpenCV_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/opencv/image.cpp")
  set(CAMXIOM_WITH_OPENCV TRUE)
  target_sources(camxiom_core PRIVATE
    src/opencv/image.cpp
    src/opencv/points.cpp
    src/opencv/cv_api.cpp
    src/opencv/pnp.cpp
  )
  target_compile_definitions(camxiom_core PUBLIC CAMXIOM_HAS_OPENCV)
  target_include_directories(camxiom_core PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(camxiom_core PUBLIC ${OpenCV_LIBS})
  message(STATUS "camxiom: OpenCV found -> building OpenCV compat layer")
else()
  message(STATUS "camxiom: OpenCV compat layer not built")
endif()

# ---------------------------------------------------------------------------
# camxiom_calib — calibration layer: default seed + initial-guess estimators
# (linear + Ceres-polished) + Ceres PnP solver + single-pass intrinsics calib.
# Only built when CAMXIOM_WITH_CERES; links camxiom_core and Ceres PUBLIC.
# ---------------------------------------------------------------------------
set(_camxiom_lib_targets camxiom_core)
if(CAMXIOM_WITH_CERES)
  add_library(camxiom_calib
    # Data-independent default camera-model seed.
    src/default_seed.cpp
    # init/ — initial-guess estimators. homography / pinhole_zhang /
    # pinhole_opencv are linear (Eigen-only) but live here because they are
    # calibration-time tools, not runtime projection; the rest are Ceres-polished.
    src/init/homography.cpp
    src/init/pinhole_opencv.cpp
    src/init/pinhole_zhang.cpp
    src/init/dlt_pnp.cpp
    src/init/seeded_planar_init.cpp
    src/init/double_sphere.cpp
    src/init/eucm.cpp
    src/init/kb4_fisheye.cpp
    src/init/mei_omni.cpp
    # optimizer/ — Ceres-based PnP solver.
    src/optimizer/pnp/pnp_solver.cpp
    src/optimizer/pnp/pnp_cost_analytical_batch.cpp
    src/optimizer/pnp/pnp_gauss_newton.cpp
    # calib/ — single-pass intrinsics calibration (D46, strategy-free).
    src/calib/intrinsics.cpp
    src/calib/convert.cpp
  )
  add_library(camxiom::calib ALIAS camxiom_calib)

  target_compile_features(camxiom_calib PUBLIC cxx_std_17)
  target_compile_options(camxiom_calib PRIVATE ${CAMXIOM_WARNING_OPTIONS})
  target_include_directories(camxiom_calib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(camxiom_calib PUBLIC camxiom_core)

  if(TARGET Ceres::ceres)
    target_link_libraries(camxiom_calib PUBLIC Ceres::ceres)
  else()
    target_include_directories(camxiom_calib PUBLIC ${CERES_INCLUDE_DIRS})
    target_link_libraries(camxiom_calib PUBLIC ${CERES_LIBRARIES})
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(camxiom_calib PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(camxiom_calib PRIVATE CAMXIOM_HAS_OPENMP)
  endif()

  list(APPEND _camxiom_lib_targets camxiom_calib)
  message(STATUS "camxiom: Ceres found -> building camxiom_calib (seed/init/PnP/calib)")
else()
  message(STATUS "camxiom: Ceres-based calibration disabled -> core-only build")
endif()

# ---------------------------------------------------------------------------
# camxiom — umbrella INTERFACE target (full feature set). Existing consumers
# link `camxiom::camxiom` and transparently get core (+ calib when built).
# ---------------------------------------------------------------------------
add_library(camxiom INTERFACE)
add_library(camxiom::camxiom ALIAS camxiom)
target_link_libraries(camxiom INTERFACE camxiom_core)
if(CAMXIOM_WITH_CERES)
  target_link_libraries(camxiom INTERFACE camxiom_calib)
endif()

install(DIRECTORY include/
  DESTINATION include
)
install(TARGETS ${_camxiom_lib_targets} camxiom
  EXPORT camxiomTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# ---------------------------------------------------------------------------
# Packaging — dual mode.
#
# camxiom is a pure-CMake library and does not require ROS 2. Inside a ROS 2 /
# colcon (ament) workspace, ament_cmake is on the prefix path and we register
# with the ament index. Outside ROS 2, we install a standalone CMake package
# config carrying the proper find_dependency() calls. Both paths export the SAME
# camxiom:: targets, so downstream `find_package(camxiom)` works either way; no
# separate `camxiom_ros` package is produced.
# ---------------------------------------------------------------------------
if(ament_cmake_FOUND)
  # ament_export_targets() installs the EXPORT set into the ament location.
  ament_export_targets(camxiomTargets HAS_LIBRARY_TARGET)
  set(_camxiom_export_deps Eigen3)
  if(CAMXIOM_WITH_OPENMP)
    list(APPEND _camxiom_export_deps OpenMP)
  endif()
  if(CAMXIOM_WITH_CERES)
    list(APPEND _camxiom_export_deps Ceres)
  endif()
  if(CAMXIOM_WITH_ROS)
    list(APPEND _camxiom_export_deps sensor_msgs)
  endif()
  if(CAMXIOM_WITH_OPENCV)
    list(APPEND _camxiom_export_deps OpenCV)
  endif()
  ament_export_dependencies(${_camxiom_export_deps})
else()
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  set(CAMXIOM_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/camxiom)

  install(EXPORT camxiomTargets
    FILE camxiomTargets.cmake
    NAMESPACE camxiom::
    DESTINATION ${CAMXIOM_INSTALL_CMAKEDIR}
  )

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/camxiomConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/camxiomConfig.cmake
    INSTALL_DESTINATION ${CAMXIOM_INSTALL_CMAKEDIR}
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/camxiomConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/camxiomConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/camxiomConfigVersion.cmake
    DESTINATION ${CAMXIOM_INSTALL_CMAKEDIR}
  )
endif()

# ---------------------------------------------------------------------------
# Benchmarks — optional, OFF by default. A dependency-light executable that
# links camxiom_core only (Eigen) and measures the float/double projection hot
# path. Used to capture/compare a performance baseline (e.g. across the #1
# float/double template unification). Build with an optimised config:
#   -DCMAKE_BUILD_TYPE=Release -DCAMXIOM_BUILD_BENCHMARKS=ON
# ---------------------------------------------------------------------------
option(CAMXIOM_BUILD_BENCHMARKS "Build camxiom microbenchmarks" OFF)
if(CAMXIOM_BUILD_BENCHMARKS)
  add_executable(camxiom_projection_benchmark benchmarks/projection_benchmark.cpp)
  target_link_libraries(camxiom_projection_benchmark PRIVATE camxiom_core)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(WARNING
      "camxiom: CAMXIOM_BUILD_BENCHMARKS is ON but CMAKE_BUILD_TYPE is unset; "
      "benchmark numbers are only meaningful with an optimised build "
      "(e.g. -DCMAKE_BUILD_TYPE=Release).")
  endif()
  # calibrate() diagnostic-overhead benchmark: needs the calibration layer
  # (Ceres), so it is only built when the calib target exists.
  if(CAMXIOM_WITH_CERES)
    add_executable(camxiom_calibrate_benchmark benchmarks/calibrate_benchmark.cpp)
    target_link_libraries(camxiom_calibrate_benchmark PRIVATE camxiom_calib camxiom_core)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests — dual mode. Under ament we use ament_add_gtest; in a standalone CMake
# build we fall back to a plain GTest::gtest_main executable (only when GTest is
# discoverable). Set BUILD_TESTING=ON to enable (colcon sets it by default).
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  set(_camxiom_have_gtest FALSE)
  if(ament_cmake_FOUND)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
    find_package(ament_cmake_gtest REQUIRED)
    set(_camxiom_have_gtest TRUE)
  else()
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
      set(_camxiom_have_gtest TRUE)
    else()
      message(STATUS "camxiom: GTest not found -> skipping standalone tests")
    endif()
  endif()

  if(_camxiom_have_gtest)
    # Declare a gtest in whichever mode we are in, then link camxiom.
    function(camxiom_add_gtest test_name test_src)
      if(ament_cmake_FOUND)
        ament_add_gtest(${test_name} ${test_src})
      else()
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} GTest::gtest GTest::gtest_main)
        add_test(NAME ${test_name} COMMAND ${test_name})
      endif()
      if(TARGET ${test_name})
        target_link_libraries(${test_name} camxiom)
      endif()
    endfunction()

    # --- Core tests (runtime geometry, no Ceres) --------------------------
    camxiom_add_gtest(camxiom_types_test tests/types_test.cpp)
    # Validation-tier contract: coefficient-free types must carry all-zero
    # coeffs (scalar/SIMD divergence guard) and polynomial-fisheye caps must
    # sit inside the monotone range (forward/inverse round-trip guard).
    camxiom_add_gtest(camxiom_model_validate_test tests/model_validate_test.cpp)
    camxiom_add_gtest(camxiom_model_compare_test tests/model_compare_test.cpp)
    camxiom_add_gtest(camxiom_projection_smoke_test tests/projection_smoke_test.cpp)
    camxiom_add_gtest(camxiom_batch_parity_test tests/batch_parity_test.cpp)
    # Error-path production (DOMAIN_ERROR / NUMERIC_ERROR): exact-zero
    # denominators and overflow triggers through the public projection API,
    # float and double, scalar and batch.
    camxiom_add_gtest(camxiom_projection_error_paths_test tests/projection_error_paths_test.cpp)
    camxiom_add_gtest(camxiom_camera_info_yaml_test tests/camera_info_yaml_test.cpp)
    camxiom_add_gtest(camxiom_remap_test tests/remap_test.cpp)
    camxiom_add_gtest(camxiom_remap_kernel_test tests/remap_kernel_test.cpp)
    camxiom_add_gtest(camxiom_jacobian_batch_lut_test tests/jacobian_batch_lut_test.cpp)
    # Convenience/umbrella headers (#2). A CORE test so it is built even when
    # CAMXIOM_WITH_CERES is OFF, guarding that core.hpp / calibration.hpp /
    # camxiom.hpp stay include-safe (Ceres-free) in a core-only build.
    camxiom_add_gtest(camxiom_convenience_headers_test tests/convenience_headers_test.cpp)
    # compat POD interchange (src/compat/*, src/<model>/compat.cpp): CameraInfo
    # <-> CameraModel and the per-model import/export profile conversions. A CORE
    # test (Ceres/ROS/OpenCV-free) that guards the library's canonical entry point.
    camxiom_add_gtest(camxiom_compat_test tests/compat_test.cpp)
    # ValidatedCameraModel (C1): validated-once immutable wrapper. A CORE test
    # (Ceres/ROS/OpenCV-free) asserting bit-for-bit equivalence with the generic
    # rayToPixel/pixelToRay, tryMake rejection of invalid models, and input-guard
    # parity.
    camxiom_add_gtest(camxiom_validated_model_test tests/validated_model_test.cpp)
    # Property / fuzz invariants (Q1): a CORE test (Ceres/ROS/OpenCV-free) that
    # generates many random valid models + rays from a FIXED seed and asserts
    # projection invariants (project∘unproject ≈ id, FOV-interior status OK,
    # positive-scale invariance, determinism, pixel round-trip).
    camxiom_add_gtest(camxiom_property_test tests/property_test.cpp)

    # OpenCV interop layer (src/opencv/*). A CORE test (Ceres-free) that is only
    # built when the OpenCV compat layer was compiled in (CAMXIOM_WITH_OPENCV),
    # since otherwise the camxiom::opencv symbols do not exist. Guards the
    # cv::Mat drop-in API + verifies it thinly wraps the core geometry.
    if(CAMXIOM_WITH_OPENCV)
      camxiom_add_gtest(camxiom_opencv_interop_test tests/opencv_interop_test.cpp)
    endif()

    # ROS interop layer (src/compat/ros.cpp). A CORE test (Ceres-free) built only
    # when the ROS compat layer was compiled in (CAMXIOM_WITH_ROS), so the
    # sensor_msgs symbols exist. Guards the sensor_msgs <-> CameraModel field
    # wiring on top of the ROS-free POD path already covered by compat_test.
    if(CAMXIOM_WITH_ROS)
      camxiom_add_gtest(camxiom_ros_compat_test tests/ros_compat_test.cpp)
    endif()

    # --- Calibration tests (seed/init/PnP/calib; need camxiom_calib) ------
    if(CAMXIOM_WITH_CERES)
      # AutoDiff (ceres::Jet) <-> analytic projection-Jacobian agreement (T4):
      # guards that the projection_template forward path (used by the PnP cost
      # functors for autodiff) stays consistent with the analytic rayToPixel-
      # WithJacobian. Needs ceres::Jet, hence Ceres-gated.
      camxiom_add_gtest(camxiom_autodiff_jacobian_test tests/autodiff_jacobian_test.cpp)
      camxiom_add_gtest(camxiom_default_seed_test tests/default_seed_test.cpp)
      camxiom_add_gtest(camxiom_homography_test tests/homography_test.cpp)
      camxiom_add_gtest(camxiom_pinhole_opencv_test tests/pinhole_opencv_test.cpp)
      camxiom_add_gtest(camxiom_pinhole_zhang_test tests/pinhole_zhang_test.cpp)
      camxiom_add_gtest(camxiom_calib_intrinsics_test tests/calib_intrinsics_test.cpp)
      camxiom_add_gtest(camxiom_model_convert_test tests/model_convert_test.cpp)
      camxiom_add_gtest(camxiom_calib_uncertainty_test tests/calib_uncertainty_test.cpp)
      # C5: this test also unit-tests the internal pure helpers
      # (calib/uncertainty_detail.hpp, optimizer/pnp/pnp_parameter_bounds.hpp),
      # which live under src/ and are NOT part of the installed public API, so it
      # needs the PRIVATE src include path (mirrors camxiom_calib's own).
      if(TARGET camxiom_calib_uncertainty_test)
        target_include_directories(camxiom_calib_uncertainty_test PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/src)
      endif()
      camxiom_add_gtest(camxiom_dlt_pnp_test tests/dlt_pnp_test.cpp)
      camxiom_add_gtest(camxiom_pnp_huber_test tests/pnp_huber_test.cpp)
      camxiom_add_gtest(camxiom_double_sphere_test tests/double_sphere_test.cpp)
      camxiom_add_gtest(camxiom_eucm_test tests/eucm_test.cpp)
      camxiom_add_gtest(camxiom_integration_test tests/integration_test.cpp)
      camxiom_add_gtest(camxiom_kb4_fisheye_test tests/kb4_fisheye_test.cpp)
      camxiom_add_gtest(camxiom_mei_omni_test tests/mei_omni_test.cpp)
    endif()
  endif()

  # --- Public API symbol-set snapshot (Rel1a) ---------------------------
  # A gtest-INDEPENDENT CTest: a stdlib-only Python script extracts the public
  # symbol surface from include/camxiom/** (internal/ excluded) and diffs it
  # against the checked-in golden tests/api_snapshot/public_api.txt, failing on
  # any drift (the guard against unintended public API changes, e.g. the #3
  # per-model namespace removal, before the 0.1.0 release). It parses header
  # SOURCE TEXT, so it is config-independent: identical output on the aarch64 dev
  # box and the x86 CI, regardless of whether Ceres/OpenCV/ROS were found. It
  # adds no gtest (the gtest-case count is unchanged). Regenerate the golden after
  # a DELIBERATE API change: CAMXIOM_UPDATE_API_SNAPSHOT=1 ctest -R
  # camxiom_api_snapshot_test (or run the script with --update).
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    set(_camxiom_api_snapshot_cmd
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/api_snapshot/check_public_api.py
      --include-dir ${CMAKE_CURRENT_SOURCE_DIR}/include
      --snapshot ${CMAKE_CURRENT_SOURCE_DIR}/tests/api_snapshot/public_api.txt)
    if(ament_cmake_FOUND)
      find_package(ament_cmake_test REQUIRED)
      ament_add_test(camxiom_api_snapshot_test
        COMMAND ${_camxiom_api_snapshot_cmd}
        GENERATE_RESULT_FOR_RETURN_CODE_ZERO
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    else()
      add_test(NAME camxiom_api_snapshot_test COMMAND ${_camxiom_api_snapshot_cmd})
    endif()
  else()
    message(STATUS "camxiom: Python3 not found -> skipping public API snapshot test")
  endif()
endif()

if(ament_cmake_FOUND)
  ament_package()
endif()
