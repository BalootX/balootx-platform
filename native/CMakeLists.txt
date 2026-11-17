cmake_minimum_required(VERSION 3.20)
project(baloot_native VERSION 21.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
add_library(baloot_native SHARED src/native.cpp src/double_dummy.cpp)
find_package(Threads REQUIRED)
target_link_libraries(baloot_native PRIVATE Threads::Threads)
target_compile_features(baloot_native PUBLIC cxx_std_20)
target_include_directories(baloot_native PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(MSVC)
  target_compile_options(baloot_native PRIVATE /W4 /WX)
else()
  target_compile_options(baloot_native PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
set_target_properties(baloot_native PROPERTIES OUTPUT_NAME baloot_native CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
add_executable(baloot_native_tests tests/native_tests.cpp)
target_link_libraries(baloot_native_tests PRIVATE baloot_native)
# Test assertions include the API calls under test and must run in Release too.
if(MSVC)
  target_compile_options(baloot_native_tests PRIVATE /UNDEBUG)
else()
  target_compile_options(baloot_native_tests PRIVATE -UNDEBUG)
endif()

# The compiler links the ASan runtime into sanitized targets.  Do not inject a
# second runtime through LD_PRELOAD: on Debian-family systems that can create a
# recursive AddressSanitizer:DEADLYSIGNAL loop even for a healthy binary.
set(
  BALOOT_ASAN_OPTIONS
  ""
  CACHE STRING
  "Optional AddressSanitizer options applied only to a sanitized native test child"
)

enable_testing()
if(BALOOT_ASAN_OPTIONS)
  add_test(
    NAME native
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "ASAN_OPTIONS=${BALOOT_ASAN_OPTIONS}"
      "$<TARGET_FILE:baloot_native_tests>"
  )
else()
  add_test(NAME native COMMAND "$<TARGET_FILE:baloot_native_tests>")
endif()
set_tests_properties(native PROPERTIES TIMEOUT 120 LABELS "native")
install(TARGETS baloot_native EXPORT baloot_nativeTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT baloot_nativeTargets NAMESPACE baloot:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/baloot_native)
