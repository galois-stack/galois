cmake_minimum_required(VERSION 3.14)

project(galois CXX C)

set(CMAKE_WARN_DEPRECATED OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 用于智能提示
set(CMAKE_CXX_STANDARD 23) # c++20目前系统库还不支持, 没有必要
set(CXX_STANDARD_REQUIRED ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()
if (${CMAKE_BUILD_TYPE} STREQUAL "release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 在mac平台下， 因为llvm设置为hidden，所以避免编译warning这里全局设置为hidden
# 之所以不用target_link_options设置，是因为其不能设置第三方项目
if (NOT BUILD_SHARED_LIBS)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (MSVC)
    add_compile_options(/wd4996)
else()
    add_compile_options(-Wno-deprecated-this-capture)
    add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)
endif()



set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(third_party/prajna)
# eigen的uninstall和z3的冲突， 所以这里自己定义
add_library (eigen INTERFACE)
target_include_directories (eigen INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>
)

add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
add_subdirectory(third_party/z3 EXCLUDE_FROM_ALL)

set(CPUINFO_BUILD_UNIT_TESTS  OFF)
set(CPUINFO_BUILD_MOCK_TESTS  OFF)
set(CPUINFO_BUILD_BENCHMARKS  OFF) # disable benchmark, the googletest benchmark configure has bug
set(CPUINFO_BUILD_PKG_CONFIG  OFF)
SET(CPUINFO_LIBRARY_TYPE "default")
add_subdirectory(third_party/cpuinfo EXCLUDE_FROM_ALL)

add_library(thpool third_party/C-Thread-Pool/thpool.c)
target_include_directories(thpool PUBLIC third_party/C-Thread-Pool)

add_subdirectory(third_party/pybind11)

option(BUILD_SHARED_LIBS "Build shared libs" OFF) # 需要加入, 否则其并不是一个option
# option(GALOIS_BUILD_EXAMPLES "Build Prajna examples" ON)
# option(GALOIS_BUILD_TESTS "Build Prajna tests" ON)

add_subdirectory(galois)
add_subdirectory(tests)
