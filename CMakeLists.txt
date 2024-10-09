cmake_minimum_required(VERSION 3.14)

project(galois CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # 用于智能提示
set(CMAKE_CXX_STANDARD 17) # c++20目前系统库还不支持, 没有必要
set(CXX_STANDARD_REQUIRED ON)  # 若不支持c++17选项则报错，而不是使用c++14
set(CMAKE_CXX_EXTENSIONS OFF)  # 使用 -std=c++17而不是 -std=gnu++17

# 在mac平台下， 因为llvm设置为hidden，所以避免编译warning这里全局设置为hidden
# 之所以不用target_link_options设置，是因为其不能设置第三方项目
if (NOT BUILD_SHARED_LIBS)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# llvm compiler-rt不支持"Linux", 所以需要改为"linxu-gnu"
if (UNIX AND NOT APPLE)
    set(CMAKE_SYSTEM_NAME "linux-gnu")

    # llvm compiler-rt需要, 交叉编译需要重新设置
    set(CMAKE_C_COMPILER_TARGET "${CMAKE_SYSTEM_PROCESSOR}-unknown-${CMAKE_SYSTEM_NAME}")
    # 不可设置C++的, 会导致头文件找不到
    # set(CMAKE_CXX_COMPILER_TARGET "${CMAKE_SYSTEM_PROCESSOR}-unknown-${CMAKE_SYSTEM_NAME}")
    message("galois build target triple: ${CMAKE_C_COMPILER_TARGET}")

    set(COMPILER_RT_DEFAULT_TARGET_ONLY ON)
    set(COMPILER_RT_BUILD_BUILTINS ON)
    set(COMPILER_RT_BUILTINS_HIDE_SYMBOLS Off)
endif()

# add_subdirectory(third_party/llvm-project/llvm EXCLUDE_FROM_ALL)
# # 需要手动添加llvm的头文件路径
# add_library(llvm_include_dir INTERFACE)
# target_include_directories(llvm_include_dir
#     INTERFACE
#     $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/include>
#     $<INSTALL_INTERFACE:include>
#     INTERFACE
#     $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/include>
#     $<INSTALL_INTERFACE:include>
# )

add_subdirectory(prajna)

# use third_party from prajna
# add_subdirectory(third_party/boost EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/cxxopts EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/mtensor EXCLUDE_FROM_ALL)
add_subdirectory(third_party/eigen EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/json EXCLUDE_FROM_ALL) # prajna depends

set(CPUINFO_BUILD_UNIT_TESTS  OFF)
set(CPUINFO_BUILD_MOCK_TESTS  OFF)
set(CPUINFO_BUILD_BENCHMARKS  OFF) # disable benchmark, the googletest benchmark configure has bug
set(CPUINFO_BUILD_PKG_CONFIG  OFF)
add_subdirectory(third_party/cpuinfo EXCLUDE_FROM_ALL)
# add_subdirectory(third_party/z3 EXCLUDE_FROM_ALL)

add_library(thpool third_party/C-Thread-Pool/thpool.c)
target_include_directories(thpool PUBLIC third_party/C-Thread-Pool)

option(BUILD_SHARED_LIBS "Build shared libs" OFF) # 需要加入, 否则其并不是一个option
# option(GALOIS_BUILD_EXAMPLES "Build Prajna examples" ON)
# option(GALOIS_BUILD_TESTS "Build Prajna tests" ON)

add_subdirectory(galois)
add_subdirectory(tests)
