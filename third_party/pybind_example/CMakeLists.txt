cmake_minimum_required(VERSION 3.14)
project(pybind_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_EXAMPLE_BASE_DIR "/home/wanrui/miniconda3/envs/torch-1.13/")

# list(APPEND CMAKE_PREFIX_PATH
#     $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11
#     $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Torch
# )

set(pybind11_DIR ${PYBIND11_EXAMPLE_BASE_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
set(Torch_DIR ${PYBIND11_EXAMPLE_BASE_DIR}/lib/python3.10/site-packages/torch/share/cmake/Torch)

set(Python3_EXECUTABLE ${PYBIND11_EXAMPLE_BASE_DIR}/bin/python3.10)
find_package(Python3 COMPONENTS Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)
if(pybind11_FOUND)
    message(STATUS "Found pybind11: ${pybind11_DIR}")
    message(STATUS "pybind11 include directories: ${pybind11_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find pybind11")
endif()

find_package(Torch REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include  # 项目公共头文件
    ${Python3_INCLUDE_DIRS}        # Python头文件
    ${pybind11_INCLUDE_DIRS}
)

file(GLOB_RECURSE PYBIND_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(pybind_example STATIC ${PYBIND_EXAMPLE_SOURCES})

target_link_libraries(pybind_example
    PUBLIC pybind11::pybind11
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE Python3::Python
)

target_include_directories(pybind_example
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)