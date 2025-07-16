cmake_minimum_required(VERSION 3.14)
project(pybind_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH
    $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11
    $ENV{CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Torch
)

set(Python3_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3.10)
find_package(Python3 COMPONENTS Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include  # 项目公共头文件
    ${Python3_INCLUDE_DIRS}        # Python头文件
)

file(GLOB_RECURSE PYBIND_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(pybind_example STATIC ${PYBIND_EXAMPLE_SOURCES})

target_link_libraries(pybind_example
    PRIVATE pybind11::pybind11
    PRIVATE ${TORCH_LIBRARIES}
    PRIVATE Python3::Python
)

target_include_directories(pybind_example
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)