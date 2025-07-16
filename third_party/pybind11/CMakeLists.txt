# third_party/pybind11/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/wanrui/miniconda3/envs/torch-1.13/lib/python3.10/site-packages/torch/share/cmake/Torch/)
set(Python3_EXECUTABLE /home/wanrui/miniconda3/envs/torch-1.13/bin/python)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})

add_library(error_calculation STATIC error_calculation.cpp)

target_include_directories(error_calculation PUBLIC ${pybind11_INCLUDE_DIRS})
target_link_libraries(error_calculation PUBLIC pybind11::pybind11 ${TORCH_LIBRARIES} ${Python3_LIBRARIES})