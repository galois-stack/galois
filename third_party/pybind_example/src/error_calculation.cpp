// error_calculation.cpp
#include "pybind_example.h"
#include <fmt/core.h>

void init_python() {
    static bool initialized = false;
    if (!initialized) {
        pybind11::initialize_interpreter();
        initialized = true;
    }
}

void calculate_error(void* mem_a, void* mem_b, void* mem_c, int normalize_m, int normalize_k, int normalize_n) {
    init_python();

    // pybind11::scoped_interpreter guard{};

    // 使用pybind11thon的Numpybind11和pybind11Torch进行误差计算
    pybind11::module_ torch = pybind11::module_::import("torch");
    
    // 创建pybind11Torch张量
    pybind11::array_t<float> arr_a({normalize_m, normalize_k}, static_cast<float*>(mem_a));
    pybind11::array_t<float> arr_b({normalize_k, normalize_n}, static_cast<float*>(mem_b));
    pybind11::array_t<float> arr_c_custom({normalize_m, normalize_n}, static_cast<float*>(mem_c));
    
    // 转换为pybind11Torch张量
    pybind11::object tensor_a = torch.attr("from_numpy")(arr_a);
    pybind11::object tensor_b = torch.attr("from_numpy")(arr_b);
    pybind11::object tensor_c_custom = torch.attr("from_numpy")(arr_c_custom);
    
    // 执行矩阵乘法
    pybind11::object tensor_c = tensor_a.attr("matmul")(tensor_b);
    
    // 计算误差指标
    pybind11::object abs_error = tensor_c.attr("sub")(tensor_c_custom).attr("abs")();
    float max_error = abs_error.attr("max")().attr("item")().cast<float>();
    float mean_error = abs_error.attr("mean")().attr("item")().cast<float>();

    // 使用 printf 打印结果
    printf("Maximum error: %.6f\n", max_error);
    printf("Average Error: %.6f\n", mean_error);
}