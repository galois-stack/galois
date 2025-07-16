// error_calculation.cpp
#include "error_calculation.h"
#include <fmt/core.h>

void calculate_error(void* mem_a, void* mem_b, void* mem_c, int normalize_m, int normalize_k, int normalize_n) {
    py::scoped_interpreter guard{};

    // 使用Python的NumPy和PyTorch进行误差计算
    py::module_ torch = py::module_::import("torch");
    
    // 创建PyTorch张量
    py::array_t<float> arr_a({normalize_m, normalize_k}, static_cast<float*>(mem_a));
    py::array_t<float> arr_b({normalize_k, normalize_n}, static_cast<float*>(mem_b));
    py::array_t<float> arr_c_custom({normalize_m, normalize_n}, static_cast<float*>(mem_c));
    
    // 转换为PyTorch张量
    py::object tensor_a = torch.attr("from_numpy")(arr_a);
    py::object tensor_b = torch.attr("from_numpy")(arr_b);
    py::object tensor_c_custom = torch.attr("from_numpy")(arr_c_custom);
    
    // 执行矩阵乘法
    py::object tensor_c = tensor_a.attr("matmul")(tensor_b);
    
    // 计算误差指标
    py::object abs_error = tensor_c.attr("sub")(tensor_c_custom).attr("abs")();
    float max_error = abs_error.attr("max")().attr("item")().cast<float>();
    float mean_error = abs_error.attr("mean")().attr("item")().cast<float>();

    fmt::print("Max Error: {}\n", max_error);
    fmt::print("Mean Error: {}\n", mean_error);
    fmt::print("Results are close: {}\n", (max_error < 1e-5) ? "Yes" : "No");
}