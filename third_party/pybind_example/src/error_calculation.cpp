// error_calculation.cpp
#include "pybind_example.h"
#include <fmt/core.h>

void print_matrix(const char* name, pybind11::array_t<float>& arr) {
    if (arr.ndim() != 2) {  // 用 ndim() 判断是否为二维矩阵
        printf("%s is not a 2D matrix\n", name);
        return;
    }

    const ssize_t* shape = arr.shape();
    int rows = shape[0];  // 行数（第一个维度）
    int cols = shape[1];  // 列数（第二个维度）
    
    // 获取矩阵数据的访问器
    auto arr_access = arr.unchecked<2>();  // 2D矩阵
    
    printf("Matrix %s (%d rows, %d cols):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        printf("[ ");
        for (int j = 0; j < cols; ++j) {
            printf("%.6f ", arr_access(i, j));
        }
        printf("]\n");
    }
    printf("\n");
}

void print_torch_tensor(const char* name, pybind11::object& tensor) {
    pybind11::array_t<float> arr = tensor.attr("cpu")().attr("numpy")().cast<pybind11::array_t<float>>();
    print_matrix(name, arr);  // 复用上面的矩阵打印函数
}

void init_python() {
    static bool initialized = false;
    if (!initialized) {
        pybind11::initialize_interpreter();
        initialized = true;
    }
}

void calculate_error(void* mem_a, void* mem_b, void* mem_c, int normalize_m, int normalize_k, int normalize_n) {
    init_python();

    pybind11::module_ torch = pybind11::module_::import("torch");
    
    pybind11::array_t<float> arr_a({normalize_m, normalize_k}, static_cast<float*>(mem_a));
    pybind11::array_t<float> arr_b({normalize_k, normalize_n}, static_cast<float*>(mem_b));
    pybind11::array_t<float> arr_c({normalize_m, normalize_n}, static_cast<float*>(mem_c));
    
    pybind11::object tensor_a = torch.attr("from_numpy")(arr_a);
    pybind11::object tensor_b = torch.attr("from_numpy")(arr_b);
    pybind11::object tensor_c = tensor_a.attr("matmul")(tensor_b);

    // printf("===== 输入矩阵 arr_a (自定义计算的输入A) =====\n");
    // print_matrix("arr_a", arr_a);

    // printf("===== 输入矩阵 arr_b (自定义计算的输入B) =====\n");
    // print_matrix("arr_b", arr_b);

    pybind11::array_t<float> arr_c_tensor = tensor_c.attr("cpu")()
                                            .attr("numpy")()
                                            .cast<pybind11::array_t<float>>();

    auto shape = arr_c_tensor.shape();
    if (shape[0] != normalize_m || shape[1] != normalize_n) {
        printf("Error: tensor_c shape mismatch! Expected (%d, %d), got (%lld, %lld)\n",
               normalize_m, normalize_n,
               static_cast<long long>(shape[0]), static_cast<long long>(shape[1]));
        return;
    }
    
    const float* c_tensor_data = arr_c_tensor.data();
    float* c_mem_data = static_cast<float*>(mem_c);
    
    size_t data_size = normalize_m * normalize_n * sizeof(float); 
    memcpy(c_mem_data, c_tensor_data, data_size);
}