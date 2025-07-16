#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <memory>

// 定义一个函数来计算误差
void calculate_error(void* mem_a, void* mem_b, void* mem_c, int normalize_m, int normalize_k, int normalize_n);