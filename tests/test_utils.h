#pragma once

#include <memory>
#include <string>
#include <tuple>
#include "gtest/gtest.h"  
#include "tests/galois_test.hpp"

namespace galois::test {

inline std::string ToString(const std::shared_ptr<ir::TensorType>& type) {
    auto dtype = type->DataType();

    if (auto float_type = std::dynamic_pointer_cast<ir::FloatType>(dtype)) {
        if (float_type->bits == 32) return "f32";
        if (float_type->bits == 64) return "f64";
    }

    if (auto int_type = std::dynamic_pointer_cast<ir::IntType>(dtype)) {
        if (int_type->bits == 32 && int_type->is_signed) return "i32";
        if (int_type->bits == 8 && int_type->is_signed) return "i8";
    }

    return "unknown";
}

// 支持 PrintToStringParamName  
inline std::string PrintTestName(
    const testing::TestParamInfo<std::tuple<std::shared_ptr<ir::TensorType>, int64_t, int64_t, int64_t>>& info) {

    const auto& [ir_data_type, m, k, n] = info.param;
    return ToString(ir_data_type) + "_" + std::to_string(m) + "_" +
           std::to_string(k) + "_" + std::to_string(n);
}


// 自定义测试名称生成器
inline std::string PrintTestNameWithVector(
    const testing::TestParamInfo<std::tuple<std::shared_ptr<ir::TensorType>, std::vector<int64_t>, float>>& info) {
    const auto& [ir_data_type, shape, value] = info.param;

    std::ostringstream shape_ss;
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_ss << shape[i];
        if (i < shape.size() - 1) shape_ss << "_";
    }

    std::ostringstream value_ss;
    value_ss << static_cast<int>(value);   
    std::string value_str = value_ss.str();
    return ToString(ir_data_type) + "_" + shape_ss.str() + "_fill_" + value_str;
}

} // namespace galois::test
