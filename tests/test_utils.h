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

} // namespace galois::test
