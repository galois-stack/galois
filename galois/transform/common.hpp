#pragma once

#include <map>
#include <memory>
#include <set>

#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/transform/each.hpp"

namespace galois::transform {

inline std::set<std::shared_ptr<ir::Tensor>> CaptureExternalTensors(
    std::shared_ptr<ir::Block> ir_block) {
    std::set<std::shared_ptr<ir::Tensor>> ir_captured_tensor_set;

    Each<ir::Instruction>(ir_block, [&](std::shared_ptr<ir::Instruction> ir_instruction) {
        for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
            auto ir_operand = ir_instruction->GetOperand(i);
            if (!ir_operand->IsInsideOf(ir_block) && !Is<ir::Operator>(ir_operand)) {
                ir_captured_tensor_set.insert(ir_operand);
            }
        }
    });

    return ir_captured_tensor_set;
}

template <typename Tensor_>
inline std::list<std::shared_ptr<Tensor_>> GetAll(std::shared_ptr<ir::Block> ir_block) {
    std::list<std::shared_ptr<Tensor_>> ir_values;
    Each<Tensor_>(ir_block, [&ir_values](std::shared_ptr<Tensor_> ir_tensor) {
        ir_values.push_back(ir_tensor);
    });
    return ir_values;
}

inline void ApplyTransformMatrix(std::shared_ptr<ir::Grid> op, Eigen::Matrix2Xi transform_matrix) {
    // auto t_matrix = transform_matrix.transpose();
    // op->shape =
    //     ((t_matrix * transform_matrix).Cast<double>().inverse() *
    //     op->shape.Cast<double>())
    //         .Cast<int>();

    // for (auto ir_instruction : op->instructions) {
    //     // if (auto ir_accessor = Cast<ir::Accessor>(ir_instruction)) {
    //     //     ir_accessor->transform_matrix = ir_accessor->transform_matrix * transform_matrix;
    //     // }
    // }
}

}  // namespace galois::transform
