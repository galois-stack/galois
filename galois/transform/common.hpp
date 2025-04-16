#pragma once

#include <map>
#include <set>

#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"

namespace galois::transform {

inline void EachTensor(std::shared_ptr<ir::Block> ir_block,
                       std::shared_ptr<galois::ir::Visitor> visitor);

inline void EachTensor(std::shared_ptr<ir::Tensor> ir_value,
                       std::shared_ptr<galois::ir::Visitor> visitor) {
    if (auto ir_block = Cast<ir::Block>(ir_value)) {
        EachTensor(ir_block, visitor);
    } else {
        ir_value->ApplyVisitor(visitor);
    }
}

inline void EachTensor(std::shared_ptr<ir::Block> ir_block,
                       std::shared_ptr<galois::ir::Visitor> visitor) {
    ir_block->ApplyVisitor(visitor);
    for (auto ir_tensor : ir_block->tensors) {
        EachTensor(ir_tensor, visitor);
    }
}

template <typename Value_>
class EachVisitor : public galois::ir::Visitor {
   private:
    std::function<void(std::shared_ptr<Value_>)> callback_;

   public:
    EachVisitor(std::function<void(std::shared_ptr<Value_>)> callback) : callback_(callback) {}

    void Visit(std::shared_ptr<Value_> value) override { callback_(value); }
};
template <typename Value_>
inline void Each(std::shared_ptr<ir::Block> ir_block,
                 std::function<void(std::shared_ptr<Value_>)> callback) {
    auto visitor = std::make_shared<EachVisitor<Value_>>(callback);
    EachTensor(ir_block, visitor);
}

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

template <typename Matrix_>
inline void RemoveRow(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (index < numRows)
        matrix.block(index, 0, numRows - index, numCols) = matrix.bottomRows(numRows - index);

    matrix.conservativeResize(numRows, numCols);
}

template <typename Matrix_>
inline void RemoveColumn(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (index < numCols)
        matrix.block(0, index, numRows, numCols - index) = matrix.rightCols(numCols - index);

    matrix.conservativeResize(numRows, numCols);
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
