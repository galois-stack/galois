#pragma once

#include <memory>
#include <stack>

#include "galois/ir/tensor.hpp"

namespace galois::ir {

class OperationCounter : public ir::Visitor {
   protected:
    OperationCounter() = default;

   public:
    static std::shared_ptr<OperationCounter> Create() {
        auto self = std::shared_ptr<OperationCounter>(new OperationCounter);
        return self;
    }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        this->operation_count_stack.push(0);
        for (auto ir_tensor : *ir_block) {
            if (Is<ir::Operator>(ir_tensor)) continue;  // 内嵌的Operator, 不做处理
            ir_tensor->ApplyVisitor(this->shared_from_this());
        }
        auto block_operation_count = this->operation_count_stack.top();
        this->operation_count_stack.pop();
        this->operation_count_stack.top() += block_operation_count;
    }

    void Visit(std::shared_ptr<ir::Grid> ir_grid) override {
        // 计算shape的元素总数
        int64_t grid_iterations = 1;
        for (int64_t dim : ir_grid->shape) {
            grid_iterations *= dim;
        }

        operation_count_stack.push(0);
        ir_grid->block->ApplyVisitor(this->shared_from_this());
        int64_t block_operation_count = operation_count_stack.top();
        operation_count_stack.pop();
        this->operation_count_stack.top() += block_operation_count * grid_iterations;
    }

    void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) override {
        operation_count_stack.top() += ir_arithmetic_instruction->type->NormalizeSize();
    }

    void Visit(std::shared_ptr<ir::Call> ir_call) override {
        this->operation_count_stack.push(0);
        ir_call->Operator()->ApplyVisitor(this->shared_from_this());
        auto block_operation_count = this->operation_count_stack.top();
        this->operation_count_stack.pop();
        this->operation_count_stack.top() += block_operation_count;
    }

    void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {}

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        this->operation_count_stack.push(0);
        ir_operator->block->ApplyVisitor(this->shared_from_this());
        auto block_operation_count = this->operation_count_stack.top();
        this->operation_count_stack.pop();
        this->operation_count_stack.top() = block_operation_count;
    }

    int64_t CountOperation(std::shared_ptr<ir::Tensor> ir_tensor) {
        this->operation_count_stack.push(0);
        ir_tensor->ApplyVisitor(this->shared_from_this());
        auto block_operation_count = this->operation_count_stack.top();
        this->operation_count_stack.pop();
        return block_operation_count;
    }

    void Visit(std::shared_ptr<ir::VectorBroadcast> ir_vbroadcast) override {}

    void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override {}
    void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) override {}
    void Visit(std::shared_ptr<ir::Instruction> ir_instruction) override {}

    void Visit(std::shared_ptr<ir::view::BitCast> ir_bit_cast) override {}
    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {}
    void Visit(std::shared_ptr<ir::Free> ir_free) override {}
    void Visit(std::shared_ptr<ir::Return> ir_return) override {}
    void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override {}

    void Visit(std::shared_ptr<ir::Write> ir_write) override {}

    void Visit(std::shared_ptr<ir::view::Viewer> ir_viewer) override {}
    void Visit(std::shared_ptr<ir::view::SqueezeDim> ir_squeeze_dim_view) override {}
    void Visit(std::shared_ptr<ir::view::UnsqueezeDim> ir_unsqueeze_dim_view) override {}
    void Visit(std::shared_ptr<ir::view::Slice> ir_slice_view) override {}
    void Visit(std::shared_ptr<ir::view::Squeeze> ir_squeeze_view) override {}
    void Visit(std::shared_ptr<ir::view::Flatten> ir_flatten_view) override {}
    void Visit(std::shared_ptr<ir::view::Transpose> ir_transpose_view) override {}

   private:
    std::stack<int64_t> operation_count_stack;
};

}  // namespace galois::ir
