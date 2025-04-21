#pragma once

#include <memory>

#include "galois/ir/tensor.hpp"

namespace galois::transform {

class OperationCountVisitor : public ir::Visitor {
   protected:
    OperationCountVisitor() = default;

   public:
    static std::shared_ptr<OperationCountVisitor> Create() {
        auto self = std::shared_ptr<OperationCountVisitor>(new OperationCountVisitor);
        return self;
    }
    int64_t GetOperationCount() { return operation_count; }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        for (auto tensor : *ir_block) {
            tensor->ApplyVisitor(this->shared_from_this());
        }
    }
    void Visit(std::shared_ptr<ir::Grid> ir_grid) override {
        // 计算shape的元素总数
        int64_t element_count = 1;
        for (int64_t dim : ir_grid->shape) {
            element_count *= dim;
        }

        // 压入新的计数器，用于block的操作数
        operation_count_stack.push(0);

        // 操作数累加到栈顶
        ir_grid->block->ApplyVisitor(this->shared_from_this());

        // 弹出block的操作数，乘以element_count，累加到全局operation_count
        int64_t block_operation_count = operation_count_stack.top();
        operation_count_stack.pop();
        operation_count += block_operation_count * element_count;
    }

    void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) override {

        int64_t op_count = ir_arithmetic_instruction->Size();
        if (!operation_count_stack.empty()) {
            operation_count_stack.top() += op_count;
        } else {
            operation_count += op_count;
        }
    }

    void Visit(std::shared_ptr<ir::PthreadBlock> ir_pthread_block) override {
        for (auto tensor : *ir_pthread_block) {
            tensor->ApplyVisitor(shared_from_this());
        }
    }

    void Visit(std::shared_ptr<ir::Call> ir_call) override {
        ir_call->Operator()->ApplyVisitor(this->shared_from_this());
    }
    void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {
    
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        ir_operator->block->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<Broadcast> ir_broadcast) override {
        
    }

    void Visit(std::shared_ptr<VectorBroadcast> ir_vbroadcast) override {
       
    }

    void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override {}
    void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) override {}
    void Visit(std::shared_ptr<ir::Instruction> ir_instruction) override {}

    void Visit(std::shared_ptr<ir::BitCast> ir_bit_cast) override {}
    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {}
    void Visit(std::shared_ptr<ir::Free> ir_free) override {}
    void Visit(std::shared_ptr<ir::Return> ir_return) override {}
    void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override {}

    void Visit(std::shared_ptr<ir::Write> ir_write) override {}

    void Visit(std::shared_ptr<ir::Viewer> ir_viewer) override {}
    void Visit(std::shared_ptr<ir::SqueezeDimView> ir_squeeze_dim_view) override {}
    void Visit(std::shared_ptr<ir::SliceView> ir_slice_view) override {}
    void Visit(std::shared_ptr<ir::SqueezeView> ir_squeeze_view) override {}

   private:
    int64_t operation_count = 0;
    std::stack<int64_t> operation_count_stack;
};

}  // namespace galois::transform