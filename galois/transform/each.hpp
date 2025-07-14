#pragma once

#include <memory>

#include "galois/ir/tensor.hpp"

namespace galois::transform {

class EachTensorVisitor : public ir::Visitor {
   protected:
    EachTensorVisitor() = default;

   public:
    static std::shared_ptr<EachTensorVisitor> Create(
        std::function<void(std::shared_ptr<ir::Tensor>)> callback) {
        auto self = std::shared_ptr<EachTensorVisitor>(new EachTensorVisitor);
        self->callback_ = callback;
        return self;
    }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        this->callback(ir_block);
        for (auto tensor : Clone(*ir_block)) {  // Each的过程中会修改
            tensor->ApplyVisitor(this->shared_from_this());
        }
    }

    void Visit(std::shared_ptr<ir::Grid> ir_grid) override {
        this->callback(ir_grid);
        ir_grid->block->ApplyVisitor(this->shared_from_this());
    }
    void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override { this->callback(ir_accessor); }
    void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) override {
        this->callback(ir_grid_index);
    }
    void Visit(std::shared_ptr<ir::Instruction> ir_instruction) override {
        this->callback(ir_instruction);
    }
    void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) override {
        this->callback(ir_arithmetic_instruction);
    }
    void Visit(std::shared_ptr<ir::CompareInstruction> ir_compare_instruction) override {
        this->callback(ir_compare_instruction);
    }
    void Visit(std::shared_ptr<ir::SelectInstruction> ir_select_instruction) override {
        this->callback(ir_select_instruction);
    }
    void Visit(std::shared_ptr<ir::view::BitCast> ir_bit_cast) override {
        this->callback(ir_bit_cast);
    }
    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override { this->callback(ir_alloca); }
    void Visit(std::shared_ptr<ir::Free> ir_free) override { this->callback(ir_free); }
    void Visit(std::shared_ptr<ir::Return> ir_return) override { this->callback(ir_return); }
    void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override { this->callback(ir_prefetch); }
    void Visit(std::shared_ptr<ir::Write> ir_write) override { this->callback(ir_write); }
    void Visit(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) override {
        this->callback(ir_vector_broadcast);
    }
    void Visit(std::shared_ptr<ir::Call> ir_call) override { this->callback(ir_call); }
    void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {
        this->callback(ir_unary_intrinsic);
    }
    void Visit(std::shared_ptr<ir::view::Viewer> ir_viewer) override { this->callback(ir_viewer); }
    void Visit(std::shared_ptr<ir::view::SqueezeDim> ir_squeeze_dim_view) override {
        this->callback(ir_squeeze_dim_view);
    }
    void Visit(std::shared_ptr<ir::view::UnsqueezeDim> ir_unsqueeze_dim_view) override {
        this->callback(ir_unsqueeze_dim_view);
    }
    void Visit(std::shared_ptr<ir::view::Slice> ir_slice_view) override {
        this->callback(ir_slice_view);
    }
    void Visit(std::shared_ptr<ir::view::Squeeze> ir_squeeze_view) override {
        this->callback(ir_squeeze_view);
    }
    void Visit(std::shared_ptr<ir::view::Flatten> ir_flatten_view) override {
        this->callback(ir_flatten_view);
    }
    void Visit(std::shared_ptr<ir::view::Transpose> ir_transpose_view) override {
        this->callback(ir_transpose_view);
    }
    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        this->callback(ir_operator);
        for (auto ir_input : ir_operator->inputs) {
            ir_input->ApplyVisitor(this->shared_from_this());
        }
        ir_operator->block->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<ir::Input> ir_input) override { this->callback(ir_input); }

    void Visit(std::shared_ptr<ir::Constant> ir_constant) override { this->callback(ir_constant); }
    void Visit(std::shared_ptr<ir::ConstantRealNumber> ir_constant_real_number) override {
        this->callback(ir_constant_real_number);
    }
    void Visit(std::shared_ptr<ir::ConstantInt> ir_constant_int) override {
        this->callback(ir_constant_int);
    }
    void Visit(std::shared_ptr<ir::ConstantFloat> ir_constant_float) override {
        this->callback(ir_constant_float);
    }

    void Visit(std::shared_ptr<ir::Indexing> ir_index) override { this->callback(ir_index); }

   private:
    void callback(std::shared_ptr<ir::Tensor> ir_tensor) {
        //   GALOIS_ASSERT(value);
        callback_(ir_tensor);
    }

   private:
    std::function<void(std::shared_ptr<ir::Tensor>)> callback_;
};

template <typename Value_>
inline void Each(std::shared_ptr<ir::Tensor> ir_tensor,
                 std::function<void(std::shared_ptr<Value_>)> callback) {
    auto visitor = EachTensorVisitor::Create([=](std::shared_ptr<ir::Tensor> ir_tensor) {
        if (auto ir_value = Cast<Value_>(ir_tensor)) {
            callback(ir_value);
        }
    });
    ir_tensor->ApplyVisitor(visitor);
}

}  // namespace galois::transform
