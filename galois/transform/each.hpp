#pragma once

#include <memory>

#include "galois/ir/tensor.hpp"

namespace galois::transform {


template <typename Value_>
class EachTensorVisitor : public ir::Visitor {
   protected:
    EachTensorVisitor() = default;

   public:
    static std::shared_ptr<EachTensorVisitor<Value_>> Create(
        std::function<void(std::shared_ptr<Value_>)> callback) {
        auto self = std::shared_ptr<EachTensorVisitor<Value_>>(new EachTensorVisitor);
        self->callback_ = callback;
        return self;
    }


    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        for (auto tensor : *ir_block) {
            tensor->ApplyVisitor(this->shared_from_this());
        }
    }

     void Visit(std::shared_ptr<ir::Grid> ir_grid) override {
        ir_grid->block->ApplyVisitor(this->shared_from_this());
     }
     void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override {
        this->callback(ir_accessor);
     }
     void Visit(std::shared_ptr<ir::GridIndex> ir_grid_index) override {
        this->callback(ir_grid_index);
     }
     void Visit(std::shared_ptr<ir::Instruction> ir_instruction) override {
        this->callback(ir_instruction);
     }
     void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arithmetic_instruction) override {
        this->callback(ir_arithmetic_instruction);
     }
     void Visit(std::shared_ptr<ir::BitCast> ir_bit_cast) override {
        this->callback(ir_bit_cast);
     }
     void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {
        this->callback(ir_alloca);
     }
     void Visit(std::shared_ptr<ir::Free> ir_free) override {
        this->callback(ir_free);
     }
     void Visit(std::shared_ptr<ir::Return> ir_return) override {
        this->callback(ir_return);
     }
     void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override {
        this->callback(ir_prefetch);
     }
     void Visit(std::shared_ptr<ir::PthreadBlock> ir_pthread_block) override {
        this->callback(ir_pthread_block);
     }
     void Visit(std::shared_ptr<ir::Write> ir_write) override {
        this->callback(ir_write);
     }
     void Visit(std::shared_ptr<ir::VectorBroadcast> ir_vector_broadcast) override {
        this->callback(ir_vector_broadcast);
     }
     void Visit(std::shared_ptr<ir::Broadcast> ir_broadcast) override {
        this->callback(ir_broadcast);
     }
     void Visit(std::shared_ptr<ir::Call> ir_call) override {
        this->callback(ir_call);
     }
     void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {
        this->callback(ir_unary_intrinsic);
     }
     void Visit(std::shared_ptr<ir::Viewer> ir_viewer) override {
        this->callback(ir_viewer);
     }
     void Visit(std::shared_ptr<ir::SqueezeDimView> ir_squeeze_dim_view) override {
        this->callback(ir_squeeze_dim_view);
     }
     void Visit(std::shared_ptr<ir::SliceView> ir_slice_view) override {
        this->callback(ir_slice_view);
     }
     void Visit(std::shared_ptr<ir::SqueezeView> ir_squeeze_view) override {
        this->callback(ir_squeeze_view);
     }
     void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        ir_operator->block->ApplyVisitor(this->shared_from_this());
     }
     void Visit(std::shared_ptr<ir::Constant> ir_constant) override {
        this->callback(ir_constant);
     }
     void Visit(std::shared_ptr<ir::ConstantRealNumber> ir_constant_real_number) override {
        this->callback(ir_constant_real_number);
     }
     void Visit(std::shared_ptr<ir::ConstantInt> ir_constant_int) override {
        this->callback(ir_constant_int);
     }
     void Visit(std::shared_ptr<ir::ConstantFloat> ir_constant_float) override {
        this->callback(ir_constant_float);
     }

    
   private:
    void callback(std::shared_ptr<ir::Tensor> ir_tensor) {
        auto value = Cast<Value_>(ir_tensor);
        GALOIS_ASSERT(value);
        callback_(value);
    } 

   private:
    std::function<void(std::shared_ptr<Value_>)> callback_;
};

template <typename Value_>
inline void Each(std::shared_ptr<ir::Tensor> ir_tensor,
                 std::function<void(std::shared_ptr<Value_>)> callback) {
    auto visitor = EachTensorVisitor<Value_>::Create(callback);
    ir_tensor->ApplyVisitor(visitor);
}




}  // namespace galois::transform