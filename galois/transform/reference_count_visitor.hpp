#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::transform {

class ReferenceCountVisitor;

class IncreaseVisitor : public ir::Visitor {
   protected:
    IncreaseVisitor() = default;

   public:
    static std::shared_ptr<IncreaseVisitor> Create(
        std::shared_ptr<std::unordered_map<std::shared_ptr<ir::Tensor>, int64_t>>
            sp_reference_count_dict) {
        auto self = std::shared_ptr<IncreaseVisitor>(new IncreaseVisitor);
        self->sp_reference_count_dict = sp_reference_count_dict;
        return self;
    }

    void IncreaseViewInstruction(std::shared_ptr<ir::Instruction> ir_instruction) {
        this->IncreaseReferenceCount(ir_instruction);
        for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
            auto ir_operand = ir_instruction->GetOperand(i);
            this->IncreaseReferenceCount(ir_operand);
        }
    }

    void Visit(std::shared_ptr<ir::BitCastView> ir_bit_cast) override {
        this->IncreaseViewInstruction(ir_bit_cast);
    }

    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {
        this->IncreaseReferenceCount(ir_alloca);
    }

    void Visit(std::shared_ptr<ir::Free> ir_free) override {
        (*sp_reference_count_dict)[ir_free->Tensor()] = -1;  // 置为-1， 我们只有0时才会插入free指令
    }

    void Visit(std::shared_ptr<ir::Return> ir_return) override {
        this->IncreaseViewInstruction(ir_return);
    }

    void Visit(std::shared_ptr<ir::Write> ir_write) override {}

    void Visit(std::shared_ptr<ir::Call> ir_call) override {
        this->IncreaseReferenceCount(ir_call);
    }

    void Visit(std::shared_ptr<ir::Viewer> ir_viewer) override {
        this->IncreaseViewInstruction(ir_viewer);
    }

    void Visit(std::shared_ptr<ir::SqueezeDimView> ir_squeeze_dim_view) override {
        this->IncreaseViewInstruction(ir_squeeze_dim_view);
    }

    void Visit(std::shared_ptr<ir::UnsqueezeDimView> ir_unsqueeze_dim_view) override {
        this->IncreaseViewInstruction(ir_unsqueeze_dim_view);
    }

    void Visit(std::shared_ptr<ir::SliceView> ir_slice_view) override {
        this->IncreaseViewInstruction(ir_slice_view);
    }

    void Visit(std::shared_ptr<ir::SqueezeView> ir_squeeze_view) override {
        this->IncreaseViewInstruction(ir_squeeze_view);
    }

    void Visit(std::shared_ptr<ir::FlattenView> ir_flatten_view) override {
        this->IncreaseViewInstruction(ir_flatten_view);
    }

    void Visit(std::shared_ptr<ir::TransposeView> ir_transpose_view) override {
        this->IncreaseViewInstruction(ir_transpose_view);
    }

    void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override {
        this->IncreaseViewInstruction(ir_accessor);
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        ir_operator->block->ApplyVisitor(shared_from_this());
    }

   private:
    void IncreaseReferenceCount(std::shared_ptr<ir::Tensor> ir_tensor) {
        auto &ref_count_dict = *sp_reference_count_dict;
        if (!ref_count_dict.count(ir_tensor)) {
            ref_count_dict[ir_tensor] = 1;
        } else {
            ref_count_dict[ir_tensor]++;
        }
    }

   private:
    std::shared_ptr<std::unordered_map<std::shared_ptr<ir::Tensor>, int64_t>>
        sp_reference_count_dict;
};

class DecreaseVisitor : public ir::Visitor {
   protected:
    DecreaseVisitor() = default;

   private:
    static void FreeTensor(std::shared_ptr<ir::Tensor> ir_tensor) {
        auto ir_parent_block = Lock(ir_tensor->parent_block);
        auto ir_free = ir::Free::Create(ir_tensor);
        ir_free->parent_block = ir_parent_block;
        auto iter =
            std::find_if(RANGE((*ir_parent_block)), [](auto ir_x) { return Is<ir::Return>(ir_x); });
        ir_parent_block->insert(iter, ir_free);
    }

   public:
    static std::shared_ptr<DecreaseVisitor> Create(
        std::shared_ptr<std::unordered_map<std::shared_ptr<ir::Tensor>, int64_t>>
            sp_reference_count_dict) {
        auto self = std::shared_ptr<DecreaseVisitor>(new DecreaseVisitor);
        self->sp_reference_count_dict = sp_reference_count_dict;
        return self;
    }

    void DecreaseViewInstruction(std::shared_ptr<ir::Instruction> ir_instruction) {
        this->DecreaseReferenceCount(ir_instruction);
        if (this->IsFree(ir_instruction)) {
            for (int64_t i = 0; i < ir_instruction->OperandSize(); ++i) {
                auto ir_operand = ir_instruction->GetOperand(i);
                ir_operand->ApplyVisitor(shared_from_this());
            }
        }
    }

    void Visit(std::shared_ptr<ir::BitCastView> ir_bit_cast) override {
        this->DecreaseViewInstruction(ir_bit_cast);
    }

    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {
        this->DecreaseReferenceCount(ir_alloca);
        if (this->IsFree(ir_alloca)) {
            FreeTensor(ir_alloca);
        }
    }

    void Visit(std::shared_ptr<ir::Return> ir_return) override {
        // return 指令不释放
    }

    void Visit(std::shared_ptr<ir::Write> ir_write) override {}

    void Visit(std::shared_ptr<ir::Call> ir_call) override {
        this->DecreaseReferenceCount(ir_call);
        if (this->IsFree(ir_call) && !Is<ir::VoidType>(ir_call->type)) {
            FreeTensor(ir_call);
        }
    }

    void Visit(std::shared_ptr<ir::Viewer> ir_viewer) override {
        this->DecreaseViewInstruction(ir_viewer);
    }

    void Visit(std::shared_ptr<ir::SqueezeDimView> ir_squeeze_dim_view) override {
        this->DecreaseViewInstruction(ir_squeeze_dim_view);
    }

    void Visit(std::shared_ptr<ir::UnsqueezeDimView> ir_unsqueeze_dim_view) override {
        this->DecreaseViewInstruction(ir_unsqueeze_dim_view);
    }

    void Visit(std::shared_ptr<ir::SliceView> ir_slice_view) override {
        this->DecreaseViewInstruction(ir_slice_view);
    }

    void Visit(std::shared_ptr<ir::SqueezeView> ir_squeeze_view) override {
        this->DecreaseViewInstruction(ir_squeeze_view);
    }

    void Visit(std::shared_ptr<ir::FlattenView> ir_flatten_view) override {
        this->DecreaseViewInstruction(ir_flatten_view);
    }

    void Visit(std::shared_ptr<ir::TransposeView> ir_transpose_view) override {
        this->DecreaseViewInstruction(ir_transpose_view);
    }

    void Visit(std::shared_ptr<ir::Accessor> ir_accessor) override {
        this->DecreaseViewInstruction(ir_accessor);
    }

   private:
    void DecreaseReferenceCount(std::shared_ptr<ir::Tensor> ir_tensor) {
        auto &ref_count_dict = *sp_reference_count_dict;
        GALOIS_ASSERT(ref_count_dict.count(ir_tensor));
        // GALOIS_ASSERT(ref_count_dict[ir_tensor] > 0);
        --ref_count_dict[ir_tensor];
    }

    bool IsFree(std::shared_ptr<ir::Tensor> ir_tensor) {
        return (*this->sp_reference_count_dict)[ir_tensor] == 0;
    }

   private:
    std::shared_ptr<std::unordered_map<std::shared_ptr<ir::Tensor>, int64_t>>
        sp_reference_count_dict;
};

class ReferenceCountVisitor : public ir::Visitor {
   protected:
    ReferenceCountVisitor() = default;

   public:
    static std::shared_ptr<ReferenceCountVisitor> Create() {
        auto self = std::shared_ptr<ReferenceCountVisitor>(new ReferenceCountVisitor);
        auto sp_reference_count_dict =
            std::make_shared<std::unordered_map<std::shared_ptr<ir::Tensor>, int64_t>>();
        self->increase_visitor = IncreaseVisitor::Create(sp_reference_count_dict);
        self->decrease_visitor = DecreaseVisitor::Create(sp_reference_count_dict);
        return self;
    }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        // 进入Block时, 需要增加Tensor的引用计数
        for (auto ir_tensor : *ir_block) {
            ir_tensor->ApplyVisitor(this->increase_visitor);
        }

        // 推出Block时, 需要减少Tensor的引用计数
        for (auto ir_tensor : Clone(*ir_block)) {  // 拷贝tensors， 因为free tensor的时候会改变结构
            ir_tensor->ApplyVisitor(this->decrease_visitor);
        }
    }

    void Visit(std::shared_ptr<ir::Grid> ir_grid) override {
        ir_grid->block->ApplyVisitor(shared_from_this());
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        ir_operator->block->ApplyVisitor(shared_from_this());
    }

    std::shared_ptr<IncreaseVisitor> increase_visitor = nullptr;
    std::shared_ptr<DecreaseVisitor> decrease_visitor = nullptr;
};

}  // namespace galois::transform
