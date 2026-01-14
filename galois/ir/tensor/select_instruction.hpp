#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class SelectInstruction : public Instruction {
   public:
    static std::shared_ptr<SelectInstruction> Create(std::shared_ptr<Tensor> ir_condition,
                                                     std::shared_ptr<Tensor> ir_true_value,
                                                     std::shared_ptr<Tensor> ir_false_value) {
        GALOIS_ASSERT(ir_condition);
        GALOIS_ASSERT(ir_true_value);
        GALOIS_ASSERT(ir_false_value);
        GALOIS_ASSERT(ir_true_value->type == ir_false_value->type);

        if (ir_condition->type->IsScalar()) {
            GALOIS_ASSERT(ir_condition->type->DataType() == ir::bool_);
        } else {
            GALOIS_ASSERT(ir_condition->type->DataType() == ir::bool_);
            GALOIS_ASSERT(ir_condition->type->shape.size() == ir_true_value->type->shape.size());
            for (int64_t i = 0; i < ir_condition->type->shape.size(); ++i) {
                GALOIS_ASSERT(ir_condition->type->shape[i] == ir_true_value->type->shape[i] ||
                              ir_condition->type->shape[i] == 1 ||
                              ir_true_value->type->shape[i] == 1);
            }
        }

        std::shared_ptr<SelectInstruction> self(new SelectInstruction);
        self->OperandResize(3);
        self->Condition(ir_condition);
        self->TrueValue(ir_true_value);
        self->FalseValue(ir_false_value);
        self->type = ir_true_value->type;
        self->tag = "SelectInstruction";
        return self;
    }

    std::shared_ptr<Tensor> Condition() { return this->GetOperand(0); }
    void Condition(std::shared_ptr<Tensor> ir_condition) { this->SetOperand(0, ir_condition); }

    std::shared_ptr<Tensor> TrueValue() { return this->GetOperand(1); }
    void TrueValue(std::shared_ptr<Tensor> ir_true_value) { this->SetOperand(1, ir_true_value); }

    std::shared_ptr<Tensor> FalseValue() { return this->GetOperand(2); }
    void FalseValue(std::shared_ptr<Tensor> ir_false_value) { this->SetOperand(2, ir_false_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SelectInstruction>(this->shared_from_this()));
    }
};

}  // namespace galois::ir
