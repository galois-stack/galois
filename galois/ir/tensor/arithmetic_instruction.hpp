#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class ArithmeticInstruction : public Instruction {
   public:
    enum Operation { Add, Sub, Mul, Div };

    static std::shared_ptr<ArithmeticInstruction> Create(Operation op,
                                                         std::shared_ptr<Tensor> ir_operand0,
                                                         std::shared_ptr<Tensor> ir_operand1) {
        GALOIS_ASSERT(ir_operand0->type == ir_operand1->type);
        std::shared_ptr<ArithmeticInstruction> self(new ArithmeticInstruction);
        self->operation = op;
        self->OperandResize(2);
        self->Operand_var0 = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Operand_var1 = OperandProperty(Cast<Instruction>(self->shared_from_this()), 1);
        self->Operand_var0 = ir_operand0;
        self->Operand_var1 = ir_operand1;
        self->type = ir_operand0->type;
        self->tag = "ArithmeticInstruction";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<ArithmeticInstruction>(this->shared_from_this()));
    }

   public:
    Operation operation;
    OperandProperty Operand_var0 = OperandProperty(nullptr, 0);
    OperandProperty Operand_var1 = OperandProperty(nullptr, 1);
};

}  // namespace galois::ir
