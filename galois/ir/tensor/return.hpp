#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Return : public Instruction {
   protected:
    Return() = default;

   public:
    static std::shared_ptr<Return> Create(std::shared_ptr<Tensor> ir_value) {
        std::shared_ptr<Return> self(new Return);
        self->OperandResize(1);
        self->type = ir_value->type;
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_value;
        self->tag = "Return";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Return>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
