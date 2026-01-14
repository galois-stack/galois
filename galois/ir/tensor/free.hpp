#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Free : public Instruction {
   protected:
    Free() = default;

   public:
    static std::shared_ptr<Free> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<Free> self(new Free);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;
        self->tag = "Free";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Free>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
