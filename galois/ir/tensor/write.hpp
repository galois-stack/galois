#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Write : public Instruction {
   public:
    static std::shared_ptr<Write> Create(std::shared_ptr<Tensor> value,
                                         std::shared_ptr<Tensor> accessor) {
        std::shared_ptr<Write> self(new Write);
        GALOIS_ASSERT(value->type == accessor->type);
        self->OperandResize(2);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Variable = OperandProperty(Cast<Instruction>(self->shared_from_this()), 1);
        self->Tensor = value;
        self->Variable = accessor;
        self->tag = "Write";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Write>(this->shared_from_this()));
    }

   public:
    OperandProperty Tensor = OperandProperty(nullptr, 0);
    OperandProperty Variable = OperandProperty(nullptr, 1);
};

}  // namespace galois::ir
