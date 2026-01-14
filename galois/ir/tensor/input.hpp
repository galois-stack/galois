#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class Input : public Tensor {
   public:
    static std::shared_ptr<Input> Create(std::shared_ptr<TensorType> ir_type) {
        std::shared_ptr<Input> self(new Input);
        self->type = ir_type;
        self->tag = "Input";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Input>(this->shared_from_this()));
    }
};

}  // namespace galois::ir
