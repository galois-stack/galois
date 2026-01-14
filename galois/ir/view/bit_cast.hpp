#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class BitCast : public Instruction {
   public:
    static std::shared_ptr<BitCast> Create(std::shared_ptr<Tensor> ir_value,
                                           std::shared_ptr<TensorType> ir_type) {
        GALOIS_ASSERT(ir_type);
        std::shared_ptr<BitCast> self(new BitCast);
        GALOIS_ASSERT(ir_value->type->bytes == ir_type->bytes);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_value;
        self->type = ir_type;
        self->tag = "BitCast";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<BitCast>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
