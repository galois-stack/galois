#pragma once

#include "galois/ir/tensor/base.hpp"

namespace galois::ir {

class VectorBroadcast : public Instruction {
   protected:
    VectorBroadcast() = default;

   public:
    static std::shared_ptr<VectorBroadcast> Create(std::shared_ptr<Tensor> ir_value,
                                                   std::shared_ptr<ir::TensorType> ir_type,
                                                   int64_t lane_id) {
        std::shared_ptr<VectorBroadcast> self(new VectorBroadcast);
        self->OperandResize(1);
        self->Vector = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Vector = ir_value;
        self->lane_id = lane_id;
        self->type = ir_type;
        self->tag = "VectorBroadcast";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<VectorBroadcast>(this->shared_from_this()));
    }

    int64_t lane_id;
    OperandProperty Vector = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir
