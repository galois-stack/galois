#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class Slice : public Instruction {
   public:
    static std::shared_ptr<Slice> Create(std::shared_ptr<Accessor> ir_origin,
                                         Eigen::VectorXi64 shape) {
        GALOIS_ASSERT(ir_origin->Tensor->type->shape.size() == shape.size());
        std::shared_ptr<Slice> self(new Slice);
        self->OperandResize(1);
        self->Origin(ir_origin);
        self->shape = shape;

        auto stride = ir_origin->Tensor->type->stride;
        GALOIS_ASSERT(ir_origin->Tensor->type->value_type);
        self->type = ir::TensorType::Create(ir_origin->Tensor->type->value_type, shape, stride);
        self->tag = "Slice";
        return self;
    }

    std::shared_ptr<Accessor> Origin() { return Cast<Accessor>(this->GetOperand(0)); }
    void Origin(std::shared_ptr<ir::Accessor> ir_accessor) { this->SetOperand(0, ir_accessor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Slice>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
};

}  // namespace galois::ir::view
