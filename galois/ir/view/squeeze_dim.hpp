#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class SqueezeDim : public Instruction {
   public:
    static std::shared_ptr<SqueezeDim> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<SqueezeDim> self(new SqueezeDim);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;

        auto shape = ir_tensor->type->shape;
        auto stride = ir_tensor->type->stride;
        RemoveRow(shape, dim);
        RemoveColumn(stride, dim);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->dim = dim;
        self->tag = "SqueezeDim";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeDim>(this->shared_from_this()));
    }

   public:
    int64_t dim;
    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
