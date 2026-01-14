#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class Transpose : public Instruction {
   public:
    static std::shared_ptr<Transpose> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim0,
                                             int64_t dim1) {
        GALOIS_ASSERT(ir_tensor);
        const auto& old_shape = ir_tensor->type->shape;
        const auto& old_stride = ir_tensor->type->stride;
        int64_t rank = old_shape.size();
        GALOIS_ASSERT(dim0 >= 0 && dim0 < rank);
        GALOIS_ASSERT(dim1 >= 0 && dim1 < rank);
        GALOIS_ASSERT(dim0 != dim1);

        std::shared_ptr<Transpose> self(new Transpose);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;
        self->dim0 = dim0;
        self->dim1 = dim1;

        Eigen::VectorXi64 new_shape = old_shape;
        Eigen::VectorXi64 new_stride = old_stride;

        std::swap(new_shape(dim0), new_shape(dim1));
        std::swap(new_stride(dim0), new_stride(dim1));

        self->type = TensorType::Create(ir_tensor->type->value_type, new_shape, new_stride);
        self->tag = "Transpose";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Transpose>(this->shared_from_this()));
    }

   public:
    int64_t dim0;
    int64_t dim1;
    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
