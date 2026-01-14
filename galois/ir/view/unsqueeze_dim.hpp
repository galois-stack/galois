#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class UnsqueezeDim : public Instruction {
   public:
    static std::shared_ptr<UnsqueezeDim> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<UnsqueezeDim> self(new UnsqueezeDim);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;

        auto old_shape = ir_tensor->type->shape;
        auto old_stride = ir_tensor->type->stride;
        int64_t old_rank = old_shape.size();
        int64_t new_rank = old_rank + 1;

        Eigen::VectorXi64 shape(new_rank);
        shape.head(dim) = old_shape.head(dim);
        shape(dim) = 1;
        shape.tail(new_rank - dim - 1) = old_shape.tail(old_rank - dim);

        Eigen::VectorXi64 stride(new_rank);
        stride.head(dim) = old_stride.head(dim);
        int64_t inserted_stride = (dim < old_stride.size()) ? old_stride(dim) : 1;
        stride(dim) = inserted_stride;
        stride.tail(new_rank - dim - 1) = old_stride.tail(old_rank - dim);

        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->dim = dim;
        self->tag = "UnsqueezeDim";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<UnsqueezeDim>(this->shared_from_this()));
    }

   public:
    int64_t dim;
    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
