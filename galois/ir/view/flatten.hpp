#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class Flatten : public Instruction {
   public:
    static std::shared_ptr<Flatten> Create(std::shared_ptr<Tensor> ir_tensor) {
        GALOIS_ASSERT(ir_tensor);
        std::shared_ptr<Flatten> self(new Flatten);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;

        int64_t total_size = 1;
        const auto& shape = ir_tensor->type->shape;
        for (int64_t i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }

        Eigen::VectorXi64 new_shape(1);
        new_shape(0) = total_size;

        Eigen::VectorXi64 new_stride(1);
        new_stride(0) = 1;

        self->type = TensorType::Create(ir_tensor->type->value_type, new_shape, new_stride);
        self->tag = "Flatten";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Flatten>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
