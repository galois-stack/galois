#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::view {

class Squeeze : public Instruction {
   public:
    static std::shared_ptr<Squeeze> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<Squeeze> self(new Squeeze);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_tensor;

        int64_t valid_shape_size = 0;
        Eigen::VectorXi64 shape(ir_tensor->type->shape.size());
        Eigen::VectorXi64 stride(ir_tensor->type->stride.size());
        for (int64_t i = 0; i < ir_tensor->type->shape.size(); ++i) {
            if (ir_tensor->type->shape[i] != 1) {
                shape[valid_shape_size] = ir_tensor->type->shape[i];
                stride[valid_shape_size] = ir_tensor->type->stride[i];
                valid_shape_size++;
            }
        }
        shape.conservativeResize(valid_shape_size);
        stride.conservativeResize(valid_shape_size);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->tag = "Squeeze";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Squeeze>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
