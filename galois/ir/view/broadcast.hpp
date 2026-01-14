#pragma once

#include "../tensor.hpp"

namespace galois::ir::view {

class Broadcast : public Instruction {
   public:
    static std::shared_ptr<Broadcast> Create(std::shared_ptr<Tensor> ir_value,
                                             Eigen::MatrixXi64 output_shape) {
        std::shared_ptr<Broadcast> self(new Broadcast);
        self->OperandResize(1);
        self->Tensor = OperandProperty(Cast<Instruction>(self->shared_from_this()), 0);
        self->Tensor = ir_value;

        if (ir_value->type->IsScalar()) {
            auto new_stride = Eigen::VectorXi64::Zero(output_shape.size());
            self->type =
                ir::TensorType::Create(ir_value->type->DataType(), output_shape, new_stride);
        } else {
            GALOIS_ASSERT(ir_value->type->shape.size() == output_shape.size());
            auto input_shape = ir_value->type->shape;
            auto new_stride = ir_value->type->stride;
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (input_shape[i] == 1) {
                    new_stride[i] = 0;
                }
            }
            self->type =
                ir::TensorType::Create(ir_value->type->DataType(), output_shape, new_stride);
        }

        self->tag = "Broadcast";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Broadcast>(this->shared_from_this()));
    }

    OperandProperty Tensor = OperandProperty(nullptr, 0);
};

}  // namespace galois::ir::view
