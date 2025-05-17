#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir {

class BitCastView : public Instruction {
   public:
    static std::shared_ptr<BitCastView> Create(std::shared_ptr<Tensor> ir_value,
                                               std::shared_ptr<TensorType> ir_type) {
        GALOIS_ASSERT(ir_type);
        std::shared_ptr<BitCastView> self(new BitCastView);
        GALOIS_ASSERT(ir_value->type->bytes == ir_type->bytes);
        self->OperandResize(1);
        self->Tensor(ir_value);
        self->type = ir_type;
        self->tag = "BitCastView";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> ir_value) { this->SetOperand(0, ir_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<BitCastView>(this->shared_from_this()));
    }
};

class Viewer : public Instruction {
   public:
    static std::shared_ptr<Viewer> Create(std::shared_ptr<Tensor> ir_tensor,
                                          Eigen::MatrixXi64 transform_matrix,
                                          Eigen::VectorXi64 shift_vector) {
        std::shared_ptr<Viewer> self(new Viewer);
        self->ir_tensor = ir_tensor;
        self->transform_matrix = transform_matrix;
        self->type = ir_tensor->type;
        self->shift_vector = shift_vector;
        self->tag = "Viewer";
        return self;
    }

    static std::shared_ptr<Viewer> Shift(std::shared_ptr<Tensor> ir_tensor,
                                         Eigen::VectorXi64 shift_vector) {
        auto tensor_rank = ir_tensor->type->shape.size();
        auto identity_matrix = Eigen::MatrixXi64::Identity(tensor_rank, tensor_rank);
        GALOIS_ASSERT(shift_vector.size() == tensor_rank);
        return Create(ir_tensor, identity_matrix, shift_vector);
    }

    static std::shared_ptr<Viewer> Stride(std::shared_ptr<Tensor> ir_tensor,
                                          Eigen::VectorXi64 stride_vector) {
        auto tensor_rank = ir_tensor->type->shape.size();
        GALOIS_ASSERT(stride_vector.size() == tensor_rank);
        Eigen::MatrixXi64 transform_matrix = Eigen::MatrixXi64::Zero(tensor_rank, tensor_rank);
        for (int64_t i = 0; i < tensor_rank; ++i) {
            transform_matrix(i, i) = stride_vector[i];
        }
        return Create(ir_tensor, transform_matrix, Eigen::VectorXi64::Zero(tensor_rank));
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Viewer>(this->shared_from_this()));
    }

    Eigen::MatrixXi64 transform_matrix;
    Eigen::VectorXi64 shift_vector;
    std::shared_ptr<Tensor> ir_tensor = nullptr;
};

class SliceView : public Instruction {
   public:
    static std::shared_ptr<SliceView> Create(std::shared_ptr<Accessor> ir_origin,
                                             Eigen::VectorXi64 shape) {
        GALOIS_ASSERT(ir_origin->Tensor()->type->shape.size() == shape.size());
        std::shared_ptr<SliceView> self(new SliceView);
        self->OperandResize(1);
        self->Origin(ir_origin);
        self->shape = shape;

        auto stride = ir_origin->Tensor()->type->stride;
        GALOIS_ASSERT(ir_origin->Tensor()->type->value_type);
        self->type = ir::TensorType::Create(ir_origin->Tensor()->type->value_type, shape, stride);
        self->tag = "SliceView";
        return self;
    }

    std::shared_ptr<Accessor> Origin() { return Cast<Accessor>(this->GetOperand(0)); }
    void Origin(std::shared_ptr<ir::Accessor> ir_accessor) { this->SetOperand(0, ir_accessor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SliceView>(this->shared_from_this()));
    }

    Eigen::VectorXi64 shape;
};

class SqueezeDimView : public Instruction {
   public:
    static std::shared_ptr<SqueezeDimView> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<SqueezeDimView> self(new SqueezeDimView);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

        auto shape = ir_tensor->type->shape;
        auto stride = ir_tensor->type->stride;
        RemoveRow(shape, dim);
        RemoveColumn(stride, dim);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->dim = dim;
        self->tag = "Squeeze";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeDimView>(this->shared_from_this()));
    }

    int64_t dim;
};

class SqueezeView : public Instruction {
   public:
    static std::shared_ptr<SqueezeView> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<SqueezeView> self(new SqueezeView);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

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

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeView>(this->shared_from_this()));
    }
};

}  // namespace galois::ir
