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
        self->Tensor(ir_value);
        self->type = ir_type;
        self->tag = "BitCast";
        return self;
    }

    std::shared_ptr<Tensor> Tensor() const { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<class Tensor> ir_value) { this->SetOperand(0, ir_value); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<BitCast>(this->shared_from_this()));
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

class Slice : public Instruction {
   public:
    static std::shared_ptr<Slice> Create(std::shared_ptr<Accessor> ir_origin,
                                         Eigen::VectorXi64 shape) {
        GALOIS_ASSERT(ir_origin->Tensor()->type->shape.size() == shape.size());
        std::shared_ptr<Slice> self(new Slice);
        self->OperandResize(1);
        self->Origin(ir_origin);
        self->shape = shape;

        auto stride = ir_origin->Tensor()->type->stride;
        GALOIS_ASSERT(ir_origin->Tensor()->type->value_type);
        self->type = ir::TensorType::Create(ir_origin->Tensor()->type->value_type, shape, stride);
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

class SqueezeDim : public Instruction {
   public:
    static std::shared_ptr<SqueezeDim> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<SqueezeDim> self(new SqueezeDim);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

        auto shape = ir_tensor->type->shape;
        auto stride = ir_tensor->type->stride;
        RemoveRow(shape, dim);
        RemoveColumn(stride, dim);
        self->type = TensorType::Create(ir_tensor->type->value_type, shape, stride);
        self->dim = dim;
        self->tag = "SqueezeDim";
        return self;
    }

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<SqueezeDim>(this->shared_from_this()));
    }

    int64_t dim;
};

class Squeeze : public Instruction {
   public:
    static std::shared_ptr<Squeeze> Create(std::shared_ptr<Tensor> ir_tensor) {
        std::shared_ptr<Squeeze> self(new Squeeze);
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
        interpreter->Visit(Cast<Squeeze>(this->shared_from_this()));
    }
};

class UnsqueezeDim : public Instruction {
   public:
    static std::shared_ptr<UnsqueezeDim> Create(std::shared_ptr<Tensor> ir_tensor, int64_t dim) {
        std::shared_ptr<UnsqueezeDim> self(new UnsqueezeDim);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

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

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<UnsqueezeDim>(this->shared_from_this()));
    }

    int64_t dim;
};

class Flatten : public Instruction {
   public:
    static std::shared_ptr<Flatten> Create(std::shared_ptr<Tensor> ir_tensor) {
        GALOIS_ASSERT(ir_tensor);
        std::shared_ptr<Flatten> self(new Flatten);
        self->OperandResize(1);
        self->Tensor(ir_tensor);

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

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Flatten>(this->shared_from_this()));
    }
};

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
        self->Tensor(ir_tensor);
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

    std::shared_ptr<ir::Tensor> Tensor() { return this->GetOperand(0); }
    void Tensor(std::shared_ptr<ir::Tensor> ir_tensor) { this->SetOperand(0, ir_tensor); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Transpose>(this->shared_from_this()));
    }

    int64_t dim0;
    int64_t dim1;
};

}  // namespace galois::ir::view
