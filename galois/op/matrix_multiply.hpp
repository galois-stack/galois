#pragma once

#include "galois/op/binary.hpp"

namespace galois::op {

class MatrixMultiplyKernel {
   public:
    virtual bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
                       std::shared_ptr<ir::TensorType> ir_mat_type_b) = 0;

    virtual void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                         std::shared_ptr<ir::Tensor> ir_mat_c,
                         std::shared_ptr<ir::Builder> ir_builder) = 0;
};

class SimdMatrixMultiplyKernel : public MatrixMultiplyKernel {
   public:
    static std::shared_ptr<SimdMatrixMultiplyKernel>  Create(int64_t bits) {
        std::shared_ptr<SimdMatrixMultiplyKernel> self(new SimdMatrixMultiplyKernel);
        self->bits = bits;
        // self->sim_cols = simd_cols;
        self->bytes = self->bits / 8;
        return self;
    }

    bool IsVectorized(std::shared_ptr<ir::TensorType> ir_type) {
        if (ir_type->shape.size() == 2) {
            if (ir_type->shape[0] == 1 || ir_type->shape[1] == 1) {
                if (ir_type->bytes == this->bytes) {
                    return true;
                }
            }
        }

        return false;
    }

    bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
               std::shared_ptr<ir::TensorType> ir_mat_type_b) override {
        GALOIS_ASSERT(ir_mat_type_a->shape.size() == 2);
        GALOIS_ASSERT(ir_mat_type_b->shape.size() == 2);
        auto simd_lanes = this->bytes / ir_mat_type_a->value_type->bytes;
        if (ir_mat_type_a->value_type == ir_mat_type_b->value_type) {
            if (ir_mat_type_a->shape[1] == 1 && ir_mat_type_a->shape[0] <= simd_lanes) {
                if (ir_mat_type_b->shape[0] == 1 && ir_mat_type_b->shape[1] == simd_lanes) {
                    return true;
                }
            }
        }

        return false;
    }

    void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                 std::shared_ptr<ir::Tensor> ir_mat_c,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        Eigen::VectorXi64 lanes_a(1);
        lanes_a[0] = ir_mat_a->type->shape[0];
        Eigen::VectorXi64 lanes_b(1);
        lanes_b[0] = ir_mat_b->type->shape[1];
        auto ir_data_type = ir_mat_a->type->DataType();
        auto ir_simd_type_a = ir::TensorType::Create(ir_data_type, lanes_a);
        auto ir_simd_type_b = ir::TensorType::Create(ir_data_type, lanes_b);

        auto ir_vec_bit_cast_a = ir_builder->Create<ir::BitCast>(ir_mat_a, ir_simd_type_a);
        auto ir_vec_bit_cast_b = ir_builder->Create<ir::BitCast>(ir_mat_b, ir_simd_type_b);
        auto ir_mat_bit_cast_c = ir_builder->Create<ir::BitCast>(
            ir_mat_c, ir::TensorType::Create(ir_simd_type_b, lanes_a));

        for (int64_t i = 0; i < lanes_a[0]; ++i) {
            auto ir_vector_broadcast_a = ir_builder->Create<ir::VectorBroadcast>(ir_vec_bit_cast_a, ir_simd_type_b, i);
            auto ir_mul = ir_builder->Create<ir::Mul>(ir_vector_broadcast_a, ir_vec_bit_cast_b);
            auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_bit_cast_c);
            ir_accessor_c->shift_vector[0] = i;
            auto ir_sum = ir_builder->Create<ir::Add>(ir_mul, ir_accessor_c);
            auto ir_write =
                ir_builder->Create<ir::Write>(ir_sum, Cast<ir::Accessor>(ir_accessor_c->Clone()));
        }
    }

   private:
   int64_t bits = 128;
   int64_t bytes = 16;
   int64_t simd_cols;
};

class MatrixMultiplyCreator : public BinaryCreator {
   public:
    static std::shared_ptr<MatrixMultiplyCreator> Create() {
        auto self = std::make_shared<MatrixMultiplyCreator>();
        self->name = "MatrixMultiply";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_mat_a_type,
        std::shared_ptr<ir::TensorType> ir_mat_b_type) override {
        if (ir_mat_a_type->IsScalar() && ir_mat_b_type->IsScalar()) {
            GALOIS_ASSERT(ir_mat_a_type == ir_mat_b_type);
            return ir_mat_a_type;
        }

        auto ir_value_type =
            this->InferType({ir_mat_a_type->value_type, ir_mat_b_type->value_type});
        return ir::TensorType::CreateMatrixType(ir_value_type, ir_mat_a_type->shape[0],
                                                ir_mat_b_type->shape[1]);
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_mat_a,
                           std::shared_ptr<ir::Tensor> ir_mat_b,
                           std::shared_ptr<ir::Tensor> ir_mat_c,
                           std::shared_ptr<ir::Builder> ir_builder) override {
        for (auto ir_kernel : ir_builder->matrix_multiply_kernel_queue) {
            if (ir_kernel->Match(ir_mat_a->type, ir_mat_b->type)) {
                ir_kernel->Express(ir_mat_a, ir_mat_b, ir_mat_c, ir_builder);
                return;
            }
        }

        if (ir_mat_a->type->IsScalar()) {
            auto ir_re = ir_builder->Create<ir::Add>(
                ir_builder->Create<ir::Mul>(ir_mat_a, ir_mat_b), ir_mat_c);
            ir_builder->Create<ir::Write>(ir_re, ir_mat_c);
            return;
        }

        GALOIS_ASSERT(ir_mat_a->type->shape[1] == ir_mat_b->type->shape[0]);
        GALOIS_ASSERT(ir_mat_c->type->shape[0] == ir_mat_a->type->shape[0]);
        GALOIS_ASSERT(ir_mat_c->type->shape[1] == ir_mat_b->type->shape[1]);

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(Eigen::Vector3i64(
            ir_mat_a->type->shape[0], ir_mat_a->type->shape[1], ir_mat_b->type->shape[1]));
        // std::unique_ptr<ScopeGuard> pthread_block_scope;
        // ir_grid->enable_multi_thread = ir_mat_a->type->enable_multi_thread;

        auto ir_accessor_a = ir_builder->CreateAccessor(ir_mat_a);
        ir_accessor_a->transform_matrix(0, 0) = 1;
        ir_accessor_a->transform_matrix(1, 1) = 1;
        auto ir_accessor_b = ir_builder->CreateAccessor(ir_mat_b);
        ir_accessor_b->transform_matrix(0, 1) = 1;
        ir_accessor_b->transform_matrix(1, 2) = 1;
        auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_c);
        ir_accessor_c->transform_matrix(0, 0) = 1;
        ir_accessor_c->transform_matrix(1, 2) = 1;

        this->AffineExpressImpl(ir_accessor_a, ir_accessor_b, ir_accessor_c, ir_builder);
    }
};

}  // namespace galois::op
