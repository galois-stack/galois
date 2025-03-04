#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/binary_operator.hpp"

namespace galois::op {

using namespace ir;

class MatrixMultiplyKernel {
   public:
    virtual bool Match(std::shared_ptr<TensorType> ir_mat_type_a,
                       std::shared_ptr<TensorType> ir_mat_type_b) = 0;

    virtual void Express(std::shared_ptr<Tensor> ir_mat_a, std::shared_ptr<Tensor> ir_mat_b,
                         std::shared_ptr<Tensor> ir_mat_c, std::shared_ptr<Builder> ir_builder) = 0;
};

class MatrixMultiplyKernel4x1x4 : public MatrixMultiplyKernel {
   public:
    static std::shared_ptr<MatrixMultiplyKernel4x1x4> Create() {
        return std::make_shared<MatrixMultiplyKernel4x1x4>();
    }

    bool Match(std::shared_ptr<TensorType> ir_mat_type_a,
               std::shared_ptr<TensorType> ir_mat_type_b) override {
        if ((ir_mat_type_a == TensorType::CreateMatrixType(FloatType::Create(32), 4, 1)) &&
            ir_mat_type_b == TensorType::CreateMatrixType(FloatType::Create(32), 1, 4)) {
            return true;
        }
        return false;
    }

    void Express(std::shared_ptr<Tensor> ir_mat_a, std::shared_ptr<Tensor> ir_mat_b,
                 std::shared_ptr<Tensor> ir_mat_c, std::shared_ptr<Builder> ir_builder) override {
        Eigen::VectorXi64 v4(1);
        v4[0] = 4;
        auto ir_f32x4_type = TensorType::Create(FloatType::Create(32), v4);
        auto ir_bit_cast_a = ir_builder->Create<BitCast>(ir_mat_a, ir_f32x4_type);
        auto ir_bit_cast_b = ir_builder->Create<BitCast>(ir_mat_b, ir_f32x4_type);
        auto ir_bit_cast_c =
            ir_builder->Create<BitCast>(ir_mat_c, TensorType::Create(ir_f32x4_type, v4));

        for (int64_t i = 0; i < 4; ++i) {
            auto ir_vector_broadcast_a = ir_builder->Create<VectorBroadcast>(ir_bit_cast_a, i);
            auto ir_mul = ir_builder->Create<Mul>(ir_vector_broadcast_a, ir_bit_cast_b);
            auto ir_accessor_c = ir_builder->CreateAccessor(ir_bit_cast_c);
            ir_accessor_c->shift_vector[0] = i;
            auto ir_sum = ir_builder->Create<Add>(ir_mul, ir_accessor_c);
            auto ir_write =
                ir_builder->Create<Write>(ir_sum, Cast<Accessor>(ir_accessor_c->Clone()));
        }
    }
};

class MatrixMultiplyKernelI8_16x1x16 : public MatrixMultiplyKernel {
   public:
    static std::shared_ptr<MatrixMultiplyKernelI8_16x1x16> Create() {
        return std::make_shared<MatrixMultiplyKernelI8_16x1x16>();
    }

    bool Match(std::shared_ptr<TensorType> ir_mat_type_a,
               std::shared_ptr<TensorType> ir_mat_type_b) override {
        if ((ir_mat_type_a == TensorType::CreateMatrixType(IntType::Create(8, true), 16, 1)) &&
            ir_mat_type_b == TensorType::CreateMatrixType(IntType::Create(8, true), 1, 16)) {
            return true;
        }
        return false;
    }

    void Express(std::shared_ptr<Tensor> ir_mat_a, std::shared_ptr<Tensor> ir_mat_b,
                 std::shared_ptr<Tensor> ir_mat_c, std::shared_ptr<Builder> ir_builder) override {
        Eigen::VectorXi64 v16(1);
        v16[0] = 16;
        auto ir_i8x16_type = TensorType::Create(IntType::Create(8, true), v16);
        auto ir_bit_cast_a = ir_builder->Create<BitCast>(ir_mat_a, ir_i8x16_type);
        auto ir_bit_cast_b = ir_builder->Create<BitCast>(ir_mat_b, ir_i8x16_type);
        auto ir_bit_cast_c =
            ir_builder->Create<BitCast>(ir_mat_c, TensorType::Create(ir_i8x16_type, v16));

        for (int64_t i = 0; i < 16; ++i) {
            auto ir_vector_broadcast_a = ir_builder->Create<VectorBroadcast>(ir_bit_cast_a, i);
            auto ir_mul = ir_builder->Create<Mul>(ir_vector_broadcast_a, ir_bit_cast_b);
            auto ir_accessor_c = ir_builder->CreateAccessor(ir_bit_cast_c);
            ir_accessor_c->shift_vector[0] = i;
            auto ir_sum = ir_builder->Create<Add>(ir_mul, ir_accessor_c);
            auto ir_write =
                ir_builder->Create<Write>(ir_sum, Cast<Accessor>(ir_accessor_c->Clone()));
        }
    }
};

class MatrixMultiplyKernel8x1x8 : public MatrixMultiplyKernel {
   public:
    static std::shared_ptr<MatrixMultiplyKernel8x1x8> Create() {
        return std::make_shared<MatrixMultiplyKernel8x1x8>();
    }

    bool Match(std::shared_ptr<TensorType> ir_mat_type_a,
               std::shared_ptr<TensorType> ir_mat_type_b) override {
        if ((ir_mat_type_a == TensorType::CreateMatrixType(FloatType::Create(32), 8, 1)) &&
            ir_mat_type_b == TensorType::CreateMatrixType(FloatType::Create(32), 1, 8)) {
            return true;
        }
        return false;
    }

    void Express(std::shared_ptr<Tensor> ir_mat_a, std::shared_ptr<Tensor> ir_mat_b,
                 std::shared_ptr<Tensor> ir_mat_c, std::shared_ptr<Builder> ir_builder) override {
        Eigen::VectorXi64 v8(1);
        v8[0] = 8;
        auto ir_f32x8_type = TensorType::Create(FloatType::Create(32), v8);
        auto ir_bit_cast_a = ir_builder->Create<BitCast>(ir_mat_a, ir_f32x8_type);
        auto ir_bit_cast_b = ir_builder->Create<BitCast>(ir_mat_b, ir_f32x8_type);
        auto ir_bit_cast_c =
            ir_builder->Create<BitCast>(ir_mat_c, TensorType::Create(ir_f32x8_type, v8));

        for (int64_t i = 0; i < 8; ++i) {
            auto ir_vector_broadcast_a = ir_builder->Create<VectorBroadcast>(ir_bit_cast_a, i);
            auto ir_mul = ir_builder->Create<Mul>(ir_vector_broadcast_a, ir_bit_cast_b);
            auto ir_accessor_c = ir_builder->CreateAccessor(ir_bit_cast_c);
            ir_accessor_c->shift_vector[0] = i;
            auto ir_sum = ir_builder->Create<Add>(ir_mul, ir_accessor_c);
            auto ir_write =
                ir_builder->Create<Write>(ir_sum, Cast<Accessor>(ir_accessor_c->Clone()));
        }
    }
};

class MatrixMultiplyCreator : public BinaryOperatorCreator {
   public:
    static std::shared_ptr<MatrixMultiplyCreator> Create() {
        return std::make_shared<MatrixMultiplyCreator>();
    }

    std::shared_ptr<TensorType> InferTypeImpl(std::shared_ptr<TensorType> ir_mat_a_type,
                                              std::shared_ptr<TensorType> ir_mat_b_type) override {
        if (ir_mat_a_type->IsScalar() && ir_mat_b_type->IsScalar()) {
            GALOIS_ASSERT(ir_mat_a_type == ir_mat_a_type);
            return ir_mat_a_type;
        }

        auto ir_value_type =
            this->InferType({ir_mat_a_type->value_type, ir_mat_b_type->value_type});
        return TensorType::CreateMatrixType(ir_value_type, ir_mat_a_type->shape[0],
                                            ir_mat_b_type->shape[1]);
    }

    void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_mat_a,
                           std::shared_ptr<ir::Tensor> ir_mat_b,
                           std::shared_ptr<ir::Tensor> ir_mat_c,
                           std::shared_ptr<Builder> ir_builder) override {
        for (auto ir_kernel : ir_builder->kernel_queue) {
            if (ir_kernel->Match(ir_mat_a->type, ir_mat_b->type)) {
                ir_kernel->Express(ir_mat_a, ir_mat_b, ir_mat_c, ir_builder);
                return;
            }
        }

        if (ir_mat_a->type->IsScalar()) {
            auto ir_re =
                ir_builder->Create<Add>(ir_builder->Create<Mul>(ir_mat_a, ir_mat_b), ir_mat_c);
            ir_builder->Create<Write>(ir_re, ir_mat_c);
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

   private:
    std::shared_ptr<SetZeroCreator> set_zero_creator = SetZeroCreator::Create();
};

}  // namespace galois::op
