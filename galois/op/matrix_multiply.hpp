#pragma once

#include "galois/op/binary.hpp"

namespace galois::op {

class MatrixMultiplyMicroKernel {
   public:
    virtual bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
                       std::shared_ptr<ir::TensorType> ir_mat_type_b) = 0;

    virtual void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                         std::shared_ptr<ir::Tensor> ir_mat_c,
                         std::shared_ptr<ir::Builder> ir_builder) = 0;
};

class SimdMatrixMultiplyKernel : public MatrixMultiplyMicroKernel {
   public:
    static std::shared_ptr<SimdMatrixMultiplyKernel> Create(int64_t bits) {
        std::shared_ptr<SimdMatrixMultiplyKernel> self(new SimdMatrixMultiplyKernel);
        self->bits = bits;
        // self->sim_cols = simd_cols;
        self->bytes = self->bits / 8;
        return self;
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

        // auto cloner = ir::Cloner::Create();
        for (int64_t i = 0; i < lanes_a[0]; ++i) {
            auto ir_vector_broadcast_a =
                ir_builder->Create<ir::VectorBroadcast>(ir_vec_bit_cast_a, ir_simd_type_b, i);
            auto ir_mul = ir_builder->Mul(ir_vector_broadcast_a, ir_vec_bit_cast_b);
            auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_bit_cast_c);
            ir_accessor_c->transform_matrix.resize(0, 0);
            ir_accessor_c->shift_vector[0] = i;
            auto ir_sum = ir_builder->Add(ir_mul, ir_accessor_c);
            auto ir_write = ir_builder->Create<ir::Write>(ir_sum, ir_accessor_c);
        }
    }

   private:
    int64_t bits = 128;
    int64_t bytes = 16;
    int64_t simd_cols;
};

class NeonMatrixMultiplyKernel : public MatrixMultiplyMicroKernel {
   public:
    static std::shared_ptr<NeonMatrixMultiplyKernel> Create(int64_t bits, int64_t rows,
                                                            int64_t cols) {
        std::shared_ptr<NeonMatrixMultiplyKernel> self(new NeonMatrixMultiplyKernel);
        self->bits = bits;
        self->bytes = self->bits / 8;
        self->rows = rows;
        self->cols = cols;
        return self;
    }

    bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
               std::shared_ptr<ir::TensorType> ir_mat_type_b) override {
        GALOIS_ASSERT(ir_mat_type_a->shape.size() == 2);
        GALOIS_ASSERT(ir_mat_type_b->shape.size() == 2);
        auto simd_lanes = this->bytes / ir_mat_type_a->value_type->bytes;
        if (ir_mat_type_a->value_type == ir_mat_type_b->value_type) {
            if (ir_mat_type_a->shape[1] == 1 && ir_mat_type_a->shape[0] == this->rows) {
                if (ir_mat_type_b->shape[0] == 1 && ir_mat_type_b->shape[1] == this->cols) {
                    return true;
                }
            }
        }

        return false;
    }

    void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                 std::shared_ptr<ir::Tensor> ir_mat_c,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        int64_t lanes_a = ir_mat_a->type->shape[0];
        int64_t lanes_b = ir_mat_b->type->shape[1];
        auto ir_data_type = ir_mat_a->type->DataType();
        auto simd_lanes = this->bytes / ir_data_type->bytes;
        GALOIS_ASSERT(lanes_a % simd_lanes == 0);
        GALOIS_ASSERT(lanes_b % simd_lanes == 0);
        auto ir_simd_type_a = ir_data_type->Tile(simd_lanes)->Tile(lanes_a / simd_lanes);
        auto ir_simd_type_b = ir_data_type->Tile(simd_lanes)->Tile(lanes_b / simd_lanes);

        auto ir_vec_bit_cast_a = ir_builder->Create<ir::BitCast>(ir_mat_a, ir_simd_type_a);
        auto ir_vec_bit_cast_b = ir_builder->Create<ir::BitCast>(ir_mat_b, ir_simd_type_b);
        auto ir_mat_bit_cast_c =
            ir_builder->Create<ir::BitCast>(ir_mat_c, ir_simd_type_b->Tile(lanes_a));

        for (int64_t r = 0; r < ir_simd_type_a->shape[0]; ++r) {
            auto ir_accessor_a = ir_builder->CreateAccessor(ir_vec_bit_cast_a);
            ir_accessor_a->transform_matrix.resize(0, 0);
            ir_accessor_a->shift_vector[0] = r;
            for (int64_t c = 0; c < ir_simd_type_b->shape[0]; ++c) {
                auto ir_accessor_b = ir_builder->CreateAccessor(ir_vec_bit_cast_b);
                ir_accessor_b->transform_matrix.resize(0, 0);
                ir_accessor_b->shift_vector[0] = c;

                for (int64_t i = 0; i < simd_lanes; ++i) {
                    auto ir_vector_broadcast_a = ir_builder->Create<ir::VectorBroadcast>(
                        ir_accessor_a, ir_accessor_b->type, i);
                    auto ir_mul = ir_builder->Mul(ir_vector_broadcast_a, ir_accessor_b);
                    auto ir_accessor_c_row = ir_builder->CreateAccessor(ir_mat_bit_cast_c);
                    ir_accessor_c_row->transform_matrix.resize(0, 0);
                    ir_accessor_c_row->shift_vector[0] = r * simd_lanes + i;
                    auto ir_accessor_c = ir_builder->CreateAccessor(ir_accessor_c_row);
                    ir_accessor_c->transform_matrix.resize(0, 0);
                    ir_accessor_c->shift_vector[0] = c;
                    auto ir_sum = ir_builder->Add(ir_mul, ir_accessor_c);
                    auto ir_write = ir_builder->Create<ir::Write>(ir_sum, ir_accessor_c);
                }
            }
        }
    }

   private:
    int64_t bits = 128;
    int64_t bytes = 16;
    int64_t rows;
    int64_t cols;
};

class AvxMatrixMultiplyKernel : public MatrixMultiplyMicroKernel {
   public:
    static std::shared_ptr<AvxMatrixMultiplyKernel> Create(int64_t bits, int64_t rows,
                                                           int64_t cols) {
        std::shared_ptr<AvxMatrixMultiplyKernel> self(new AvxMatrixMultiplyKernel);
        self->bits = bits;
        self->bytes = self->bits / 8;
        self->rows = rows;
        self->cols = cols;
        return self;
    }

    bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
               std::shared_ptr<ir::TensorType> ir_mat_type_b) override {
        GALOIS_ASSERT(ir_mat_type_a->shape.size() == 2);
        GALOIS_ASSERT(ir_mat_type_b->shape.size() == 2);
        auto simd_lanes = this->bytes / ir_mat_type_a->value_type->bytes;
        if (ir_mat_type_a->value_type == ir_mat_type_b->value_type) {
            if (ir_mat_type_a->shape[1] == 1 && ir_mat_type_a->shape[0] == this->rows) {
                if (ir_mat_type_b->shape[0] == 1 && ir_mat_type_b->shape[1] == this->cols) {
                    return true;
                }
            }
        }

        return false;
    }

    void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                 std::shared_ptr<ir::Tensor> ir_mat_c,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        int64_t lanes_a = ir_mat_a->type->shape[0];
        int64_t lanes_b = ir_mat_b->type->shape[1];
        auto ir_data_type = ir_mat_a->type->DataType();
        auto simd_lanes = this->bytes / ir_data_type->bytes;
        // GALOIS_ASSERT(lanes_a % simd_lanes == 0);
        GALOIS_ASSERT(lanes_b % simd_lanes == 0);
        auto ir_simd_type_a = ir_data_type->Tile(lanes_a);
        auto ir_simd_type_b = ir_data_type->Tile(simd_lanes)->Tile(lanes_b / simd_lanes);

        auto ir_vec_bit_cast_a = ir_builder->Create<ir::BitCast>(ir_mat_a, ir_simd_type_a);
        auto ir_vec_bit_cast_b = ir_builder->Create<ir::BitCast>(ir_mat_b, ir_simd_type_b);
        auto ir_mat_bit_cast_c =
            ir_builder->Create<ir::BitCast>(ir_mat_c, ir_simd_type_b->Tile(lanes_a));

        for (int64_t i = 0; i < lanes_a; ++i) {
            auto ir_accessor_a = ir_builder->CreateAccessor(ir_vec_bit_cast_a);
            ir_accessor_a->transform_matrix.resize(0, 0);
            ir_accessor_a->shift_vector[0] = i;
            auto ir_accessor_a_vector =
                ir_builder->Create<ir::BitCast>(ir_accessor_a, ir_accessor_a->type->Tile(1));
            auto ir_vector_broadcast_a = ir_builder->Create<ir::VectorBroadcast>(
                ir_accessor_a_vector, ir_vec_bit_cast_b->type->value_type, 0);
            for (int64_t c = 0; c < ir_simd_type_b->shape[0]; ++c) {
                auto ir_accessor_b = ir_builder->CreateAccessor(ir_vec_bit_cast_b);
                ir_accessor_b->transform_matrix.resize(0, 0);
                ir_accessor_b->shift_vector[0] = c;
                auto ir_mul = ir_builder->Mul(ir_vector_broadcast_a, ir_accessor_b);
                auto ir_accessor_c_row = ir_builder->CreateAccessor(ir_mat_bit_cast_c);
                ir_accessor_c_row->transform_matrix.resize(0, 0);
                ir_accessor_c_row->shift_vector[0] = i;
                auto ir_accessor_c = ir_builder->CreateAccessor(ir_accessor_c_row);
                ir_accessor_c->transform_matrix.resize(0, 0);
                ir_accessor_c->shift_vector[0] = c;
                auto ir_sum = ir_builder->Add(ir_mul, ir_accessor_c);
                auto ir_write = ir_builder->Create<ir::Write>(ir_sum, ir_accessor_c);
            }
        }
    }

   private:
    int64_t bits = 128;
    int64_t bytes = 16;
    int64_t rows;
    int64_t cols;
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

    void ExpressInline(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                       std::shared_ptr<ir::Tensor> ir_mat_c,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        for (auto ir_kernel : ir_builder->matrix_multiply_kernel_queue) {
            if (ir_kernel->Match(ir_mat_a->type, ir_mat_b->type)) {
                ir_kernel->Express(ir_mat_a, ir_mat_b, ir_mat_c, ir_builder);
                return;
            }
        }

        if (ir_mat_a->type->IsScalar()) {
            auto ir_re = ir_builder->Add(ir_builder->Mul(ir_mat_a, ir_mat_b), ir_mat_c);
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
        ir_grid->unroll_grid = ir_mat_a->type->unroll_grid;

        auto ir_accessor_a = ir_builder->CreateAccessor(ir_mat_a);
        ir_accessor_a->transform_matrix(0, 0) = 1;
        ir_accessor_a->transform_matrix(1, 1) = 1;
        auto ir_accessor_b = ir_builder->CreateAccessor(ir_mat_b);
        ir_accessor_b->transform_matrix(0, 1) = 1;
        ir_accessor_b->transform_matrix(1, 2) = 1;
        auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_c);
        ir_accessor_c->transform_matrix(0, 0) = 1;
        ir_accessor_c->transform_matrix(1, 2) = 1;

        this->ExpressInline(ir_accessor_a, ir_accessor_b, ir_accessor_c, ir_builder);
    }
};

}  // namespace galois::op
