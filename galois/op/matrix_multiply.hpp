#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"

namespace galois::op {

using namespace ir;

class ProductKernel : public Kernel {
   public:
    bool Match(std::vector<std::shared_ptr<Tensor>> ir_inputs,
               std::vector<std::shared_ptr<Tensor>> ir_outputs, std::shared_ptr<Builder>) override {
        auto ir_mat_a = ir_inputs[0];
        auto ir_mat_b = ir_inputs[1];
        if ((ir_mat_a->type == TensorType::CreateMatrixType(FloatType::Create(32), 4, 1)) &&
            ir_mat_b->type == TensorType::CreateMatrixType(FloatType::Create(32), 1, 4)) {
            return true;
        }
        return false;
    }

    void Build(std::vector<std::shared_ptr<Tensor>> ir_inputs,
               std::vector<std::shared_ptr<Tensor>> ir_outputs,
               std::shared_ptr<Builder> ir_builder) override {
        auto ir_mat_a = ir_inputs[0];
        auto ir_mat_b = ir_inputs[1];
        auto ir_mat_c = ir_outputs[0];
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

class MatrixMultiplyCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        if (ir_input_types[0]->IsScalar() && ir_input_types[1]->IsScalar()) {
            GALOIS_ASSERT(ir_input_types[0] == ir_input_types[1]);
            return ir_input_types[0];
        }

        auto ir_value_type =
            this->InferType({ir_input_types[0]->value_type, ir_input_types[1]->value_type});
        return TensorType::CreateMatrixType(ir_value_type, ir_input_types[0]->shape[0],
                                            ir_input_types[1]->shape[1]);
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        for (auto ir_kernel : ir_builder->kernel_queue) {
            if (ir_kernel->Match(ir_inputs, ir_outputs, ir_builder)) {
                ir_kernel->Build(ir_inputs, ir_outputs, ir_builder);
                return;
            }
        }

        auto ir_mat_a = ir_inputs[0];
        auto ir_mat_b = ir_inputs[1];
        auto ir_mat_c = ir_outputs[0];
        GALOIS_ASSERT(ir_mat_a->type->shape[1] == ir_mat_b->type->shape[0]);
        GALOIS_ASSERT(ir_mat_c->type->shape[0] == ir_mat_a->type->shape[0]);
        GALOIS_ASSERT(ir_mat_c->type->shape[1] == ir_mat_b->type->shape[1]);

        if (ir_mat_a->type->IsScalar()) {
            auto ir_re =
                ir_builder->Create<Add>(ir_builder->Create<Mul>(ir_mat_a, ir_mat_b), ir_mat_c);
            ir_builder->Create<Write>(ir_re, ir_mat_c);
        }

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(Eigen::Vector3i64(
            ir_mat_a->type->shape[0], ir_mat_a->type->shape[1], ir_mat_b->type->shape[1]));

        auto ir_accessor_a = ir_builder->CreateAccessor(ir_mat_a);
        ir_accessor_a->transform_matrix(0, 0) = 1;
        ir_accessor_a->transform_matrix(1, 1) = 1;
        auto ir_accessor_b = ir_builder->CreateAccessor(ir_mat_b);
        ir_accessor_b->transform_matrix(0, 1) = 1;
        ir_accessor_b->transform_matrix(1, 2) = 1;
        auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_c);
        ir_accessor_c->transform_matrix(0, 0) = 1;
        ir_accessor_c->transform_matrix(1, 2) = 1;

        this->AffineExpress({ir_accessor_a, ir_accessor_b}, {ir_accessor_c}, ir_builder);
    }
};

}  // namespace galois::op
