#pragma once

#include "cpuinfo.h"
#include "galois/graph/graph.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/ir/matrix.hpp"
#include "galois/transform/transform.hpp"

/*/

namespace galois::lowering {

using namespace ir;

inline Eigen::VectorXi64 GetNormalizeShape(std::shared_ptr<ir::TensorType> ir_tensor_type) {
    GALOIS_ASSERT(ir_tensor_type);
    auto ir_value_type = ir_tensor_type->value_type;
    if (Is<ir::TensorType>(ir_value_type)) {
        return ir_tensor_type->shape.array() *
               GetNormalizeShape(Cast<ir::TensorType>(ir_value_type)).array();
    } else {
        return ir_tensor_type->shape;
    }
}

// class ScalarProductKernel : public Kernel {
//    public:
//     bool Match(std::vector<std::shared_ptr<Value>> ir_inputs) {
//         auto ir_scalar_a = ir_inputs[0];
//         auto ir_scalar_b = ir_inputs[1];
//         if ((ir_scalar_a->type == FloatType::Create(32) &&
//              ir_scalar_b->type == FloatType::Create(32))) {
//             return true;
//         }
//         return false;
//     }

//     void Build(std::shared_ptr<Value> ir_mat_a, std::shared_ptr<Value> ir_mat_b,
//                std::shared_ptr<Value> ir_mat_c, std::shared_ptr<Builder> ir_builder) {
//         auto ir_mul = ir_builder->Create<Mul>(ir_mat_a, ir_mat_b);
//         auto ir_sum = ir_builder->Create<Add>(ir_mul, ir_mat_c);
//         auto ir_write = ir_builder->Create<Write>(ir_sum, Cast<Accessor>(ir_mat_c));
//     }
// };

class PackKernel : public Kernel {
   public:
    bool Match(std::vector<std::shared_ptr<Value>> ir_inputs,
               std::vector<std::shared_ptr<Value>> ir_outputs,
               std::shared_ptr<Builder> ir_builder) override {
        auto ir_output = ir_outputs[0];
        if (ir_output->type->value_type == FloatType::Create(32)) {
            return true;
        }
        return false;
    }

    void Build(std::vector<std::shared_ptr<Value>> ir_inputs,
               std::vector<std::shared_ptr<Value>> ir_outputs,
               std::shared_ptr<Builder> ir_builder) override {
        auto ir_input = ir_inputs[0];
        auto ir_output = ir_outputs[0];

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor_input = ir_builder->CreateIdentityAccessor(ir_input);
        auto ir_accessor_output = ir_builder->CreateIdentityAccessor(ir_output);
        ir_builder->Create<Write>(ir_accessor_input, ir_accessor_output);
    }
};

// class ProductKernel : public Kernel {
//    public:
//     bool Match(std::vector<std::shared_ptr<Value>> ir_inputs,
//                std::vector<std::shared_ptr<Value>> ir_outputs, std::shared_ptr<Builder>) override
//                {
//         auto ir_mat_a = ir_inputs[0];
//         auto ir_mat_b = ir_inputs[1];
//         if ((ir_mat_a->type == TensorType::CreateMatrixType(FloatType::Create(32), 4, 1)) &&
//             ir_mat_b->type == TensorType::CreateMatrixType(FloatType::Create(32), 1, 4)) {
//             return true;
//         }
//         return false;
//     }

//     void Build(std::vector<std::shared_ptr<Value>> ir_inputs,
//                std::vector<std::shared_ptr<Value>> ir_outputs,
//                std::shared_ptr<Builder> ir_builder) override {
//         auto ir_mat_a = ir_inputs[0];
//         auto ir_mat_b = ir_inputs[1];
//         auto ir_mat_c = ir_outputs[0];
//         Eigen::VectorXi64 v4(1);
//         v4[0] = 4;
//         auto ir_f32x4_type = VectorType::CreateImp(FloatType::CreateImp(32), 4);
//         auto ir_bit_cast_a = ir_builder->Create<BitCast>(ir_mat_a, ir_f32x4_type);
//         auto ir_bit_cast_b = ir_builder->Create<BitCast>(ir_mat_b, ir_f32x4_type);
//         auto ir_bit_cast_c =
//             ir_builder->Create<BitCast>(ir_mat_c, TensorType::Create(ir_f32x4_type, v4));

//         for (int64_t i = 0; i < 4; ++i) {
//             auto ir_vector_broadcast_a = ir_builder->Create<VectorBroadcast>(ir_bit_cast_a, i);
//             auto ir_mul = ir_builder->Create<Mul>(ir_vector_broadcast_a, ir_bit_cast_b);
//             auto ir_accessor_c = ir_builder->CreateAccessor(ir_bit_cast_c);
//             ir_accessor_c->shift_vector[0] = i;
//             auto ir_sum = ir_builder->Create<Add>(ir_mul, ir_accessor_c);
//             auto ir_write =
//                 ir_builder->Create<Write>(ir_sum, Cast<Accessor>(ir_accessor_c->Clone()));
//         }
//     }
// };

// class ProductKernel2 : public Kernel {
//    public:
//     bool Match(std::vector<std::shared_ptr<Value>> ir_inputs,
//                std::vector<std::shared_ptr<Value>> ir_outputs, std::shared_ptr<Builder>) override
//                {
//         auto ir_mat_a = ir_inputs[0];
//         auto ir_mat_b = ir_inputs[1];
//         if ((ir_mat_a->type == TensorType::CreateMatrixType(FloatType::Create(32), 8, 1)) &&
//             ir_mat_b->type == TensorType::CreateMatrixType(FloatType::Create(32), 1, 8)) {
//             return true;
//         }
//         return false;
//     }

//     void Build(std::vector<std::shared_ptr<Value>> ir_inputs,
//                std::vector<std::shared_ptr<Value>> ir_outputs,
//                std::shared_ptr<Builder> ir_builder) override {
//         auto ir_mat_a = ir_inputs[0];
//         auto ir_mat_b = ir_inputs[1];
//         auto ir_mat_c = ir_outputs[0];
//         auto ir_ts_type_a = TensorType::CreateMatrixType(
//             TensorType::CreateMatrixType(FloatType::Create(32), 4, 1), 2, 1);
//         auto ir_ts_type_b = TensorType::CreateMatrixType(
//             TensorType::CreateMatrixType(FloatType::Create(32), 1, 4), 1, 2);
//         auto ir_ts_type_c = TensorType::CreateMatrixType(
//             TensorType::CreateMatrixType(FloatType::Create(32), 4, 4), 2, 2);

//         auto ir_bit_cast_a = ir_builder->Create<BitCast>(ir_mat_a, ir_ts_type_a);
//         auto ir_bit_cast_b = ir_builder->Create<BitCast>(ir_mat_b, ir_ts_type_b);
//         auto ir_bit_cast_c = ir_builder->Create<BitCast>(ir_mat_c, ir_ts_type_c);

//         ProductKernel product_kernel;

//         for (int64_t i = 0; i < 2; ++i) {
//             for (int64_t j = 0; j < 2; ++j) {
//                 auto ir_accessor_a = ir_builder->CreateAccessor(ir_bit_cast_a);
//                 ir_accessor_a->shift_vector[0] = i;
//                 auto ir_accessor_b = ir_builder->CreateAccessor(ir_bit_cast_b);
//                 ir_accessor_b->shift_vector[1] = j;
//                 auto ir_accessor_c = ir_builder->CreateAccessor(ir_bit_cast_c);
//                 ir_accessor_c->shift_vector[0] = i;
//                 ir_accessor_c->shift_vector[1] = j;
//                 product_kernel.Build({ir_accessor_a, ir_accessor_b}, {ir_accessor_c},
//                 ir_builder);
//             }
//         }
//     }
// };

inline void LowerProductImp(std::shared_ptr<Value> ir_mat_a, std::shared_ptr<Value> ir_mat_b,
                            std::shared_ptr<Value> ir_mat_c, std::shared_ptr<Builder> ir_builder) {
    for (auto ir_kernel : ir_builder->kernel_queue) {
        if (ir_kernel->Match({ir_mat_a, ir_mat_b}, {ir_mat_c}, ir_builder)) {
            ir_kernel->Build({ir_mat_a, ir_mat_b}, {ir_mat_c}, ir_builder);
            return;
        }
    }

    auto ir_mat_type_a = ir_mat_a->type;
    auto ir_mat_type_b = ir_mat_b->type;
    auto ir_mat_type_c = ir_mat_c->type;

    GALOIS_ASSERT(ir_mat_type_a->shape[1] == ir_mat_type_b->shape[0]);
    GALOIS_ASSERT(ir_mat_type_c->shape[0] == ir_mat_type_a->shape[0]);
    GALOIS_ASSERT(ir_mat_type_c->shape[1] == ir_mat_type_b->shape[1]);

    auto [ir_grid, scope_guard] = ir_builder->CreateGrid(Eigen::Vector3i64(
        ir_mat_type_a->shape[0], ir_mat_type_a->shape[1], ir_mat_type_b->shape[1]));

    auto ir_accessor_a = ir_builder->CreateAccessor(ir_mat_a);
    ir_accessor_a->transform_matrix(0, 0) = 1;
    ir_accessor_a->transform_matrix(1, 1) = 1;
    auto ir_accessor_b = ir_builder->CreateAccessor(ir_mat_b);
    ir_accessor_b->transform_matrix(0, 1) = 1;
    ir_accessor_b->transform_matrix(1, 2) = 1;
    auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_c);
    ir_accessor_c->transform_matrix(0, 0) = 1;
    ir_accessor_c->transform_matrix(1, 2) = 1;

    LowerProductImp(ir_accessor_a, ir_accessor_b, ir_accessor_c, ir_builder);
}

inline void Pack(std::vector<std::shared_ptr<Value>> ir_inputs,
                 std::vector<std::shared_ptr<Value>> ir_outputs,
                 std::shared_ptr<Builder> ir_builder) {
    auto ir_input = ir_inputs.front();
    auto ir_output = ir_outputs.front();
    auto ir_output_normalize_shape = GetNormalizeShape(ir_output->type);
    GALOIS_ASSERT(ir_input->type->shape == ir_output_normalize_shape);
    auto ir_pack_kernel = std::make_shared<PackKernel>();
    if (ir_pack_kernel->Match(ir_inputs, ir_outputs, ir_builder)) {
        ir_pack_kernel->Build(ir_inputs, ir_outputs, ir_builder);
        return;
    }
    auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_output->type->shape);
    auto ir_output_block = ir_builder->CreateIdentityAccessor(ir_output);
    auto ir_input_block_origin = ir_builder->CreateIdentityAccessor(ir_input);
    ir_input_block_origin->transform_matrix.diagonal().array() *=
        ir_output_block->type->shape.array();
    auto ir_input_block =
        ir_builder->Create<Slice>(ir_input_block_origin, ir_output_block->type->shape);
    Pack({ir_input_block}, {ir_output_block}, ir_builder);
}

inline void SetZero(std::vector<std::shared_ptr<Value>> ir_inputs,
                    std::vector<std::shared_ptr<Value>>, std::shared_ptr<Builder> ir_builder) {
    auto ir_input = ir_inputs.front();
    auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
    auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
    auto ir_zero = ir_builder->Create<ConstantFloat>(FloatType::Create(32), 0.0);
    ir_builder->Create<Write>(ir_zero, ir_accessor);
}


inline void CombineViewer(std::shared_ptr<ir::Grid> ir_grid) {
    transform::Each<ir::Accessor>(ir_grid, [&](std::shared_ptr<ir::Accessor> ir_accessor) {
        if (auto ir_tensor_viewer = Cast<ir::Viewer>(ir_accessor->Tensor())) {
            ir_accessor->transform_matrix =
                ir_tensor_viewer->transform_matrix * ir_accessor->transform_matrix;
            ir_accessor->shift_vector = ir_tensor_viewer->shift_vector + ir_accessor->shift_vector;
            ir_accessor->Tensor() = ir_tensor_viewer->tensor;
        }
    });
}

class GemmBuilder {
   private:
    static int64_t ConvergeMImp(int64_t cache_size, int64_t k, int64_t m) {
        auto next_m = std::floor(std::sqrt(static_cast<double>(cache_size - 2 * m * k)));
        return next_m;
    }

    static int64_t GetM(int64_t cache_size, int64_t k) {
        int64_t pre_m = 0;
        int64_t m = 0;
        for (int64_t i = 0; i < 100; ++i) {
            m = ConvergeMImp(cache_size, k, pre_m);
            if (m == pre_m) {
                break;
            } else {
                pre_m = m;
            }
        }

        return m;
    }

   public:
    static std::shared_ptr<GemmBuilder> Create() {
        std::shared_ptr<GemmBuilder> self(new GemmBuilder);
        cpuinfo_initialize();

        // 大部分cpu的registers_count应该都是32
        auto ir_value_type = ir::FloatType::Create(32);
        int64_t registers_count = 32;
        std::shared_ptr<TensorType> ir_matrix_type_a =
            TensorType::CreateMatrixType(ir_value_type, 4, 1);
        std::shared_ptr<TensorType> ir_matrix_type_b =
            TensorType::CreateMatrixType(ir_value_type, 1, 4);
        ir_matrix_type_a = TensorType::CreateMatrixType(ir_matrix_type_a, 3, 1);
        ir_matrix_type_b = TensorType::CreateMatrixType(ir_matrix_type_b, 1, 2);

        // l1 cache
        auto l1_cache = cpuinfo_get_l1d_caches();
        auto normalize_shape_a = GetNormalizeShape(Cast<TensorType>(ir_matrix_type_a));
        auto normalize_shape_b = GetNormalizeShape(Cast<TensorType>(ir_matrix_type_b));
        auto normalize_k =
            (l1_cache->size / ir_value_type->bytes) / (normalize_shape_a[0] + normalize_shape_b[1]);
        auto k = normalize_k / normalize_shape_a[1];
        ir_matrix_type_a = TensorType::CreateMatrixType(ir_matrix_type_a, 1, k);
        ir_matrix_type_b = TensorType::CreateMatrixType(ir_matrix_type_b, k, 1);

        auto l2_cache = cpuinfo_get_l2_caches();
        normalize_shape_a = GetNormalizeShape(Cast<TensorType>(ir_matrix_type_a));
        normalize_shape_b = GetNormalizeShape(Cast<TensorType>(ir_matrix_type_b));
        normalize_k = normalize_shape_a[1];
        auto normalize_m_tmp =
            (-2 * normalize_k + std::sqrt(4 * normalize_k * normalize_k +
                                          4 * (l2_cache->size / ir_value_type->bytes))) /
            2;
        auto normalize_m = static_cast<int64_t>(std::floor(normalize_m_tmp));
        // auto normalize_m = GetM(l2_cache->size, normalize_k);
        auto m = normalize_m / normalize_shape_a[0];
        auto n = normalize_m / normalize_shape_b[1];
        ir_matrix_type_a = TensorType::CreateMatrixType(ir_matrix_type_a, 1, k);
        ir_matrix_type_b = TensorType::CreateMatrixType(ir_matrix_type_b, k, 1);

        auto l3_cache = cpuinfo_get_l3_caches();
        if (l3_cache) {
            GALOIS_TODO;
        }

        return self;
    }
};

inline void LoweringAffine(std::shared_ptr<graph::ComputeGraph> ir_module) {}

}  // namespace galois::lowering

*/
