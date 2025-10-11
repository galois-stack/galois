#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "cpuinfo.h"
#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/arithmetic.hpp"
#include "galois/ir/view.hpp"
#include "galois/transform/each.hpp"

namespace galois::optimization {

// 简单的 CPU SIMD 信息结构，只记录 SIMD 位宽，方便后续判断能否矢量化。
class VectorizedCpuInfo {
   public:
    static std::shared_ptr<VectorizedCpuInfo> Create() {
        auto self = std::shared_ptr<VectorizedCpuInfo>(new VectorizedCpuInfo);
        cpuinfo_initialize();
        self->DetectCpuFeatures();
        return self;
    }

    int64_t SimdBits() const { return simd_bits; }

   private:
    VectorizedCpuInfo() = default;

    void DetectCpuFeatures() {
        simd_bits = 0;
        if (cpuinfo_has_x86_avx512f()) {
            simd_bits = 512;
            return;
        }
        if (cpuinfo_has_x86_avx2() || cpuinfo_has_x86_avx()) {
            simd_bits = 256;
            return;
        }
        if (cpuinfo_has_x86_sse() || cpuinfo_has_x86_sse2() || cpuinfo_has_arm_neon()) {
            simd_bits = 128;
            return;
        }
    }

    int64_t simd_bits = 0;
};

// VectorizedOptimizer：针对二元算子（加减乘除）做尾维整除时的 SIMD 化。
class VectorizedOptimizer {
   protected:
    VectorizedOptimizer() = default;

   private:
    struct VectorizationConfig {
        ir::ArithmeticInstruction::Operation operation;   // 算子类型：Add/Sub/Mul/Div
        std::shared_ptr<ir::TensorType> output_type;       // 原输出的标量类型
        std::shared_ptr<ir::TensorType> vector_chunk_type; // `<lanes x T>` × chunk_shape
        int64_t lanes = 0;                                 // SIMD 宽度（元素个数）
    };


   public:
    static std::shared_ptr<VectorizedOptimizer> Create() {
        std::shared_ptr<VectorizedOptimizer> self(new VectorizedOptimizer);
        self->cpu_info = VectorizedCpuInfo::Create();
        return self;
    }

    // 若满足矢量化条件，则构造新的 `_vec` 算子并返回；否则直接返回原算子。
    std::shared_ptr<ir::Operator> Optimize(std::shared_ptr<ir::Operator> ir_operator) {
        if (!ir_operator) return ir_operator;

        auto config = this->BuildVectorizationConfig(ir_operator);
        if (!config) {
            return ir_operator;
        }

        auto ir_builder = ir::Builder::Create();
        ir_builder->id = reinterpret_cast<uintptr_t>(ir_builder.get());
        auto [ir_vectorized_operator, scope] = ir_builder->CreateOperator(
            ir_operator->GetOperatorType(), ir_operator->name + "_vec");

        auto ir_lhs = ir_vectorized_operator->inputs[0];
        auto ir_rhs = ir_vectorized_operator->inputs[1];

        auto ir_output = this->EmitVectorizedComputation(ir_builder, ir_lhs, ir_rhs, *config);
        if (!ir_output) {
            return ir_operator;
        }

        ir_builder->Return(ir_output);
        scope.reset();
        return ir_vectorized_operator;
    }

   private:
    // 检测算子可否矢量化，并构造向量块类型；若不满足条件返回 nullopt。
    std::optional<VectorizationConfig> BuildVectorizationConfig(
        std::shared_ptr<ir::Operator> ir_operator) {
        if (!ir_operator || ir_operator->inputs.size() != 2) {
            return std::nullopt;
        }

        auto maybe_operation = this->DetectArithmeticOperation(ir_operator);
        if (!maybe_operation) {
            return std::nullopt;
        }

        auto ir_operator_type = ir_operator->GetOperatorType();
        if (!ir_operator_type) {
            return std::nullopt;
        }

        auto ir_output_type = ir_operator_type->output_type;
        if (!ir_output_type) {
            return std::nullopt;
        }

        if (ir_output_type->DenseType() != ir_output_type) {
            return std::nullopt;
        }

        auto ir_element_type = ir_output_type->DataType();
        auto ir_real_number_type = Cast<ir::RealNumberType>(ir_element_type);
        if (!ir_real_number_type) {
            return std::nullopt;
        }

        if (!this->cpu_info) {
            return std::nullopt;
        }

        auto simd_bits = this->cpu_info->SimdBits();
        if (simd_bits <= 0) {
            return std::nullopt;
        }

        auto element_bits = ir_real_number_type->bits;
        if (element_bits <= 0) {
            return std::nullopt;
        }

        if (simd_bits % element_bits != 0) {
            return std::nullopt;
        }

        auto lanes = simd_bits / element_bits;
        if (lanes <= 1) {
            return std::nullopt;
        }

        if (!IsPowerOfTwo(lanes)) {
            return std::nullopt;
        }

        if (ir_output_type->NormalizeSize() <= 0) {
            return std::nullopt;
        }

        auto normalized_shape = ir_output_type->NormalizeShape();
        if (!normalized_shape.size()) {
            return std::nullopt;
        }

        VectorizationConfig config;
        config.operation = *maybe_operation;
        config.output_type = ir_output_type;
        // 仅在最后一维可以被 SIMD 宽度整除时启用矢量化。
        auto vector_type = ir_element_type->Tile(lanes);
        auto last_dim_index = static_cast<int>(normalized_shape.size()) - 1;
        int64_t last_dim = normalized_shape[last_dim_index];
        if (last_dim % lanes != 0) {
            return std::nullopt;
        }

        Eigen::VectorXi64 chunk_shape = normalized_shape;
        chunk_shape[last_dim_index] = last_dim / lanes;

        // `<lanes x T>` × chunk_shape 的 tensor，即整体按向量块分块后的类型。
        config.vector_chunk_type = vector_type->Tile(chunk_shape);
        config.lanes = lanes;
        return config;
    }

    // 检查算子内部的算术指令是否一致（只支持纯加/减/乘/除）。
    std::optional<ir::ArithmeticInstruction::Operation> DetectArithmeticOperation(
        std::shared_ptr<ir::Operator> ir_operator) {
        std::optional<ir::ArithmeticInstruction::Operation> maybe_operation;
        bool mismatch = false;
        transform::Each<ir::ArithmeticInstruction>(
            ir_operator, [&](std::shared_ptr<ir::ArithmeticInstruction> instruction) {
                if (!maybe_operation) {
                    maybe_operation = instruction->operation;
                    return;
                }
                if (*maybe_operation != instruction->operation) {
                    mismatch = true;
                }
            });
        if (!maybe_operation || mismatch) {
            return std::nullopt;
        }
        return maybe_operation;
    }

    // 将输入/输出一次性 BitCast 成向量块，并复用原算子完成运算。
    std::shared_ptr<ir::Tensor> EmitVectorizedComputation(
        std::shared_ptr<ir::Builder> ir_builder, std::shared_ptr<ir::Tensor> ir_lhs,
        std::shared_ptr<ir::Tensor> ir_rhs, const VectorizationConfig& config) {

        // 先 Alloca 一个输出张量并构造向量块视图，后续在自管的 Grid 中写回。
        auto ir_output = ir_builder->Alloca(config.output_type);

        // 将原始张量视为 `<lanes x T>` 的块数组，方便后端发 SIMD 指令。
        auto ir_lhs_vec = ir_builder->BitCastView(ir_lhs, config.vector_chunk_type);
        auto ir_rhs_vec = ir_builder->BitCastView(ir_rhs, config.vector_chunk_type);
        auto ir_out_vec = ir_builder->BitCastView(ir_output, config.vector_chunk_type);

        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(config.vector_chunk_type->shape);

        auto ir_accessor_lhs = ir_builder->CreateIdentityAccessor(ir_lhs_vec);
        auto ir_accessor_rhs = ir_builder->CreateIdentityAccessor(ir_rhs_vec);
        auto ir_accessor_out = ir_builder->CreateIdentityAccessor(ir_out_vec);

        auto ir_vec_result =
            this->ApplyArithmeticOperation(ir_builder, ir_accessor_lhs, ir_accessor_rhs,
                                           config.operation);
        if (!ir_vec_result) {
            return nullptr;
        }

        ir_builder->Write(ir_vec_result, ir_accessor_out);
        return ir_output;
    }

    std::shared_ptr<VectorizedCpuInfo> cpu_info = nullptr;  // 缓存 CPU SIMD 信息

    // 直接发射底层算术指令，实现加减乘除。
    std::shared_ptr<ir::Tensor> ApplyArithmeticOperation(
        std::shared_ptr<ir::Builder> ir_builder, std::shared_ptr<ir::Tensor> lhs,
        std::shared_ptr<ir::Tensor> rhs, ir::ArithmeticInstruction::Operation operation) {
        switch (operation) {
            case ir::ArithmeticInstruction::Operation::Add:
                return ir_builder->Add(lhs, rhs);
            case ir::ArithmeticInstruction::Operation::Sub:
                return ir_builder->Sub(lhs, rhs);
            case ir::ArithmeticInstruction::Operation::Mul:
                return ir_builder->Mul(lhs, rhs);
            case ir::ArithmeticInstruction::Operation::Div:
                return ir_builder->Div(lhs, rhs);
            default:
                return nullptr;
        }
    }

};

}  // namespace galois::optimization
