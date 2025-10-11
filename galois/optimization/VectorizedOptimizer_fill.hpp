#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "cpuinfo.h"
#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/ir/view.hpp"
#include "galois/optimization/VectorizedOptimizer.hpp"
#include "galois/op/fill.hpp"
#include "galois/transform/each.hpp"

namespace galois::optimization {

// VectorizedOptimizerFill：在矢量化运算前对输出张量执行 Fill，保证初始值为 0。
// 设计目的：保留 VectorizedOptimizer 的 SIMD 优化能力，同时补全“先清零再计算”的语义，
// 以兼容需要依赖初始值的算子场景。例如某些操作期望输出张量在写入前已经被初始化，
// 这里统一用 FillCreator 写入 0，避免残留随机值影响结果。
class VectorizedOptimizerFill {
   protected:
    VectorizedOptimizerFill() = default;

   private:
    struct VectorizationConfig {
        ir::ArithmeticInstruction::Operation operation;
        std::shared_ptr<ir::TensorType> output_type;
        std::shared_ptr<ir::TensorType> vector_chunk_type;
        int64_t lanes = 0;
    };

   public:
    static std::shared_ptr<VectorizedOptimizerFill> Create() {
        auto self = std::shared_ptr<VectorizedOptimizerFill>(new VectorizedOptimizerFill);
        self->cpu_info = VectorizedCpuInfo::Create();
        return self;
    }

    std::shared_ptr<ir::Operator> Optimize(std::shared_ptr<ir::Operator> ir_operator) {
        if (!ir_operator) return ir_operator;

        // 构造矢量化配置（lane 数、向量块类型等），若失败则退化为原算子。
        auto config = this->BuildVectorizationConfig(ir_operator);
        if (!config) return ir_operator;

        auto ir_builder = ir::Builder::Create();
        ir_builder->id = reinterpret_cast<uintptr_t>(ir_builder.get());
        auto [ir_vectorized_operator, scope] = ir_builder->CreateOperator(
            ir_operator->GetOperatorType(), ir_operator->name + "_vec_fill");

        auto ir_lhs = ir_vectorized_operator->inputs[0];
        auto ir_rhs = ir_vectorized_operator->inputs[1];

        // 发射填零 + SIMD 主体，若失败同样退化。
        auto ir_output = this->EmitVectorizedComputation(ir_builder, ir_lhs, ir_rhs, *config);
        if (!ir_output) return ir_operator;

        ir_builder->Return(ir_output);
        scope.reset();
        return ir_vectorized_operator;
    }

   private:
    std::optional<VectorizationConfig> BuildVectorizationConfig(
        std::shared_ptr<ir::Operator> ir_operator) {
        if (!ir_operator || ir_operator->inputs.size() != 2) return std::nullopt;

        auto maybe_operation = this->DetectArithmeticOperation(ir_operator);
        if (!maybe_operation) return std::nullopt;

        auto ir_operator_type = ir_operator->GetOperatorType();
        if (!ir_operator_type) return std::nullopt;

        auto ir_output_type = ir_operator_type->output_type;
        if (!ir_output_type) return std::nullopt;

        if (ir_output_type->DenseType() != ir_output_type) return std::nullopt;

        auto ir_element_type = ir_output_type->DataType();
        auto ir_real_number_type = Cast<ir::RealNumberType>(ir_element_type);
        if (!ir_real_number_type) return std::nullopt;

        if (!this->cpu_info) return std::nullopt;

        auto simd_bits = this->cpu_info->SimdBits();
        if (simd_bits <= 0) return std::nullopt;

        auto element_bits = ir_real_number_type->bits;
        if (element_bits <= 0) return std::nullopt;

        if (simd_bits % element_bits != 0) return std::nullopt;

        auto lanes = simd_bits / element_bits;
        if (lanes <= 1) return std::nullopt;
        if (!IsPowerOfTwo(lanes)) return std::nullopt;
        if (ir_output_type->NormalizeSize() <= 0) return std::nullopt;

        auto normalized_shape = ir_output_type->NormalizeShape();
        if (!normalized_shape.size()) return std::nullopt;

        VectorizationConfig config;
        config.operation = *maybe_operation;
        config.output_type = ir_output_type;

        auto vector_type = ir_element_type->Tile(lanes);
        auto last_dim_index = static_cast<int>(normalized_shape.size()) - 1;
        int64_t last_dim = normalized_shape[last_dim_index];
        if (last_dim % lanes != 0) return std::nullopt;

        Eigen::VectorXi64 chunk_shape = normalized_shape;
        chunk_shape[last_dim_index] = last_dim / lanes;

        config.vector_chunk_type = vector_type->Tile(chunk_shape);
        config.lanes = lanes;
        return config;
    }

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
                if (*maybe_operation != instruction->operation) mismatch = true;
            });
        if (!maybe_operation || mismatch) return std::nullopt;
        return maybe_operation;
    }

    std::shared_ptr<ir::Tensor> EmitVectorizedComputation(
        std::shared_ptr<ir::Builder> ir_builder, std::shared_ptr<ir::Tensor> ir_lhs,
        std::shared_ptr<ir::Tensor> ir_rhs, const VectorizationConfig& config) {
        auto ir_output = ir_builder->Alloca(config.output_type);

        // 先显式 Fill，保证输出张量处于已知状态（与 BinaryCreator 行为保持一致）。
        auto ir_zero = ir_builder->GetZero(config.output_type->DataType());
        ir_builder->ExpressCreator<op::FillCreator>({ir_output, ir_zero});

        // 将张量视图转换为 `<lanes x T>` × chunk_shape`，为后续 SIMD 运算做准备。
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
        if (!ir_vec_result) return nullptr;

        ir_builder->Write(ir_vec_result, ir_accessor_out);
        return ir_output;
    }

    std::shared_ptr<ir::Tensor> ApplyArithmeticOperation(
        std::shared_ptr<ir::Builder> ir_builder, std::shared_ptr<ir::Tensor> lhs,
        std::shared_ptr<ir::Tensor> rhs, ir::ArithmeticInstruction::Operation operation) {
        // 直接复用 Builder 的底层算术指令，确保真正生成 SIMD 二元操作。
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

    // CPU SIMD 能力缓存，与基础 VectorizedOptimizer 共用探测逻辑。
    std::shared_ptr<VectorizedCpuInfo> cpu_info = nullptr;
};

}  // namespace galois::optimization
