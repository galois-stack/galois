#pragma once

#include "boost/scope/scope_exit.hpp"
#include "c++/z3++.h"
#include "cpuinfo.h"
#include "fmt/format.h"
#include "galois/ir/clone_visitor.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"
#include "galois/transform/transform.hpp"

namespace galois::optimization {

class NativeCpuInfo {
   public:
    enum Simd { None = 0, SSE, SSE2, AVX, AVX2, AVX512, AMX, NEON, SVE, SME };

   public:
    static std::shared_ptr<NativeCpuInfo> Create() {
        std::shared_ptr<NativeCpuInfo> self(new NativeCpuInfo);
        // 初始化 cpuinfo 库
        cpuinfo_initialize();
        // 检测指令集并推断 SIMD 寄存器数量和位宽
        self->DetectCpuFeatures();
        self->cache_sizes.resize(2);
        return self;
    }

    int64_t SimdBits() { return this->simd_bits; }
    int64_t SimdRegisterCount() { return this->simd_register_count; }
    int64_t CacheLevel() { return cache_sizes.size(); }
    int64_t GetCacheSize(int64_t level) { return cache_sizes[level]; }

    Simd simd;

   private:
    NativeCpuInfo() = default;

    void DetectCpuFeatures() {
        // 默认值
        simd_bits = 128;
        simd_register_count = 32;
        if (cpuinfo_has_x86_avx512f()) {
            this->simd = Simd::AVX512;
            simd_bits = 512;
            simd_register_count = 32;
            return;
        }
        if (cpuinfo_has_x86_avx2()) {
            this->simd = Simd::AVX2;
            simd_bits = 256;
            simd_register_count = 16;
            return;
        }
        if (cpuinfo_has_x86_avx()) {
            this->simd = Simd::AVX;
            simd_bits = 256;
            simd_register_count = 16;
            return;
        }
        if (cpuinfo_has_x86_sse()) {
            this->simd = Simd::SSE;
            simd_bits = 128;
            simd_register_count = 8;
            return;
        }
        if (cpuinfo_has_x86_sse2()) {
            this->simd = Simd::SSE2;
            simd_bits = 128;
            simd_register_count = 8;
            return;
        }
        if (cpuinfo_has_arm_neon()) {
            this->simd = Simd::NEON;
            simd_bits = 128;
            simd_register_count = 32;
            return;
        }
        if (cpuinfo_has_arm_sve()) {
            this->simd = Simd::SVE;
            GALOIS_UNIMPLEMENT;
            return;
        }
        if (cpuinfo_has_arm_sme()) {
            this->simd = Simd::SME;
            GALOIS_UNIMPLEMENT;
            return;
        }
    }

    int64_t simd_register_count = 0;
    int64_t simd_bits = 0;
    std::vector<int64_t> cache_sizes;
};

inline std::vector<Eigen::VectorXi64> GenerateIndexGrid(Eigen::VectorXi64 shape) {
    std::vector<Eigen::VectorXi64> index_grid;
    Eigen::VectorXi64 root;
    index_grid.push_back(root);
    for (int64_t i = 0; i < shape.size(); ++i) {
        auto index_grid_copy = index_grid;
        index_grid.clear();
        for (auto index : index_grid_copy) {
            for (int64_t j = 0; j < shape[i]; ++j) {
                auto index_local = index;
                index_local.conservativeResize(index.size() + 1, Eigen ::NoChange);
                index_local.bottomRows(1)[0] = j;
                index_grid.push_back(index_local);
            }
        }
    }
    return index_grid;
}

inline void UnrollGrid(std::shared_ptr<ir::Grid> ir_grid) {
    auto index_grid = GenerateIndexGrid(ir_grid->shape);
    auto parent_block_ptr = ir_grid->parent_block.lock();
    GALOIS_ASSERT(parent_block_ptr);
    auto ir_grid_iter = std::find(RANGE((*parent_block_ptr)), ir_grid);

    auto ir_external_tensor_set = transform::CaptureExternalTensors(ir_grid->block);
    std::unordered_map<std::shared_ptr<ir::Tensor>, std::shared_ptr<ir::Tensor>> tensor_dict;
    for (auto ir_tensor : ir_external_tensor_set) {
        tensor_dict[ir_tensor] = ir_tensor;
    }

    for (auto index : index_grid) {
        auto ir_clone_visitor = ir::CloneVisitor::Create();
        ir_clone_visitor->tensor_dict = tensor_dict;
        for (auto ir_value : Clone(*ir_grid->block)) {
            auto ir_value_clone = ir_clone_visitor->Clone(ir_value);
            GALOIS_ASSERT(ir_value_clone->tag == ir_value->tag);
            if (auto ir_accessor = Cast<ir::Accessor>(ir_value_clone)) {
                if (ir_accessor->transform_matrix.size()) {
                    ir_accessor->shift_vector += ir_accessor->transform_matrix * index;
                    ir_accessor->transform_matrix.resize(0, 0);
                }
            }
            auto parent = Lock(ir_grid->parent_block);
            parent->insert(ir_grid_iter, ir_value_clone);
        }
    }

    auto parent = Lock(ir_grid->parent_block);
    parent->remove(ir_grid);

    ir_grid->Finalize();
}

class NeonGemmTilePolicy {
   public:
    static std::shared_ptr<NeonGemmTilePolicy> Create() {
        std::shared_ptr<NeonGemmTilePolicy> self(new NeonGemmTilePolicy);
        return self;
    };

    std::tuple<int64_t, int64_t> GetKernelTileShape(std::shared_ptr<ir::TensorType> ir_data_type,
                                                    std::shared_ptr<NativeCpuInfo> cpu_info) {
        int32_t simd_register_count = cpu_info->SimdRegisterCount();
        int32_t simd_lanes = (cpu_info->SimdBits() / 8) / ir_data_type->bytes;
        /// 通过Z3来求解寄存器分块， 该问题不是一个线性规划问题， 所以采用Z3来处理
        z3::context z3_context;
        z3::params z3_params(z3_context);
        z3_params.set("priority", z3_context.str_symbol("register tile"));
        z3::optimize z3_optimize(z3_context);
        z3_optimize.set(z3_params);
        z3::expr z3_register_tile_rows = z3_context.int_const("z3_register_tile_rows");
        z3::expr z3_register_tile_cols = z3_context.int_const("z3_register_tile_cols");
        z3_optimize.add(z3_register_tile_rows > 0);
        z3_optimize.add(z3_register_tile_cols > 0);
        z3_optimize.add(z3_register_tile_rows <= z3_register_tile_cols);  // 我们不需要镜像的解
        // z3_register_tile_rows + z3_register_tile_cols : 行和列的寄存器都需要保留，
        // 这样才能复用数据 z3_register_tile_rows * z3_register_tile_cols * int32_t(simd_lanes)：
        // 用于存储外积的结果
        z3_optimize.add(z3_register_tile_rows + z3_register_tile_cols +
                            z3_register_tile_rows * z3_register_tile_cols * simd_lanes <
                        simd_register_count);
        // 最大化无依赖的计算指令数目, 同时也最大化了计算强度
        z3::optimize::handle z3_handle_x =
            z3_optimize.maximize(z3_register_tile_rows * z3_register_tile_cols);
        GALOIS_ASSERT(z3_optimize.check() == z3::sat);
        z3::model z3_model = z3_optimize.get_model();
        auto register_tile_rows = z3_model.eval(z3_register_tile_rows).get_numeral_int64();
        auto register_tile_cols = z3_model.eval(z3_register_tile_cols).get_numeral_int64();
        return {register_tile_rows * simd_lanes, register_tile_cols * simd_lanes};
    }

    std::tuple<std::shared_ptr<ir::TensorType>, std::shared_ptr<ir::TensorType>,
               std::shared_ptr<op::MatrixMultiplyMicroKernel>>
    Tile(std::shared_ptr<ir::TensorType> ir_data_type, std::shared_ptr<NativeCpuInfo> cpu_info) {
        auto [kernel_tile_rows, kernel_tile_cols] =
            this->GetKernelTileShape(ir_data_type, cpu_info);

        auto ir_tile_mat_type_a = ir_data_type->Tile(kernel_tile_rows, 1)->Tile(1, 32)->Tile(4, 1);
        auto ir_tile_mat_type_b = ir_data_type->Tile(1, kernel_tile_cols)->Tile(32, 1)->Tile(1, 4);
        return std::make_tuple(ir_tile_mat_type_a, ir_tile_mat_type_b,
                               op::NeonMatrixMultiplyKernel::Create(
                                   cpu_info->SimdBits(), kernel_tile_rows, kernel_tile_cols));
    }
};

class AvxGemmTilePolicy {
   public:
    static std::shared_ptr<AvxGemmTilePolicy> Create() {
        std::shared_ptr<AvxGemmTilePolicy> self(new AvxGemmTilePolicy);
        return self;
    };

    std::tuple<std::shared_ptr<ir::TensorType>, std::shared_ptr<ir::TensorType>,
               std::shared_ptr<op::MatrixMultiplyMicroKernel>>
    Tile(std::shared_ptr<ir::TensorType> ir_data_type, std::shared_ptr<NativeCpuInfo> cpu_info) {
        int32_t simd_register_count = cpu_info->SimdRegisterCount();
        int32_t simd_lanes = (cpu_info->SimdBits() / 8) / ir_data_type->bytes;

        z3::context z3_context;
        z3::params z3_params(z3_context);
        z3_params.set("priority", z3_context.str_symbol("kernel tile"));
        z3::optimize z3_optimize(z3_context);
        z3_optimize.set(z3_params);
        // a的simd lanes, 通过求解得来， 因为存在寄存器不够用的情况， 所以需要裁剪
        z3::expr z3_kernel_tile_rows = z3_context.bv_const("z3_kernel_tile_rows", 32);
        z3::expr z3_kernel_tile_cols = z3_context.bv_const("z3_kernel_tile_cols", 32);
        z3_optimize.add(z3_kernel_tile_cols % simd_lanes == 0);
        auto z3_register_tile_cols = z3_kernel_tile_cols / simd_lanes;
        z3_optimize.add(z3_kernel_tile_rows % 2 == 0);
        z3_optimize.add(z3_kernel_tile_rows > 0 && z3_kernel_tile_rows < simd_register_count);
        z3_optimize.add(z3_register_tile_cols > 0 && z3_register_tile_cols < simd_register_count);
        z3_optimize.add(z3_kernel_tile_rows >= z3_register_tile_cols);
        // z3_kernel_tile_rows 需要保留broadcast后的向量寄存器， 防止反复加载
        //  z3_kernel_tile_rows * z3_register_tile_cols  用于存放外积的结果
        z3_optimize.add(z3_kernel_tile_rows + z3_kernel_tile_rows * z3_register_tile_cols <=
                        simd_register_count);

        // z3_kernel_tile_rows * z3_register_tile_cols: 为了尽可能多的计算指令数目
        z3::optimize::handle z3_handle_x =
            z3_optimize.maximize(z3_kernel_tile_rows * z3_register_tile_cols);
        GALOIS_ASSERT(z3_optimize.check() == z3::sat);
        z3::model z3_model = z3_optimize.get_model();
        auto kernel_tile_rows = z3_model.eval(z3_kernel_tile_rows).get_numeral_int64();
        auto kernel_tile_cols = z3_model.eval(z3_kernel_tile_cols).get_numeral_int64();

        auto ir_tile_mat_type_a = ir_data_type->Tile(kernel_tile_rows, 1)->Tile(1, 32)->Tile(4, 1);
        auto ir_tile_mat_type_b = ir_data_type->Tile(1, kernel_tile_cols)->Tile(32, 1)->Tile(1, 4);
        return std::make_tuple(ir_tile_mat_type_a, ir_tile_mat_type_b,
                               op::AvxMatrixMultiplyKernel::Create(
                                   cpu_info->SimdBits(), kernel_tile_rows, kernel_tile_cols));
    }
};

class GemmTilePolicy {
   public:
    static std::shared_ptr<GemmTilePolicy> Create() {
        std::shared_ptr<GemmTilePolicy> self(new GemmTilePolicy);
        self->ir_neon_gemm_tile_poly = NeonGemmTilePolicy::Create();
        self->ir_avx_gemm_tile_poly = AvxGemmTilePolicy::Create();
        return self;
    };

    std::tuple<std::shared_ptr<ir::TensorType>, std::shared_ptr<ir::TensorType>,
               std::shared_ptr<op::MatrixMultiplyMicroKernel>>
    Tile(std::shared_ptr<ir::TensorType> ir_data_type, std::shared_ptr<NativeCpuInfo> cpu_info) {
        // 当i8时， Neon不支持"mla.16b v1 v2 v3[0]"形式， 必须“mla.16b v1 v2
        // v3”的形式，所以会去采用和avx一样的策略
        if (cpu_info->simd == NativeCpuInfo::Simd::NEON && ir_data_type != ir::i8) {
            return this->ir_neon_gemm_tile_poly->Tile(ir_data_type, cpu_info);
        } else {
            return this->ir_avx_gemm_tile_poly->Tile(ir_data_type, cpu_info);
        }
    }

   private:
    std::shared_ptr<NeonGemmTilePolicy> ir_neon_gemm_tile_poly = nullptr;
    std::shared_ptr<AvxGemmTilePolicy> ir_avx_gemm_tile_poly = nullptr;
};

class GemmOptimizer {
   protected:
    GemmOptimizer() = default;

   public:
    static std::shared_ptr<GemmOptimizer> Create() {
        std::shared_ptr<GemmOptimizer> self(new GemmOptimizer);
        self->cpu_info = NativeCpuInfo::Create();
        self->tile_policy = GemmTilePolicy::Create();
        return self;
    }

    std::shared_ptr<ir::Tensor> PackTensorForTile(std::shared_ptr<ir::Tensor> ir_mat,
                                                  std::shared_ptr<ir::TensorType> ir_tile_type,
                                                  std::shared_ptr<ir::Builder> ir_builder) {
        // 最小裁剪单元尺寸
        auto basic_padding_shape = ir_tile_type->NormalizeShape();
        // 计算所需最小单元的数量
        Eigen::VectorXi64 plane_shape =
            ((ir_mat->type->shape + basic_padding_shape - Eigen::VectorXi64::Ones(2)).array() /
             basic_padding_shape.array())
                .matrix();
        // 计算最终裁剪尺寸
        auto padding_shape = (plane_shape.array() * basic_padding_shape.array()).matrix();
        std::shared_ptr<ir::Tensor> ir_padded_mat = ir_mat;
        if (ir_mat->type->shape != padding_shape) {
            ir_padded_mat = ir_builder->ExpressCreator<op::PaddingCreator>({ir_mat}, padding_shape);
        }
        // 将裁剪后的矩阵分块打包
        auto ir_packed_type = ir::TensorType::Create(ir_tile_type, plane_shape);
        auto ir_packed_mat =
            ir_builder->ExpressCreator<op::PackCreator>({ir_padded_mat}, ir_packed_type);

        return ir_packed_mat;
    }

    std::shared_ptr<ir::Operator> Optimize(std::shared_ptr<ir::Operator> ir_matrix_multiply) {
        ir_matrix_multiply->block->clear();
        auto ir_builder = ir::Builder::Create();
        auto [ir_gemm_operator, scope] = ir_builder->CreateOperator(
            ir_matrix_multiply->GetOperatorType(), ir_matrix_multiply->name + "_gemm");

        auto ir_mat_a = ir_gemm_operator->inputs[0];
        auto ir_mat_b = ir_gemm_operator->inputs[1];

        auto [ir_tile_mat_type_a, ir_tile_mat_type_b, ir_simd_mat_mul_kernel] =
            this->tile_policy->Tile(ir_mat_a->type->DataType(), this->cpu_info);

        ir_builder->matrix_multiply_kernel_queue.push_back(ir_simd_mat_mul_kernel);

        auto ir_packed_mat_a = this->PackTensorForTile(ir_mat_a, ir_tile_mat_type_a, ir_builder);
        auto ir_packed_mat_b = this->PackTensorForTile(ir_mat_b, ir_tile_mat_type_b, ir_builder);
        // 将分块矩阵转为常规矩阵
        auto ir_packed_mat_c = ir_builder->ExpressCreator<op::MatrixMultiplyCreator>(
            {ir_packed_mat_a, ir_packed_mat_b});
        auto ir_packed_mat_mul_operator = Cast<ir::Call>(ir_packed_mat_c)->Operator();

        transform::Each<ir::Grid>(ir_packed_mat_mul_operator,
                                  [](std::shared_ptr<ir::Grid> ir_grid) {
                                      if (ir_grid->unroll_grid) {
                                          UnrollGrid(ir_grid);
                                      };
                                  });

        auto ir_unpacked_mat_c = ir_builder->ExpressCreator<op::UnpackCreator>({ir_packed_mat_c});
        // 裁剪矩阵到原始尺寸
        auto ir_mat_c_type = ir_matrix_multiply->GetOperatorType()->output_type;
        auto ir_mat_c =
            ir_builder->ExpressCreator<op::SliceCreator>({ir_unpacked_mat_c}, ir_mat_c_type->shape);

        ir_builder->Return(ir_mat_c);

        return ir_gemm_operator;
    }

   private:
    std::shared_ptr<NativeCpuInfo> cpu_info = nullptr;
    std::shared_ptr<GemmTilePolicy> tile_policy = nullptr;
};

}  // namespace galois::optimization
