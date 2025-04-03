#pragma once

#include "boost/scope/scope_exit.hpp"
#include "c++/z3++.h"
#include "cpuinfo.h"
#include "fmt/format.h"
#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"
#include "galois/transform/transform.hpp"

namespace galois::optimization {

class NativeCpuInfo {
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

   private:
    NativeCpuInfo() = default;

    void DetectCpuFeatures() {
        // 默认值
        simd_bits = 128;
        simd_register_count = 32;
        if (cpuinfo_has_x86_avx512f()) {
            simd_bits = 512;
            simd_register_count = 32;
        } else if (cpuinfo_has_x86_avx2() || cpuinfo_has_x86_avx()) {
            simd_bits = 256;
            simd_register_count = 16;
        } else if (cpuinfo_has_x86_sse2()) {
            simd_bits = 128;
            simd_register_count = 8;
        } else if (cpuinfo_has_arm_neon()) {
            simd_bits = 128;
            simd_register_count = 32;
        }
    }

    int64_t simd_register_count = 0;
    int64_t simd_bits = 0;
    std::vector<int64_t> cache_sizes;
};

inline std::shared_ptr<ir::Grid> GetInnerGrid3(std::shared_ptr<ir::Block> ir_block) {
    auto ir_block_iter =
        std::find_if(RANGE(ir_block->values), [&](std::shared_ptr<ir::Tensor> ir_tensor) {
            if (auto ir_grid = Cast<ir::Grid>(ir_tensor)) {
                if (ir_grid->shape.size() == 3) {
                    return true;
                }
            }
            return false;
        });

    if (ir_block_iter != ir_block->values.end()) {
        return GetInnerGrid3(Cast<ir::Block>(*ir_block_iter));
    } else {
        return Cast<ir::Grid>(ir_block);
    }
}

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

inline void ExpandGrid(std::shared_ptr<ir::Grid> ir_grid) {
    auto index_grid = GenerateIndexGrid(ir_grid->shape);
    GALOIS_ASSERT(ir_grid->parent_block);
    auto ir_grid_iter = std::find(RANGE(ir_grid->parent_block->values), ir_grid);

    auto cloner = ir::Cloner::Create();
    auto ir_external_tensor_set = transform::CaptureExternalTensors(ir_grid);
    for (auto ir_tensor : ir_external_tensor_set) {
        cloner->tensor_dict[ir_tensor] = ir_tensor;
    }

    for (auto index : index_grid) {
        for (auto ir_value : Clone(ir_grid->values)) {
            auto ir_value_clone = ir_value->Clone(cloner);
            if (auto ir_accessor = Cast<ir::Accessor>(ir_value_clone)) {
                if (ir_accessor->transform_matrix.size()) {
                    ir_accessor->shift_vector += ir_accessor->transform_matrix * index;
                    ir_accessor->transform_matrix.resize(0, 0);
                }
            }

            ir_grid->parent_block->values.insert(ir_grid_iter, ir_value_clone);
        }
    }

    ir_grid->parent_block->values.remove(ir_grid);
    ir_grid->Finalize();
}

class GemmTilePolicy {
   public:
    static std::shared_ptr<GemmTilePolicy> Create() {
        std::shared_ptr<GemmTilePolicy> self(new GemmTilePolicy);
        return self;
    };

    std::tuple<int64_t, int64_t> GetSimdTileShape(std::shared_ptr<ir::TensorType> ir_data_type,
                                                  std::shared_ptr<NativeCpuInfo> cpu_info) {
        auto simd_lanes = (cpu_info->SimdBits() / 8) / ir_data_type->bytes;
        auto simd_lanes_b = simd_lanes;  // b的simd lanes是固定的
        int32_t simd_register_count = cpu_info->SimdRegisterCount();

        /// 通过Z3来求解寄存器分块， 该问题不是一个线性规划问题， 所以采用Z3来处理
        z3::context z3_context;
        z3::params z3_params(z3_context);
        z3_params.set("priority", z3_context.str_symbol("register tile"));
        z3::optimize z3_optimize(z3_context);
        z3_optimize.set(z3_params);
        // a的simd lanes, 通过求解得来， 因为存在寄存器不够用的情况， 所以需要裁剪
        z3::expr z3_simd_lanes_a = z3_context.bv_const("z3_simd_lanes_a", 32);
        z3_optimize.add(z3_simd_lanes_a > 0 && z3_simd_lanes_a <= int32_t(simd_lanes));
        // 需要是2的倍数， 为了方便后续的计算。 若去除此限制， 需要考虑内存对齐等更多问题
        // 约束：2 的幂，且范围在 1 到 32
        z3_optimize.add((z3_simd_lanes_a & (z3_simd_lanes_a - 1)) == 0);
        // 有瑕疵， AVX的shuffe指令可能还需要寄存器， 这里不进一步细化
        z3_optimize.add(z3_simd_lanes_a + 2 < simd_register_count);
        z3::optimize::handle z3_handle_x = z3_optimize.maximize(z3_simd_lanes_a);
        GALOIS_ASSERT(z3_optimize.check() == z3::sat);
        z3::model z3_model = z3_optimize.get_model();
        auto simd_lanes_a = z3_model.eval(z3_simd_lanes_a).get_numeral_int64();
        return {simd_lanes_a, simd_lanes_b};
    }

    std::tuple<int64_t, int64_t> GetRegisterTileShape(std::shared_ptr<ir::TensorType> ir_data_type,
                                                      int64_t simd_lanes_a,
                                                      std::shared_ptr<NativeCpuInfo> cpu_info) {
        int32_t simd_register_count = cpu_info->SimdRegisterCount();
        /// 通过Z3来求解寄存器分块， 该问题不是一个线性规划问题， 所以采用Z3来处理
        z3::context z3_context;
        z3::params z3_params(z3_context);
        z3_params.set("priority", z3_context.str_symbol("register tile"));
        z3::optimize z3_optimize(z3_context);
        z3_optimize.set(z3_params);
        // a的simd lanes, 通过求解得来， 因为存在寄存器不够用的情况， 所以需要裁剪
        z3::expr z3_register_rows = z3_context.int_const("z3_register_rows");
        z3::expr z3_register_cols = z3_context.int_const("z3_register_cols");
        z3_optimize.add(z3_register_rows > 0);
        z3_optimize.add(z3_register_cols > 0);
        z3_optimize.add(z3_register_rows >= z3_register_cols);
        // 有瑕疵， AVX的shuffe指令可能还需要寄存器， 这里不进一步细化， 因为底层llvm怎么生成不好说
        z3_optimize.add(z3_register_rows + z3_register_cols +
                            z3_register_rows * z3_register_cols * int32_t(simd_lanes_a) <
                        simd_register_count);
        // 最大化尺寸
        z3::optimize::handle z3_handle_x =
            z3_optimize.maximize(z3_register_rows * z3_register_cols);
        GALOIS_ASSERT(z3_optimize.check() == z3::sat);
        z3::model z3_model = z3_optimize.get_model();
        auto register_rows = z3_model.eval(z3_register_rows).get_numeral_int64();
        auto register_cols = z3_model.eval(z3_register_cols).get_numeral_int64();
        return {register_rows, register_cols};
    }

    std::tuple<std::shared_ptr<ir::TensorType>, std::shared_ptr<ir::TensorType>,
               std::shared_ptr<op::SimdMatrixMultiplyKernel>>
    Tile(std::shared_ptr<ir::TensorType> ir_data_type, std::shared_ptr<NativeCpuInfo> cpu_info) {
        auto [simd_lanes_a, simd_lanes_b] = this->GetSimdTileShape(ir_data_type, cpu_info);
        auto [register_rows, register_cols] =
            this->GetRegisterTileShape(ir_data_type, simd_lanes_a, cpu_info);

        auto ir_tile_mat_type_a =
            ir_data_type->Tile(simd_lanes_a, 1)->Tile(register_rows, 1)->Tile(1, 32)->Tile(4, 1);
        auto ir_tile_mat_type_b =
            ir_data_type->Tile(1, simd_lanes_b)->Tile(1, register_cols)->Tile(32, 1)->Tile(1, 4);
        return std::make_tuple(ir_tile_mat_type_a, ir_tile_mat_type_b,
                               op::SimdMatrixMultiplyKernel::Create(cpu_info->SimdBits()));
    }
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
        // TODO: 后期使用编译技术， 优化此操作， 无需手动写
        boost::scope::scope_exit free_mem([&] { ir_builder->Create<ir::Free>(ir_padded_mat); },
                                          false);
        if (ir_mat->type->shape != padding_shape) {
            ir_padded_mat = ir_builder->Express<op::PaddingCreator>({ir_mat}, padding_shape);
            free_mem.set_active(true);
        }
        // 将裁剪后的矩阵分块打包
        auto ir_packed_type = ir::TensorType::Create(ir_tile_type, plane_shape);
        auto ir_packed_mat = ir_builder->Express<op::PackCreator>({ir_padded_mat}, ir_packed_type);

        return ir_packed_mat;
    }

    std::shared_ptr<ir::OperatorFunction> Optimize(
        std::shared_ptr<ir::OperatorFunction> ir_matrix_multiply) {
        ir_matrix_multiply->values.clear();
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
        auto ir_packed_mat_c =
            ir_builder->Express<op::MatrixMultiplyCreator>({ir_packed_mat_a, ir_packed_mat_b});
        // TODO: 需要更通用的方式来定位grid
        auto ir_register_tile_grid =
            GetInnerGrid3(Cast<ir::Call>(ir_packed_mat_c)->OperatorFunction());
        ExpandGrid(ir_register_tile_grid);

        ir_builder->Create<ir::Free>(ir_packed_mat_a);
        ir_builder->Create<ir::Free>(ir_packed_mat_b);

        auto ir_unpacked_mat_c = ir_builder->Express<op::UnpackCreator>({ir_packed_mat_c});
        // 裁剪矩阵到原始尺寸
        auto ir_mat_c_type = ir_matrix_multiply->GetOperatorType()->output_type;
        auto ir_mat_c =
            ir_builder->Express<op::SliceCreator>({ir_unpacked_mat_c}, ir_mat_c_type->shape);

        ir_builder->Create<ir::Free>(ir_packed_mat_c);
        ir_builder->Create<ir::Free>(ir_unpacked_mat_c);

        ir_builder->Create<ir::Return>(ir_mat_c);

        return ir_gemm_operator;
    }

   private:
    std::shared_ptr<NativeCpuInfo> cpu_info = nullptr;
    std::shared_ptr<GemmTilePolicy> tile_policy = nullptr;
};

}  // namespace galois::optimization
