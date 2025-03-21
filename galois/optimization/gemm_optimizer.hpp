#pragma once

#include <cpuinfo.h>

#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"

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

class GemmOptimizer {
   protected:
    GemmOptimizer() = default;

   public:
    static std::shared_ptr<GemmOptimizer> Create() {
        std::shared_ptr<GemmOptimizer> self(new GemmOptimizer);
        self->cpu_info = NativeCpuInfo::Create();
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
        if (ir_mat->type->shape != padding_shape) {
            ir_padded_mat = ir_builder->Express<op::PaddingCreator>({ir_mat}, padding_shape);
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

        ir_builder->matrix_multiply_kernel_queue.push_back(
            op::VectorizedMatrixMultiplyKernel::Create(this->cpu_info->SimdBits()));

        auto ir_mat_a = ir_gemm_operator->inputs[0];
        auto ir_mat_b = ir_gemm_operator->inputs[1];

        auto simd_lines = (this->cpu_info->SimdBits() / 8) / ir_mat_a->type->DataType()->bytes;
        auto ir_tile_mat_type_a = ir::f32->Tile(simd_lines, 1);
        auto ir_tile_mat_type_b = ir::f32->Tile(1, simd_lines);
        // GALOIS_ASSERT(this->cpu_info->SimdRegisterCount() == 32);
        if (simd_lines == 2) {
            ir_tile_mat_type_a = ir_tile_mat_type_a->Tile(4, 1)->Tile(1, 32)->Tile(2, 1);
            ir_tile_mat_type_b = ir_tile_mat_type_b->Tile(1, 3)->Tile(32, 1)->Tile(1, 2);
        } else if (simd_lines == 4) {
            ir_tile_mat_type_a = ir_tile_mat_type_a->Tile(3, 1)->Tile(1, 32)->Tile(4, 1);
            ir_tile_mat_type_b = ir_tile_mat_type_b->Tile(1, 2)->Tile(32, 1)->Tile(1, 4);
        } else if (simd_lines == 8) {
            ir_tile_mat_type_a = ir_tile_mat_type_a->Tile(2, 1)->Tile(1, 64)->Tile(4, 1);
            ir_tile_mat_type_b = ir_tile_mat_type_b->Tile(1, 1)->Tile(64, 1)->Tile(1, 4);
        } else if (simd_lines == 16) {
            ir_tile_mat_type_a = ir_tile_mat_type_a->Tile(1, 1)->Tile(1, 128)->Tile(4, 1);
            ir_tile_mat_type_b = ir_tile_mat_type_b->Tile(1, 1)->Tile(128, 1)->Tile(1, 4);
        } else {
            GALOIS_UNREACHABLE;
        }

        auto ir_packed_mat_a = this->PackTensorForTile(ir_mat_a, ir_tile_mat_type_a, ir_builder);
        auto ir_packed_mat_b = this->PackTensorForTile(ir_mat_b, ir_tile_mat_type_b, ir_builder);
        // 将分块矩阵转为常规矩阵
        auto ir_packed_mat_c =
            ir_builder->Express<op::MatrixMultiplyCreator>({ir_packed_mat_a, ir_packed_mat_b});
        auto ir_unpacked_mat_c = ir_builder->Express<op::UnpackCreator>({ir_packed_mat_c});
        // 裁剪矩阵到原始尺寸
        auto ir_mat_c_type = ir_matrix_multiply->GetOperatorType()->output_type;
        auto sp_padding_creator = op::SliceCreator::Create(ir_mat_c_type->shape);
        auto ir_mat_c =
            ir_builder->Express<op::SliceCreator>({ir_unpacked_mat_c}, ir_mat_c_type->shape);
        ir_builder->Create<ir::Return>(ir_mat_c);

        return ir_gemm_operator;
    }

   private:
    std::shared_ptr<NativeCpuInfo> cpu_info = nullptr;
};

}  // namespace galois::optimization
