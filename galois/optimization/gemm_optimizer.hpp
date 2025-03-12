#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"
#include "tile_size_calculator.hpp"

namespace galois::optimization {

class GemmOptimizer {
   protected:
    GemmOptimizer() = default;

   public:
    static std::shared_ptr<GemmOptimizer> Create() {
        std::shared_ptr<GemmOptimizer> self(new GemmOptimizer);
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
        auto ir_padded_mat = ir_builder->Express<op::PaddingCreator>({ir_mat}, padding_shape);
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

        ir_builder->kernel_queue.push_back(op::MatrixMultiplyKernel4x1x4::Create());
        ir_builder->kernel_queue.push_back(op::MatrixMultiplyKernel8x1x8::Create());

        // 矩阵维度
        int M = ir_matrix_multiply->inputs[0].get()->type->shape[0];
        int K = ir_matrix_multiply->inputs[0].get()->type->shape[1];
        int N = ir_matrix_multiply->inputs[1].get()->type->shape[1];
        // 缓存配置（自动检测）
        CacheConfig cache;
        // 创建分块大小求解器实例
        TileSizeCalculator<float> calculator(cache);
        TileSize ts = calculator.compute(M, N, K);
        auto ir_tile_mat_type_a = ir::f32->Tile(ts.ti_inner, 1)->Tile(ts.ti_mid, 1)->Tile(1, ts.tk_mid)->Tile(ts.ti_outer, 1);
        auto ir_tile_mat_type_b = ir::f32->Tile(1, ts.tj_inner)->Tile(1, ts.tj_mid)->Tile(ts.tk_mid, 1)->Tile(1, ts.tj_outer);

        // std::cout << "Matrix A split format: ir::f32(" << ts.ti_inner << ", 1)(" << ts.ti_mid << ", 1)(1, " 
        //       << ts.tk_mid << ")(" << ts.ti_outer << ", 1)\n";
        // std::cout << "Matrix B split format: ir::f32(1, " << ts.tj_inner << ")(1, " << ts.tj_mid << ")(" 
        //       << ts.tk_mid << ", 1)(1, " << ts.tj_outer << ")\n";
        
        // auto ir_tile_mat_type_a = ir::f32->Tile(4, 1)->Tile(2, 1)->Tile(1, 512)->Tile(64, 1);
        // auto ir_tile_mat_type_b = ir::f32->Tile(1, 4)->Tile(1, 3)->Tile(512, 1)->Tile(1, 64);

        auto ir_mat_a = ir_gemm_operator->inputs[0];
        auto ir_mat_b = ir_gemm_operator->inputs[1];

        auto ir_packed_mat_a = this->PackTensorForTile(ir_mat_a, ir_tile_mat_type_a, ir_builder);
        auto ir_packed_mat_b = this->PackTensorForTile(ir_mat_b, ir_tile_mat_type_b, ir_builder);
        // 将分块矩阵转为常规矩阵
        auto ir_packed_mat_c =
            ir_builder->Express<op::MatrixMultiplyCreator>({ir_packed_mat_a, ir_packed_mat_b});
        auto ir_unpacked_mat_c = ir_builder->Express<op::UnpackCreator>({ir_packed_mat_c});
        // 裁剪矩阵到原始尺寸
        auto ir_mat_c_type = ir_matrix_multiply->GetOperatorType()->out_type;
        auto sp_padding_creator = op::SliceCreator::Create(ir_mat_c_type->shape);
        auto ir_mat_c =
            ir_builder->Express<op::SliceCreator>({ir_unpacked_mat_c}, ir_mat_c_type->shape);
        ir_builder->Create<ir::Return>(ir_mat_c);

        return ir_gemm_operator;
    }
};

}  // namespace galois::optimization
