#pragma once

#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"

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
        Eigen::VectorXi64 tile_shape(2);
        // TODO: tile_shape is magical number
        tile_shape[0] = 512;
        tile_shape[1] = 512;

        Eigen::VectorXi64 plane_shape =
            ((ir_mat->type->shape + tile_shape - Eigen::VectorXi64::Ones(2)).array() /
             tile_shape.array())
                .matrix();
        auto padding_shape = (plane_shape.array() * tile_shape.array()).matrix();
        auto ir_padded_mat = ir_builder->Express<op::PaddingCreator>({ir_mat}, padding_shape);
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

        auto ir_ts_type_a = ir::f32->Tile(4, 1)->Tile(2, 1)->Tile(1, 512)->Tile(64, 1);
        auto ir_ts_type_b = ir::f32->Tile(1, 4)->Tile(1, 2)->Tile(512, 1)->Tile(1, 64);

        auto ir_mat_a = ir_gemm_operator->inputs[0];
        auto ir_mat_b = ir_gemm_operator->inputs[1];

        auto ir_packed_mat_a = this->PackTensorForTile(ir_mat_a, ir_ts_type_a, ir_builder);
        auto ir_packed_mat_b = this->PackTensorForTile(ir_mat_b, ir_ts_type_b, ir_builder);
        auto ir_packed_mat_c =
            ir_builder->Express<op::MatrixMultiplyCreator>({ir_packed_mat_a, ir_packed_mat_b});
        auto ir_unpacked_mat_c = ir_builder->Express<op::UnpackCreator>({ir_packed_mat_c});

        auto ir_mat_c_type = ir_matrix_multiply->GetOperatorType()->out_type;
        auto sp_padding_creator = op::SliceCreator::Create(ir_mat_c_type->shape);
        auto ir_mat_c =
            ir_builder->Express<op::SliceCreator>({ir_unpacked_mat_c}, ir_mat_c_type->shape);
        ir_builder->Create<ir::Return>(ir_mat_c);

        return ir_gemm_operator;
    }
};

}  // namespace galois::optimization
