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

    void Optimize(std::shared_ptr<ir::OperatorFunction> ir_matrix_multiply) {
        auto ir_mat_type_a = ir_matrix_multiply->input_types[0];
        auto ir_mat_type_b = ir_matrix_multiply->input_types[1];
        auto ir_mat_type_c = ir_matrix_multiply->output_types[2];
        auto ir_mat_a = ir_matrix_multiply->inputs[0];
        auto ir_mat_b = ir_matrix_multiply->inputs[1];
        auto ir_mat_c = ir_matrix_multiply->outputs[0];

        ir_matrix_multiply->values.clear();
        auto ir_builder = ir::Builder::Create();
        ir_builder->kernel_queue.push_back(op::ProductKernel::Create());
        ir_builder->block_stack.push(ir_matrix_multiply);
        ir_builder->iterator_stack.push(ir_matrix_multiply->values.end());

        auto ir_ts_type_a = ir::f32(4, 1)(2, 1)(1, 512)(64, 1);
        auto ir_ts_type_b = ir::f32(1, 4)(1, 2)(512, 1)(1, 64);

        auto ir_packed_mat_a = this->PackTensorForTile(ir_mat_a, ir_ts_type_a, ir_builder);
        auto ir_packed_mat_b = this->PackTensorForTile(ir_mat_b, ir_ts_type_b, ir_builder);
        auto ir_packed_mat_c =
            ir_builder->Express<op::MatrixMultiplyCreator>({ir_packed_mat_a, ir_packed_mat_b});
        auto ir_unpacked_mat_c = ir_builder->Express<op::UnpackCreator>({ir_packed_mat_c});

        auto sp_padding_creator = op::PaddingCreator::Create(ir_mat_c->type->shape);
        sp_padding_creator->AffineExpress({ir_unpacked_mat_c}, {ir_mat_c}, ir_builder);
    }
};

}  // namespace galois::optimization
