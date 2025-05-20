#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/tensor.hpp"
#include "galois/op/copy.hpp"

namespace galois::op {

class TransposeCreator : public op::Creator {
   public:
    int64_t dim0;
    int64_t dim1;

    static std::shared_ptr<TransposeCreator> Create(int64_t dim0, int64_t dim1) {
        auto self = std::make_shared<TransposeCreator>();
        self->name = "Transpose";
        self->fullname = self->name;
        self->dim0 = dim0;
        self->dim1 = dim1;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        auto input_type = ir_input_types[0];
        int64_t rank = input_type->shape.size();
        GALOIS_ASSERT(dim0 >= 0 && dim0 < rank);
        GALOIS_ASSERT(dim1 >= 0 && dim1 < rank);
        GALOIS_ASSERT(dim0 != dim1);

        Eigen::VectorXi64 new_shape = input_type->shape;
        std::swap(new_shape(dim0), new_shape(dim1));
        return ir::TensorType::Create(input_type->value_type, new_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(ir_inputs.size() == 1);
        auto input = ir_inputs[0];
        auto ir_transpose_view = ir_builder->Create<ir::TransposeView>(input, dim0, dim1);
        auto out_type = this->InferType({input->type});
        auto output = ir_builder->Alloca(out_type);
        op::CopyCreator::Create()->ExpressInline(ir_transpose_view, output, ir_builder);

        ir_builder->Return(output);
    }
};

}  // namespace galois::op
