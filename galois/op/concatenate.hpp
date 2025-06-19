#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/copy.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {

class ConcatenateCreator : public op::Creator {
   public:
    static std::shared_ptr<ConcatenateCreator> Create(int64_t dim) {
        auto self = std::make_shared<ConcatenateCreator>();
        self->dim = dim;
        self->name = "Concatenate";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        auto base_type = Cast<ir::TensorType>(ir_input_types[0]);
        auto out_shape = base_type->shape;
        int64_t total = 0;
        for (const auto& t : ir_input_types) {
            auto tensor_type = Cast<ir::TensorType>(t);
            GALOIS_ASSERT(tensor_type->shape.size() == out_shape.size());
            for (size_t i = 0; i < out_shape.size(); ++i) {
                if (i == dim) continue;
                GALOIS_ASSERT(tensor_type->shape[i] == out_shape[i]);
            }
            total += tensor_type->shape[dim];
        }
        out_shape[dim] = total;
        return ir::TensorType::Create(base_type->value_type, out_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output_type = this->InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(ir_output_type);
        this->_Express(ir_inputs, ir_output, ir_builder);
        ir_builder->Return(ir_output);
    }

    void _Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                  std::shared_ptr<ir::Tensor> ir_output, std::shared_ptr<ir::Builder> ir_builder) {
        int64_t offset = 0;
        for (const auto& input : ir_inputs) {
            auto input_shape = input->type->shape;
            Eigen::VectorXi64 out_slice_shape = input_shape;
            auto output_accessor = ir_builder->CreateAccessor(ir_output);
            output_accessor->shift_vector[dim] = offset;
            auto output_slice =
                ir_builder->Create<ir::view::Slice>(output_accessor, out_slice_shape);
            ir_builder->ExpressCreator<op::CopyCreator>({input, output_slice});
            offset += input_shape[dim];
        }
    }

    int64_t dim;
};

}  // namespace galois::op
