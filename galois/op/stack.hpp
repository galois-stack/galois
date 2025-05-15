#pragma once

#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/copy.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {

class StackCreator : public Creator {
   public:
    static std::shared_ptr<StackCreator> Create(int64_t dim) {
        auto self = std::make_shared<StackCreator>();
        self->name = "Stack";
        self->fullname = self->name;
        self->dim = dim;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(!ir_input_types.empty());
        auto first_input_type = ir_input_types.front();
        auto first_input_shape = first_input_type->shape;
        for (const auto& t : ir_input_types) {
            GALOIS_ASSERT(t->shape == first_input_shape && "All inputs must have same shape");
        }
        int64_t input_rank = first_input_shape.size();
        GALOIS_ASSERT(dim >= 0 && dim <= input_rank && "dim out of range");
        Eigen::VectorXi64 out_shape(input_rank + 1);
        for (int64_t i = 0, j = 0; i < out_shape.size(); ++i) {
            if (i == dim) {
                out_shape[i] = ir_input_types.size();
            } else {
                out_shape[i] = first_input_shape[j++];
            }
        }
        return ir::TensorType::Create(first_input_type->value_type, out_shape);
    }

    void StackLoop(const std::vector<std::shared_ptr<ir::Tensor>>& ir_inputs,
                   std::shared_ptr<ir::Tensor> ir_output, std::shared_ptr<ir::Builder> ir_builder,
                   Eigen::VectorXi64& out_indices, int axis, int dim) {
        auto out_shape = ir_output->type->shape;
        if (axis == out_shape.size()) {
            int input_idx = out_indices[dim];
            Eigen::VectorXi64 input_indices(out_shape.size() - 1);
            for (int i = 0, j = 0; i < out_indices.size(); ++i) {
                if (i == dim) continue;
                input_indices[j++] = out_indices[i];
            }

            auto input_accessor = ir_builder->CreateAccessor(ir_inputs[input_idx]);
            auto input_slice = ir_builder->Create<ir::SliceView>(
                input_accessor, Eigen::VectorXi64::Ones(input_indices.size()));
            input_accessor->shift_vector = input_indices;

            auto output_accessor = ir_builder->CreateAccessor(ir_output);
            auto output_slice = ir_builder->Create<ir::SliceView>(
                output_accessor, Eigen::VectorXi64::Ones(out_indices.size()));
            output_accessor->shift_vector = out_indices;

            ir_builder->ExpressCreator<CopyCreator>({input_slice, output_slice});
            return;
        }
        for (int64_t i = 0; i < out_shape[axis]; ++i) {
            out_indices[axis] = i;
            StackLoop(ir_inputs, ir_output, ir_builder, out_indices, axis + 1, dim);
        }
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(!ir_inputs.empty());
        auto output_type = InferType(ir::GetTensorTypes(ir_inputs));
        auto ir_output = ir_builder->Alloca(output_type);
        Eigen::VectorXi64 out_indices(output_type->shape.size());
        out_indices.setZero();
        StackLoop(ir_inputs, ir_output, ir_builder, out_indices, 0, dim);
        ir_builder->Return(ir_output);
    }

    int64_t dim = 0;
};

}  // namespace galois::op
