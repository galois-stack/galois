#pragma once

#include "galois/ir/ir.hpp"
#include "galois/ir/builder.hpp"
#include "galois/op/creator.hpp"

namespace galois::op {

class GatherCreator : public op::Creator {
   public:
    static std::shared_ptr<GatherCreator> Create() {
        auto self = std::make_shared<GatherCreator>();
        self->name = "Gather";
        self->fullname = self->name;
        return self;
    }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 2);
        auto ir_input_type = ir_input_types[0];
        auto ir_indices_type = ir_input_types[1];
        
        // For nD gather: ir::i64->Tile(n)->Tile(N)
        // indices_shape is [N] (the outer dimension - number of sample points)
        // The inner dimension [n] represents the nD coordinate size
        
        auto indices_shape = ir_indices_type->shape;
        
        // Output shape matches the outer indices shape
        return ir::TensorType::Create(ir_input_type->value_type, indices_shape);
    }

    void Express(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        GALOIS_ASSERT(ir_inputs.size() == 2);
        auto ir_input = ir_inputs[0];
        auto ir_indices = ir_inputs[1];
        
        auto ir_output_type = this->InferType({ir_input->type, ir_indices->type});
        auto ir_output = ir_builder->Alloca(ir_output_type);
        
        // Dynamic nD indexing
        {
            auto [ir_grid, _] = ir_builder->CreateGrid(ir_output_type->shape);
            auto ir_index = ir_builder->CreateIdentityAccessor(ir_indices);
            auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_output);

            // Determine the number of dimensions from the input tensor
            auto input_shape = ir_input->type->shape;
            size_t input_dims = input_shape.size();
            
            // Also get coordinate dimension from indices structure
            // For ir::i64->Tile(coord_dim)->Tile(num_points), need coord_dim
            auto indices_value_type = ir_indices->type->value_type;
            GALOIS_ASSERT(indices_value_type != nullptr, "Invalid indices structure");
            size_t coord_dims = indices_value_type->shape.size();
            GALOIS_ASSERT(coord_dims == 1, "Expected 1D coordinate vector in indices");
            size_t coordinate_size = indices_value_type->shape[0];
            
            GALOIS_ASSERT(coordinate_size == input_dims, 
                         "Coordinate size must match input tensor dimensions");

            std::vector<std::shared_ptr<ir::Tensor>> index_components;
            for (size_t i = 0; i < coordinate_size; ++i) {
                auto ir_index_component = ir_builder->CreateAccessor(ir_index);
                ir_index_component->shift_vector[0] = i;
                index_components.push_back(ir_index_component);
            }
            
            auto ir_indexing = ir_builder->Create<ir::Indexing>(ir_input, index_components);
            ir_builder->Write(ir_indexing, ir_output_accessor);
        }
        
        ir_builder->Return(ir_output);
    }
};

}  // namespace galois::op