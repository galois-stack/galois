#pragma once

#include "galois/ir/tensor/block.hpp"
#include "galois/ir/tensor/input.hpp"
#include "galois/ir/tensor_type.hpp"

namespace galois::ir {

class Operator : public Tensor {
   public:
    static std::shared_ptr<Operator> Create(std::shared_ptr<OperatorType> ir_operator_type) {
        std::shared_ptr<Operator> self(new Operator);
        self->type = ir_operator_type;
        self->block = Block::Create();
        std::transform(RANGE(ir_operator_type->ir_input_types), std::back_inserter(self->inputs),
                       [](std::shared_ptr<TensorType> ir_type) { return Input::Create(ir_type); });

        self->tag = "Operator";
        return self;
    }

    std::shared_ptr<OperatorType> GetOperatorType() { return Cast<OperatorType>(this->type); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Operator>(this->shared_from_this()));
    }

   public:
    std::shared_ptr<Block> block = nullptr;
    std::vector<std::shared_ptr<Input>> inputs;
    std::shared_ptr<pir::Function> pir_function = nullptr;
};

}  // namespace galois::ir
