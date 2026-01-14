#pragma once

#include "galois/ir/tensor/operator.hpp"

namespace galois::ir {

class Call : public Instruction {
   protected:
    Call() = default;

   public:
    static std::shared_ptr<Call> Create(std::shared_ptr<Operator> ir_operator,
                                        std::vector<std::shared_ptr<Tensor>> ir_inputs) {
        std::shared_ptr<Call> self(new Call);
        self->input_size = ir_inputs.size();
        self->OperandResize(1 + self->input_size);
        self->Operator(ir_operator);
        auto iter_inputs = ir_inputs.begin();
        for (int64_t i = 0; i < self->InputSize(); ++i, ++iter_inputs) {
            GALOIS_ASSERT(ir_operator->GetOperatorType()->ir_input_types[i] == ir_inputs[i]->type);
            self->Input(i, *iter_inputs);
        }
        self->type = ir_operator->GetOperatorType()->output_type;
        self->tag = "Call";
        return self;
    }

    std::shared_ptr<ir::Operator> Operator() { return Cast<class Operator>(this->GetOperand(0)); }
    void Operator(std::shared_ptr<ir::Operator> ir_operator) { this->SetOperand(0, ir_operator); }

    std::shared_ptr<Tensor> Input(int64_t i) { return this->GetOperand(1 + i); }
    void Input(int64_t i, std::shared_ptr<Tensor> ir_argument) {
        this->SetOperand(1 + i, ir_argument);
    }
    int64_t InputSize() { return this->input_size; }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<Call>(this->shared_from_this()));
    }

   private:
    int64_t input_size;
    int64_t output_size;
};

}  // namespace galois::ir
