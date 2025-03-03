#pragma once

#include <algorithm>
#include <list>
#include <memory>
#include <vector>

#include "Eigen/Dense"
#include "galois/assert.hpp"
#include "galois/helper.hpp"
#include "galois/ir/builder.hpp"
#include "galois/ir/ir.hpp"
#include "galois/named.hpp"
// #include "galois/op/op.hpp"

namespace galois::op {

class OperatorCreator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) = 0;
    virtual void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                               std::shared_ptr<ir::Builder> ir_builder) = 0;

    ~OperatorCreator() {}
};

class SetZeroCreator : public OperatorCreator {
   public:
    static std::shared_ptr<SetZeroCreator> Create() { return std::make_shared<SetZeroCreator>(); }

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        GALOIS_ASSERT(ir_input_types.size() == 1);
        return ir_input_types.front();
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_input = ir_inputs.front();
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(ir_input->type->shape);
        auto ir_accessor = ir_builder->CreateIdentityAccessor(ir_input);
        if (ir_accessor->type->IsScalar()) {
            auto ir_zero = ir_builder->Create<ir::ConstantFloat>(ir::FloatType::Create(32), 0.0);
            ir_builder->Create<ir::Write>(ir_zero, ir_accessor);
        } else {
            this->AffineExpress({ir_accessor}, ir_builder);
        }
    }
};

class UnaryOperatorCreator : public OperatorCreator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type) = 0;

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return this->InferTypeImpl(ir_input_types.front());
    }

    virtual void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input,
                                   std::shared_ptr<ir::Tensor> ir_output,
                                   std::shared_ptr<ir::Builder> ir_builder) = 0;

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(this->InferTypeImpl(ir_inputs[0]->type));
        this->AffineExpressImpl(ir_inputs[0], ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }
};

class BinaryOperatorCreator : public OperatorCreator {
   public:
    virtual std::shared_ptr<ir::TensorType> InferTypeImpl(
        std::shared_ptr<ir::TensorType> ir_input_type0,
        std::shared_ptr<ir::TensorType> ir_input_type1) = 0;

    std::shared_ptr<ir::TensorType> InferType(
        std::vector<std::shared_ptr<ir::TensorType>> ir_input_types) override {
        return this->InferTypeImpl(ir_input_types.front(), ir_input_types.back());
    }

    virtual void AffineExpressImpl(std::shared_ptr<ir::Tensor> ir_input0,
                                   std::shared_ptr<ir::Tensor> ir_input1,
                                   std::shared_ptr<ir::Tensor> ir_output,
                                   std::shared_ptr<ir::Builder> ir_builder) = 0;

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::shared_ptr<ir::Builder> ir_builder) override {
        auto ir_output = ir_builder->Create<ir::Alloca>(
            this->InferTypeImpl(ir_inputs[0]->type, ir_inputs[1]->type));
        set_zero_creator->AffineExpress({ir_output}, ir_builder);
        this->AffineExpressImpl(ir_inputs[0], ir_inputs[1], ir_output, ir_builder);
        ir_builder->Create<ir::Return>(ir_output);
    }

   private:
    std::shared_ptr<SetZeroCreator> set_zero_creator = SetZeroCreator::Create();
};

}  // namespace galois::op

namespace galois::graph {
using namespace ir;

class OperatorTag {
   public:
    virtual ~OperatorTag() {}
};

class MatrixMultiply {};

// static op::AddCreator add_creator;

class ComputeNode : public Instruction {
   protected:
    ComputeNode() = default;

   public:
    static std::shared_ptr<ComputeNode> Create(
        std::shared_ptr<op::OperatorCreator> ir_operator_creator,
        std::vector<std::shared_ptr<ComputeNode>> ir_inputs) {
        std::shared_ptr<ComputeNode> self(new ComputeNode);
        self->operator_creator = ir_operator_creator;
        self->inputs = ir_inputs;

        std::vector<std::shared_ptr<TensorType>> ir_types;
        std::transform(RANGE(ir_inputs), std::back_inserter(ir_types),
                       [](std::shared_ptr<ComputeNode> ir_value) { return ir_value->type; });

        self->type = self->operator_creator->InferType(ir_types);
        self->tag = "ComputeNode";
        return self;
    }

    virtual ~ComputeNode() {}

   public:
    std::shared_ptr<op::OperatorCreator> operator_creator;
    std::vector<std::shared_ptr<ComputeNode>> inputs;

    std::shared_ptr<ir::Tensor> ir_tensor;
};

class Input : public ComputeNode {
   public:
    static std::shared_ptr<Input> Create(std::shared_ptr<TensorType> ir_type) {
        std::shared_ptr<Input> self(new Input);
        self->type = ir_type;
        self->tag = "Input";
        return self;
    }
};

class ComputeGraph : public ComputeNode {
   protected:
    ComputeGraph() = default;

   public:
    static std::shared_ptr<ComputeGraph> Create() {
        std::shared_ptr<ComputeGraph> self(new ComputeGraph);
        self->tag = "ComputeGraph";
        return self;
    };

    static void BuildComputeGraphImp(std::shared_ptr<ComputeNode> ir_compute,
                                     ComputeGraph* ir_module) {
        ir_module->computes.push_front(ir_compute);
        for (auto ir_input : ir_compute->inputs) {
            BuildComputeGraphImp(ir_input, ir_module);
        }
    }

    static std::shared_ptr<ComputeGraph> BuildComputeGraph(std::shared_ptr<ComputeNode> ir_compute,
                                                           std::string name) {
        std::shared_ptr<ComputeGraph> self(new ComputeGraph);
        self->fullname = name;
        self->name = name;
        BuildComputeGraphImp(ir_compute, self.get());
        std::list<std::shared_ptr<ComputeNode>> ir_computes;
        for (auto ir_compute : self->computes) {
            if (!std::count(RANGE(ir_computes), ir_compute)) {
                ir_computes.push_back(ir_compute);
            }
            if (Is<Input>(ir_compute)) {
                self->inputs.push_back(ir_compute);
            }
        }
        self->type = ir_compute->type;
        self->computes = ir_computes;

        std::reverse(RANGE(self->inputs));

        self->outputs.push_back(ir_compute);
        // self->computes.push_back(self->output);
        self->tag = "ComputeGraph";
        return self;
    }

    void FixInputs() {}

   public:
    std::vector<std::shared_ptr<ComputeNode>> inputs;
    std::vector<std::shared_ptr<ComputeNode>> outputs;
    // std::shared_ptr<ComputeNode> output;
    // std::vector < std::shared_ptr<ComputeNode> output
    std::list<std::shared_ptr<ComputeNode>> computes;
};

// inline void ConvertToAffine(std::shared_ptr<ComputeGraph> ir_module) {}

}  // namespace galois::graph
