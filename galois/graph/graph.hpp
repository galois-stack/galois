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
                               std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                               std::shared_ptr<ir::Builder> ir_builder) = 0;

    ~OperatorCreator() {}
};

}  // namespace galois::op

namespace galois::graph {

using namespace ir;

class OperatorTag {
   public:
    virtual ~OperatorTag() {}
};

class Add : public OperatorTag {};

class Sub : public OperatorTag {};
class Div : public OperatorTag {};
class Mul : public OperatorTag {};
class Slice : public OperatorTag {};
class Pack : public OperatorTag {};
class Copy : public OperatorTag {};

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
    // std::string operator_name;
    std::shared_ptr<op::OperatorCreator> operator_creator;
    std::vector<std::shared_ptr<ComputeNode>> inputs;
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
