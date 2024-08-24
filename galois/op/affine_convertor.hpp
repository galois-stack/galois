#pragma once

#include "galois/graph/graph.hpp"
#include "galois/ir/ir.hpp"
#include "galois/op/op.hpp"
// #include "galois/op/matrix_multiply.hpp"

namespace galois::graph {

class AffineConvertor {
   public:
    static std::shared_ptr<AffineConvertor> Create() {
        std::shared_ptr<AffineConvertor> self(new AffineConvertor);
        self->ir_builder = Builder::Create();
        self->ir_builder->kernel_queue.push_back(std::make_shared<op::ProductKernel>());
        // self->ir_builder->kernel_queue.push_back(std::make_shared<op::ProductKernel2>());

        return self;
    }

    std::shared_ptr<OperatorInstance> EmitModule(std::shared_ptr<ComputeGraph> ir_module) {
        /// TODO: 名字还需要进一步细化
        std::vector<std::shared_ptr<TensorType>> ir_input_types;
        /// TODO: output 需要处理
        std::vector<std::shared_ptr<TensorType>> ir_output_types = {ir_module->type};
        std::transform(RANGE(ir_module->inputs), std::back_inserter(ir_input_types),
                       [](std::shared_ptr<ComputeNode> ir_compute) { return ir_compute->type; });
        // std::transform(RANGE(ir_module->outputs), std::back_inserter(ir_output_types),
        //    [](std::shared_ptr<ComputeNode> ir_compute) { return ir_compute->type; });
        auto [ir_operator, scope_guard] =
            this->ir_builder->CreateOperator(ir_input_types, ir_output_types, ir_module->fullname);

        for (int64_t i = 0; i < ir_module->inputs.size(); ++i) {
            this->tensor_map[ir_module->inputs[i]] = ir_operator->inputs[i];
        }
        for (int64_t i = 0; i < ir_module->outputs.size(); ++i) {
            this->tensor_map[ir_module->outputs[i]] = ir_operator->outputs[i];
        }

        this->tensor_map[ir_module] = ir_operator->outputs[0];

        for (auto ir_compute : ir_module->computes) {
            this->EmitCompute(ir_compute);
        }

        /// TODO: 释放temp tensor, 后期需要优化
        for (auto ir_temp_tensor : this->ir_builder->temp_tensors_stack.top()) {
            this->ir_builder->Create<Free>(ir_temp_tensor);
        }

        return ir_operator;
    }

    void EmitCompute(std::shared_ptr<ComputeNode> ir_compute) {
        if (Is<Input>(ir_compute)) {
            return;
        }

        if (!this->tensor_map.count(ir_compute)) {
            GALOIS_ASSERT(ir_compute->type);
            auto ir_alloca = this->ir_builder->Create<ir::Alloca>(ir_compute->type);
            this->tensor_map[ir_compute] = ir_alloca;
            this->ir_builder->temp_tensors_stack.top().push_back(ir_alloca);

            // TODO: 初始化
            op::SetZeroCreator set_zero_creator;
            set_zero_creator.AffineExpress({ir_alloca}, {}, ir_builder);
        }

        auto ir_tensor = this->tensor_map[ir_compute];
        auto name =
            ir_compute->name == "" ? "node" + std::to_string(compute_count) : ir_compute->name;
        ++this->compute_count;
        std::vector<std::shared_ptr<TensorType>> ir_input_types;
        /// TODO: output 需要处理
        std::vector<std::shared_ptr<TensorType>> ir_output_types = {ir_compute->type};
        std::transform(RANGE(ir_compute->inputs), std::back_inserter(ir_input_types),
                       [](std::shared_ptr<ComputeNode> ir_compute) { return ir_compute->type; });
        std::shared_ptr<OperatorInstance> ir_operator_tmp;
        {
            auto [ir_operator, scope_guard] =
                this->ir_builder->CreateOperator(ir_input_types, ir_output_types, name);
            ir_compute->operator_creator->AffineExpress(ir_operator->inputs, ir_operator->outputs,
                                                        this->ir_builder);
            ir_operator_tmp = ir_operator;
        }
        std::vector<std::shared_ptr<ir::Tensor>> input_tensors;
        std::transform(RANGE(ir_compute->inputs), std::back_inserter(input_tensors),
                       [&](std::shared_ptr<ComputeNode> ir_tmp_compute) {
                           return this->tensor_map.at(ir_tmp_compute);
                       });
        std::vector<std::shared_ptr<ir::Tensor>> output_tensors = {this->tensor_map.at(ir_compute)};
        this->ir_builder->Create<Call>(ir_operator_tmp, input_tensors, output_tensors);
    }

   public:
    std::shared_ptr<OperatorInstance> ir_entry_operator;
    std::shared_ptr<Builder> ir_builder;
    std::map<std::shared_ptr<ComputeNode>, std::shared_ptr<ir::Tensor>> tensor_map;
    std::int64_t compute_count = 0;
};

}  // namespace galois::graph
