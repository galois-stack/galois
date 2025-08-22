#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "galois/ir/ir.hpp"
#include "galois/ir/ir_print_visitor.hpp"
#include "galois/op/op.hpp"

namespace galois::framework {

// class Operation {};

/// @brief Manage the Operators
// class ComputingGraph {};

enum class NodeType { kCalculate, kInitialization };

struct GraphNode {
    std::string op_name;
    std::vector<std::string> tensor_input_name;
    std::vector<std::string> tensor_output_name;
};

struct GraphEdge {
    std::string from_node_id;
    std::string to_node_id;
};

// 计算图
struct ComputingGraph {
    std::unordered_map<short, GraphNode> nodes;  // 存储算子节点
    std::vector<GraphEdge> edges;
};

class BuildComputingGraph : public ir::Visitor {
   protected:
    BuildComputingGraph() : computing_graph(std::make_unique<ComputingGraph>()) {}

   public:
    static std::shared_ptr<BuildComputingGraph> Create() {
        auto self = std::shared_ptr<BuildComputingGraph>(new BuildComputingGraph);
        return self;
    }

    std::unique_ptr<ComputingGraph> GetComputingGraph() { return std::move(computing_graph); }

    void Traverse(std::shared_ptr<ir::Tensor> tensor) {
        var_counter = 0;
        node_counter = 0;
        var_name_dict.clear();           // 清空变量名映射
        operator_level = 0;              // 重置层级计数器
        computing_graph->nodes.clear();  // 清空计算图
        computing_graph->edges.clear();

        tensor->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        std::vector<std::string> input_vars;
        std::vector<std::string> output_vars;

        GraphNode node;

        if (operator_level == 0) {
            for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
                output_vars.push_back(GetVariableName(ir_operator->inputs[i]));
            }
            
            node.op_name = ir_operator->name;
            node.tensor_input_name = {};
            node.tensor_output_name = output_vars;

            computing_graph->nodes[node_counter] = node;
            node_counter++;

            operator_level++;
            ir_operator->block->ApplyVisitor(this->shared_from_this());
        }
    }

    void Visit(std::shared_ptr<ir::Block> ir_block) override {
        for (auto& tensor : *ir_block) {
            tensor->ApplyVisitor(this->shared_from_this());
        }
    }

    void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {
        std::string output_var = GetVariableName(ir_alloca);

        GraphNode node;
        node.op_name = "Alloca";
        node.tensor_input_name = {};
        node.tensor_output_name = {output_var};

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::Free> ir_free) override {
        std::string input_var = GetVariableName(ir_free->Tensor());

        GraphNode node;
        node.op_name = "Free";
        node.tensor_input_name = {input_var};
        node.tensor_output_name = {};

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::Write> ir_write) override {
        std::string input_var = GetVariableName(ir_write->Tensor());
        std::string output_var = GetVariableName(ir_write->Variable());

        GraphNode node;
        node.op_name = "Write";
        node.tensor_input_name = {input_var};
        node.tensor_output_name = {output_var};

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::Return> ir_return) override {
        std::string input_var = GetVariableName(ir_return->Tensor());

        GraphNode node;
        node.op_name = "Return";
        node.tensor_input_name = {input_var};
        node.tensor_output_name = {};

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::Call> ir_call) override {
        std::vector<std::string> input_vars;
        
        GraphNode node;
        node.op_name = ir_call->Operator()->name;
        for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
            input_vars.push_back(GetVariableName(ir_call->Input(i)));
        }
        node.tensor_input_name = input_vars;
        node.tensor_output_name = {GetVariableName(ir_call)};

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::view::Broadcast> ir_broadcast) override {}

    void PrintAllNodes() {
        std::cout << "===== 计算图节点列表 =====" << std::endl;

        // 1. 收集所有节点ID并排序
        std::vector<int> node_ids;  // 假设node_id是int类型（根据你的node_counter类型调整）
        node_ids.reserve(computing_graph->nodes.size());
        for (const auto& [id, _] : computing_graph->nodes) {
            node_ids.push_back(id);
        }
        std::sort(node_ids.begin(), node_ids.end());  // 按ID升序排序

        // 2. 按排序后的ID遍历并打印节点
        for (int id : node_ids) {
            const auto& node = computing_graph->nodes.at(id);  // 按排序后的ID取节点
            std::cout << "节点 ID: " << id << std::endl;
            std::cout << "  算子名称: " << node.op_name << std::endl;

            std::cout << "  输入张量: [";
            for (size_t i = 0; i < node.tensor_input_name.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << node.tensor_input_name[i];
            }
            std::cout << "]" << std::endl;

            std::cout << "  输出张量: [";
            for (size_t i = 0; i < node.tensor_output_name.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << node.tensor_output_name[i];
            }
            std::cout << "]" << std::endl;
            
            std::cout << "-------------------------" << std::endl;
        }
    }

    std::shared_ptr<ir::Operator> OperatorFusionOpt(std::shared_ptr<ir::Operator> ir_operator) {
        ir_operator->block->clear();

        auto ir_builder = ir::Builder::Create();
        std::vector<std::string> operations = {"add", "sub"};

        auto sp_creator = op::OperatorFusionCreator::Create(operations);
        auto [ir_operator_fused, scope] = ir_builder->CreateOperator(
            ir_operator->GetOperatorType(), ir_operator->name + "_fused");
        std::vector<std::shared_ptr<ir::Tensor>> ir_inputs;
        std::transform(RANGE(ir_operator_fused->inputs), std::back_inserter(ir_inputs),
                       [](std::shared_ptr<ir::Tensor> ir_input) { return ir_input; });
        sp_creator->Express(ir_inputs, ir_builder);

        return ir_operator_fused;
    }

   private:
    int indent_level = 0;   // 缩进级别
    int var_counter = 0;
    int node_counter = 0;
    int operator_level = 0;  // 新增：operator层级计数器
    std::unordered_map<std::shared_ptr<ir::Tensor>, std::string> var_name_dict;
    std::unique_ptr<ComputingGraph> computing_graph;  // 计算图实例

    std::map<std::string, std::string> fusion_var_map;

    std::string GetVariableName(std::shared_ptr<ir::Tensor> tensor) {
        if (var_name_dict.find(tensor) != var_name_dict.end()) {
            std::string original_num = var_name_dict[tensor];
            return original_num;
        }

        std::string original_num;
        if (!tensor->name.empty()) {
            original_num = "%" + tensor->name;
        } else {
            original_num = "%" + std::to_string(var_counter++);
        }
        var_name_dict[tensor] = original_num;
        return original_num;
    }

};

}  // namespace galois::framework
