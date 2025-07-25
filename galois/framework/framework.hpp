#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "galois/ir/ir.hpp"
#include "galois/ir/ir_print_visitor.hpp"

namespace galois::framework {

// class Operation {};

/// @brief Manage the Operators
// class ComputingGraph {};

enum class NodeType { kCalculate, kInitialization};

struct GraphNode {
    std::string op_name;
    std::vector<std::string> tensor_input_name;
    std::string tensor_output_name;
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
    // 构造函数中初始化 computing_graph，避免空指针
    BuildComputingGraph() : computing_graph(std::make_unique<ComputingGraph>()) {
    }

   public:
    static std::shared_ptr<BuildComputingGraph> Create() {
        auto self = std::shared_ptr<BuildComputingGraph>(new BuildComputingGraph);
        return self;
    }

    std::unique_ptr<ComputingGraph> GetComputingGraph() {
        return std::move(computing_graph);
    }

    std::string Traverse(std::shared_ptr<ir::Tensor> tensor) {
        var_counter = 0;
        node_counter = 0;
        var_name_dict.clear();  // 清空变量名映射
        output.str("");         // 清空输出
        operator_level = 0;     // 重置层级计数器
        computing_graph->nodes.clear();  // 清空计算图
        computing_graph->edges.clear();
        tensor->ApplyVisitor(this->shared_from_this());
        return output.str();
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        bool has_return = false;
        std::shared_ptr<ir::Return> return_inst;
        std::string output_var;
        std::vector<std::string> input_vars;

        if (operator_level == 1) {
            if (ir_operator->block) {
                for (auto& tensor : *ir_operator->block) {
                    return_inst = std::dynamic_pointer_cast<ir::Return>(tensor);
                    if (return_inst) {
                        has_return = true;
                        break;
                    }
                }
            }

            for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
                input_vars.push_back(GetVariableName(ir_operator->inputs[i]));
            }

            GraphNode node;
            node.op_name = ir_operator->name;
            node.tensor_input_name = input_vars;
            node.tensor_output_name = has_return ? output_var : "";  // 无返回值则输出为空

            computing_graph->nodes[node_counter] = node;
            node_counter++;
        }

        if (operator_level == 0) {
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
        node.tensor_output_name = output_var;

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

    void Visit(std::shared_ptr<ir::view::Broadcast> ir_broadcast) override {
        std::string input_var = GetVariableName(ir_broadcast->Tensor());
        std::string output_var = GetVariableName(ir_broadcast);
        
        GraphNode node;
        node.op_name = "Broadcast";
        node.tensor_input_name = {input_var};  // 单个输入
        node.tensor_output_name = output_var;

        computing_graph->nodes[node_counter] = node;
        node_counter++;
    }

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
            
            std::cout << "  输出张量: " << node.tensor_output_name << std::endl;
            std::cout << "-------------------------" << std::endl;
        }
    }

   private:
    std::ostringstream output;
    int indent_level = 0;    // 缩进级别
    int var_counter = 0;     // 寄存器计数器
    int node_counter = 0;
    int operator_level = 0;  // 新增：operator层级计数器
    std::unordered_map<std::shared_ptr<ir::Tensor>, std::string> var_name_dict;
    std::unique_ptr<ComputingGraph> computing_graph;  // 计算图实例

    bool fusion_var_map_loaded;
    std::map<std::string, std::string> fusion_var_map;

    std::string TypeToString(std::shared_ptr<ir::TensorType> type) {
        if (!type) return "void";
        return type->name;
    }

    void LoadFusionVariableMap(std::string file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + file_path);
        }

        std::string content((std::istreambuf_iterator<char>(file)), 
                            std::istreambuf_iterator<char>());
        file.close();

        std::regex func_pattern(
            R"(void\s+Express(?:Inline)?\s*\([^)]*\)\s*\{([\s\S]*?)\})",
            std::regex::icase
        );
        std::smatch func_match;
        std::string func_body;

        if (std::regex_search(content, func_match, func_pattern)) {
            func_body = func_match[1].str();
        } else {
            fusion_var_map_loaded = true;
            return;
        }

        std::regex var_pattern(R"(auto\s+(\w+)\s*=\s*[^;]+;)");
        std::smatch var_match;
        std::string::const_iterator search_start(func_body.cbegin());

        var_counter = 0;
        fusion_var_map.clear();

        while (std::regex_search(search_start, func_body.cend(), var_match, var_pattern)) {
            if (var_match.size() > 1) {
                std::string original_var = var_match[1].str();
                std::string mapped_var = "%" + std::to_string(var_counter++);
                fusion_var_map[original_var] = mapped_var;
            }
            search_start = var_match.suffix().first;
        }

        fusion_var_map_loaded = true;
    }

    std::string GetVariableName(std::shared_ptr<ir::Tensor> tensor) {
        if (var_name_dict.find(tensor) != var_name_dict.end()) {
            return var_name_dict[tensor];
        }
        std::string name;
        if (!tensor->name.empty()) {
            return "%" + tensor->name;
        } else {
            name = "%" + std::to_string(var_counter++);
        }
        var_name_dict[tensor] = name;
        return name;
    }
};

}  // namespace galois::framework
