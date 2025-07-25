#pragma once

#include "galois/ir/ir.hpp"
#include "galois/ir/ir_print_visitor.hpp"

namespace galois::framework {

// class Operation {};

/// @brief Manage the Operators
// class ComputingGraph {};

// 计算图节点类型
enum class NodeType {
  kOperator,
  kTensor
};

// 计算图节点结构（增加变量名字段）
struct GraphNode {
    std::shared_ptr<ir::Operator> op;  // 算子
    std::vector<std::shared_ptr<ir::Tensor>> inputs;  // 输入张量
    std::shared_ptr<ir::Tensor> output;  // 输出张量
    std::unordered_map<std::shared_ptr<ir::Tensor>, std::string> tensor_names;  // 张量→变量名映射
};

// 计算图边
struct GraphEdge {
  std::string from_node_id;
  std::string to_node_id;
};

// 计算图
struct ComputingGraph {
  std::unordered_map<std::string, GraphNode> nodes;
  std::vector<GraphEdge> edges;
};

class VarNamer {
private:
    // 记录张量节点到变量索引的映射（基础映射）
    std::unordered_map<const ir::Tensor*, int> tensor_to_index_;
    // 记录变量名到张量节点的映射（用于检测重新赋值）
    std::unordered_map<std::string, const ir::Tensor*> varname_to_tensor_;
    int next_index_ = 0;

public:
    // 为张量生成变量名（需传入变量标识符，如"ir_pow"）
    std::string GetName(const std::string& var_identifier, const std::shared_ptr<ir::Tensor>& tensor) {
        const ir::Tensor* tensor_ptr = tensor.get();
        
        // 情况1：该变量名之前绑定过其他张量（重新赋值）
        if (varname_to_tensor_.count(var_identifier) && 
            varname_to_tensor_[var_identifier] != tensor_ptr) {
            // 旧张量解绑，新张量分配新索引
            tensor_to_index_.erase(varname_to_tensor_[var_identifier]);
            next_index_++; // 强制索引+1，确保重新赋值后索引递增
        }
        
        // 情况2：首次出现的张量，分配新索引
        if (!tensor_to_index_.count(tensor_ptr)) {
            tensor_to_index_[tensor_ptr] = next_index_++;
        }
        
        // 更新变量名与张量的绑定关系
        varname_to_tensor_[var_identifier] = tensor_ptr;
        
        // 生成变量名（如"mat0", "mat1"）
        return "mat" + std::to_string(tensor_to_index_[tensor_ptr]);
    }

    // 重置状态（用于新计算图的生成）
    void Reset() {
        tensor_to_index_.clear();
        varname_to_tensor_.clear();
        next_index_ = 0;
    }
};


// 访问者类（用于遍历IR节点并生成变量名）
class GraphBuilderVisitor : public ir::Visitor {
private:
    VarNamer var_namer_;
    ComputingGraph& graph_;  // 正在构建的计算图
    // 记录当前遍历的变量标识符（如"ir_pow"，需从IR节点元信息中获取）
    std::string current_var_identifier_;

public:
    GraphBuilderVisitor(ComputingGraph& graph) : graph_(graph) {}

    // 访问张量节点时生成变量名
    void Visit(ir::Tensor* tensor) override {
        if (!current_var_identifier_.empty()) {
            // 生成变量名并关联到计算图
            std::string var_name = var_namer_.GetName(
                current_var_identifier_, 
                std::shared_ptr<ir::Tensor>(tensor)  // 假设已有智能指针管理
            );
            // 将变量名存入当前图节点的tensor_names中
            graph_.current_node()->tensor_names[std::shared_ptr<ir::Tensor>(tensor)] = var_name;
        }
    }

    // 访问算子节点时，遍历其输入输出张量
    void Visit(ir::Operator* op) override {
        // 创建新的计算图节点
        auto node = std::make_unique<GraphNode>();
        node->op = std::shared_ptr<ir::Operator>(op);
        
        // 遍历输入张量
        for (size_t i = 0; i < op->inputs().size(); ++i) {
            auto input_tensor = op->inputs()[i];
            node->inputs.push_back(input_tensor);
            // 假设输入张量的变量标识符可通过某种方式获取（如op->input_names()[i]）
            current_var_identifier_ = op->input_names()[i];  // 例如"ir_inputs[0]"
            input_tensor->Accept(this);  // 触发张量的Visit，生成变量名
        }
        
        // 处理输出张量
        auto output_tensor = op->output();
        node->output = output_tensor;
        current_var_identifier_ = op->output_name();  // 例如"ir_pow"
        output_tensor->Accept(this);  // 生成输出张量的变量名
        
        // 将节点加入计算图
        graph_.nodes().push_back(std::move(node));
    }
};

// 构建计算图的入口函数
ComputingGraph BuildComputingGraph(std::shared_ptr<ir::Operator> root_op) {
    ComputingGraph graph;
    GraphBuilderVisitor visitor(graph);
    root_op->Accept(&visitor);  // 触发遍历，同时生成变量名
    return graph;
}

}  // namespace galois::framework
