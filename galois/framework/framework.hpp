#pragma once

#include "galois/ir/ir.hpp"

namespace galois::framework {

class Operation {};

/// @brief Manage the Operators
// class ComputingGraph {};

// 计算图节点类型
enum class NodeType {
  kOperator,
  kTensor
};

// 计算图节点
struct GraphNode {
  std::string id;
  NodeType type;
  std::shared_ptr<galois::ir::Operator> op;
  std::shared_ptr<galois::ir::Tensor> tensor;
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

template <typename T>
std::string GetNodeId(std::shared_ptr<T> node);

template <>
inline std::string GetNodeId<ir::Operator>(std::shared_ptr<ir::Operator> op) {
    return "op:" + op->fullname;
}

template <>
inline std::string GetNodeId<ir::Tensor>(std::shared_ptr<ir::Tensor> tensor) {
    return "tensor:" +
           std::to_string(reinterpret_cast<uintptr_t>(tensor.get()));  // 张量ID：tensor:指针地址
}

template <>
inline std::string GetNodeId<ir::Input>(std::shared_ptr<ir::Input> tensor) {
    return "tensor:" +
           std::to_string(reinterpret_cast<uintptr_t>(tensor.get()));  // 张量ID：tensor:指针地址
}

// 向计算图中添加算子及其依赖
inline void AddOperatorToGraph(std::shared_ptr<ir::Operator> ir_op, ComputingGraph& graph) {
    std::string op_node_id = GetNodeId(ir_op);
    graph.nodes[op_node_id] = {op_node_id, NodeType::kOperator, ir_op, nullptr};

    for (auto& input_tensor : ir_op->inputs) {
        std::string tensor_id = GetNodeId(input_tensor);
        if (!graph.nodes.count(tensor_id)) {
            graph.nodes[tensor_id] = {tensor_id, NodeType::kTensor, nullptr, input_tensor};
        }
        graph.edges.push_back({tensor_id, op_node_id});
    }

    std::shared_ptr<ir::Tensor> output_tensor = nullptr;
    for (auto& tensor : *ir_op->block) {
        if (auto ret = std::dynamic_pointer_cast<ir::Return>(tensor)) {
            output_tensor = ret;
            break;
        }
    }
    if (output_tensor) {
        std::string output_tensor_id = GetNodeId(output_tensor);
        if (!graph.nodes.count(output_tensor_id)) {
            graph.nodes[output_tensor_id] = {output_tensor_id, NodeType::kTensor, nullptr,
                                             output_tensor};
        }
        graph.edges.push_back({op_node_id, output_tensor_id});
    }
}

inline ComputingGraph BuildComputingGraph(std::shared_ptr<ir::Operator> root_op) {
    ComputingGraph graph;
    std::queue<std::shared_ptr<ir::Operator>> op_queue;
    std::unordered_map<std::string, bool> visited;

    op_queue.push(root_op);
    visited[GetNodeId(root_op)] = true;

    while (!op_queue.empty()) {
        auto current_op = op_queue.front();
        op_queue.pop();

        AddOperatorToGraph(current_op, graph);

        for (auto& tensor : *current_op->block) {
            if (auto call = std::dynamic_pointer_cast<ir::Call>(tensor)) {
                auto sub_op = call->Operator();
                std::string sub_op_id = GetNodeId(sub_op);
                if (!visited.count(sub_op_id)) {
                    visited[sub_op_id] = true;
                    op_queue.push(sub_op);
                }
            }
        }
    }

    return graph;
}

inline std::string NodeTypeToString(NodeType type) {
    return type == NodeType::kOperator ? "Operator" : "Tensor";
}

inline void PrintComputingGraph(const ComputingGraph& graph) {
    std::cout << "=== Computing Graph Info ===" << std::endl;

    std::cout << "\n[Nodes] (" << graph.nodes.size() << " nodes):" << std::endl;
    for (const auto& [node_id, node] : graph.nodes) {
        std::cout << "  ID: " << std::setw(20) << node_id 
                  << " | Type: " << std::setw(8) << NodeTypeToString(node.type);

        if (node.type == NodeType::kOperator && node.op) {
            std::cout << " | Op: " << node.op->fullname;
        } else if (node.type == NodeType::kTensor && node.tensor) {
            std::cout << " | Tensor addr: " << node.tensor.get();
        }
        std::cout << std::endl;
    }

    std::cout << "\n[Edges] (" << graph.edges.size() << " edges):" << std::endl;
    for (size_t i = 0; i < graph.edges.size(); ++i) {
        const auto& edge = graph.edges[i];
        std::cout << "  Edge " << i << ": " 
                  << edge.from_node_id << " -> " << edge.to_node_id << std::endl;
    }

    std::cout << "\n===========================" << std::endl;
}

}  // namespace galois::framework
