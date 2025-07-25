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

// 计算图节点类型
enum class NodeType { kOperator, kTensor };

// 计算图节点结构（简化，只保留算子信息）
struct GraphNode {
    std::shared_ptr<ir::Operator> op;  // 算子
};

struct GraphEdge {
    std::string from_node_id;
    std::string to_node_id;
};

// 计算图
struct ComputingGraph {
    std::unordered_map<std::string, GraphNode> nodes;  // 存储算子节点
    std::vector<GraphEdge> edges;
};

// class BuildComputingGraph : public ir::Visitor {
//    protected:
//     BuildComputingGraph() = default;

//    public:
//     static std::shared_ptr<BuildComputingGraph> Create() {
//         auto self = std::shared_ptr<BuildComputingGraph>(new BuildComputingGraph);
//         return self;
//     }

//     std::string Print(std::shared_ptr<ir::Tensor> tensor) {
//         var_counter = 0;
//         var_name_dict.clear();  // 清空变量名映射
//         output.str("");         // 清空输出
//         tensor->ApplyVisitor(this->shared_from_this());
//         return output.str();
//     }

//     void Dump(std::shared_ptr<ir::Tensor> tensor) {
//         auto str = this->Print(tensor);
//         std::cout << str;
//     }

//     void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
//         output << "operator " << ir_operator->name << "(";
//         for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
//             if (i > 0) {
//                 output << ", ";
//             }
//             output << ir_operator->inputs[i]->type->name << " "
//                    << GetVariableName(ir_operator->inputs[i]);
//         }

//         output << ")->" << ir_operator->GetOperatorType()->output_type->name << "\n";
//         ir_operator->block->ApplyVisitor(this->shared_from_this());
//     }

//     void Visit(std::shared_ptr<ir::Block> ir_block) override {
//         this->Indent();
//         output << "{\n";
//         ++indent_level;
//         for (auto& tensor : *ir_block) {
//             this->Indent();
//             tensor->ApplyVisitor(this->shared_from_this());
//         }
//         --indent_level;
//         this->Indent();
//         output << "}\n";
//     }

//     void Visit(std::shared_ptr<ir::Alloca> ir_alloca) override {
//         output << GetVariableName(ir_alloca) << " = Alloca " << ir_alloca->type->name << ";\n";
//     }

//     void Visit(std::shared_ptr<ir::ArithmeticInstruction> ir_arith) override {
//         std::string op;
//         switch (ir_arith->operation) {
//             case galois::ir::ArithmeticInstruction::Add:
//                 op = "Add";
//                 break;
//             case ir::ArithmeticInstruction::Sub:
//                 op = "Sub";
//                 break;
//             case ir::ArithmeticInstruction::Mul:
//                 op = "Mul";
//                 break;
//             case ir::ArithmeticInstruction::Div:
//                 op = "Div";
//                 break;
//         }
//         output << GetVariableName(ir_arith) << " = " << op << " "
//                << GetVariableName(ir_arith->GetOperand(0)) << ", "
//                << GetVariableName(ir_arith->GetOperand(1)) << ";\n";
//     }

//     void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {
//         output << GetVariableName(ir_unary_intrinsic) << " = " <<
//         ir_unary_intrinsic->intrinsic_name
//                << " " << TypeToString(ir_unary_intrinsic->type) << " "
//                << GetVariableName(ir_unary_intrinsic->Operand()) << ";\n";
//     }

//     void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override {
//         output << "Prefetch " << GetVariableName(ir_prefetch->Address()) << ", " <<
//         ir_prefetch->rw
//                << ", " << ir_prefetch->locality << ", " << ir_prefetch->cache_type << ";\n";
//     }

//    private:
//     std::ostringstream output;
//     int indent_level = 0;  // 缩进级别
//     int var_counter = 0;   // 寄存器计数器
//     std::unordered_map<std::shared_ptr<ir::Tensor>, std::string> var_name_dict;

//     void Indent() {
//         output << std::string(indent_level * 2, ' ');  // 每级缩进2个空格
//     }
//     std::string TypeToString(std::shared_ptr<ir::TensorType> type) {
//         if (!type) return "void";
//         return type->name;
//     }
//     // 获取变量名
//     std::string GetVariableName(std::shared_ptr<ir::Tensor> tensor) {
//         if (var_name_dict.find(tensor) != var_name_dict.end()) {
//             return var_name_dict[tensor];
//         }
//         std::string name;
//         if (!tensor->name.empty()) {
//             return "%" + tensor->name;
//         } else {
//             name = "%" + std::to_string(var_counter++);
//         }
//         var_name_dict[tensor] = name;
//         return name;
//     }
// };

class BuildComputingGraph : public ir::Visitor {
   protected:
    BuildComputingGraph() = default;

   public:
    static std::shared_ptr<BuildComputingGraph> Create() {
        auto self = std::shared_ptr<BuildComputingGraph>(new BuildComputingGraph);
        return self;
    }

    std::string Print(std::shared_ptr<ir::Tensor> tensor) {
        var_counter = 0;
        var_name_dict.clear();  // 清空变量名映射
        output.str("");         // 清空输出
        operator_level = 0;     // 重置层级计数器
        tensor->ApplyVisitor(this->shared_from_this());
        return output.str();
    }

    void Dump(std::shared_ptr<ir::Tensor> tensor) {
        auto str = this->Print(tensor);
        std::cout << str;
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        bool has_return = false;
        std::shared_ptr<ir::Return> return_inst;

        if (ir_operator->block) {
            for (auto& tensor : *ir_operator->block) {
                return_inst = std::dynamic_pointer_cast<ir::Return>(tensor);
                if (return_inst) {
                    has_return = true;
                    break;
                }
            }
        }

        if (!has_return) {
            output << "Level " << operator_level << " operator " << ir_operator->name << "(";
        } else {
            output << "Level " << operator_level << " " << GetVariableName(return_inst) << " = "
                   << ir_operator->name << "(";
        }

        for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
            if (i > 0) {
                output << ", ";
            }
            output << GetVariableName(ir_operator->inputs[i]);
        }

        output << ")" << "\n";

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
        output << "Level " << operator_level << " " << GetVariableName(ir_alloca) << " = Alloca "
               << ";\n";
    }

    void Visit(std::shared_ptr<ir::view::Broadcast> ir_broadcast) override {
        output << "Level " << operator_level << " " << GetVariableName(ir_broadcast)
               << " = Broadcast " << GetVariableName(ir_broadcast->Tensor()) << ";\n";
    }

    void Visit(std::shared_ptr<ir::UnaryIntrinsic> ir_unary_intrinsic) override {
        output << GetVariableName(ir_unary_intrinsic) << " = " << ir_unary_intrinsic->intrinsic_name
               << " " << TypeToString(ir_unary_intrinsic->type) << " "
               << GetVariableName(ir_unary_intrinsic->Operand()) << ";\n";
    }

    void Visit(std::shared_ptr<ir::Prefetch> ir_prefetch) override {
        output << "Prefetch " << GetVariableName(ir_prefetch->Address()) << ", " << ir_prefetch->rw
               << ", " << ir_prefetch->locality << ", " << ir_prefetch->cache_type << ";\n";
    }

   private:
    std::ostringstream output;
    int indent_level = 0;    // 缩进级别
    int var_counter = 0;     // 寄存器计数器
    int operator_level = 0;  // 新增：operator层级计数器
    std::unordered_map<std::shared_ptr<ir::Tensor>, std::string> var_name_dict;

    void Indent() {
        output << std::string(indent_level * 2, ' ');  // 每级缩进2个空格
    }

    std::string TypeToString(std::shared_ptr<ir::TensorType> type) {
        if (!type) return "void";
        return type->name;
    }

    // 获取变量名
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
