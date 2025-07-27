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

enum class NodeType { kCalculate, kInitialization };

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
    BuildComputingGraph() : computing_graph(std::make_unique<ComputingGraph>()) {}

   public:
    static std::shared_ptr<BuildComputingGraph> Create() {
        auto self = std::shared_ptr<BuildComputingGraph>(new BuildComputingGraph);
        return self;
    }

    std::unique_ptr<ComputingGraph> GetComputingGraph() { return std::move(computing_graph); }

    void SetFusionFile(const std::string& file_path) { fusion_file_path = file_path; }

    void Traverse(std::shared_ptr<ir::Tensor> tensor) {
        var_counter = 0;
        node_counter = 0;
        var_name_dict.clear();           // 清空变量名映射
        operator_level = 0;              // 重置层级计数器
        computing_graph->nodes.clear();  // 清空计算图
        computing_graph->edges.clear();

        // 关键：在遍历IR前加载变量映射
        if (!fusion_file_path.empty()) {
            LoadFusionVariableMap(fusion_file_path);
            PrintFusionVariableMap();
        }

        tensor->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<ir::Operator> ir_operator) override {
        bool has_return = false;
        bool has_fill = false;
        std::shared_ptr<ir::Return> return_inst;
        std::vector<std::string> input_vars;
        std::string output_vars;

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

            if (ir_operator->name.find("Fill") != std::string::npos) {
                input_vars.push_back(GetVariableName(ir_operator->inputs[1]));
                output_vars = GetVariableName(ir_operator->inputs[0]);
                has_fill = true;
            } else {
                for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
                    input_vars.push_back(GetVariableName(ir_operator->inputs[i]));
                }
            }

            GraphNode node;
            node.op_name = ir_operator->name;
            node.tensor_input_name = input_vars;
            node.tensor_output_name = has_return ? GetVariableNoneName(ir_operator)
                                      : has_fill ? output_vars
                                                 : std::string();
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
    int indent_level = 0;  // 缩进级别
    int var_counter = 0;
    int var_counter_2 = 0;
    int node_counter = 0;
    int operator_level = 0;  // 新增：operator层级计数器
    std::string fusion_file_path;
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
            R"(void\s+(Express(?:Inline)?)\s*\([^)]*\)\s*(override\s*)?\{([\s\S]*)\})",
            std::regex::icase);

        std::string::const_iterator search_start = content.cbegin();
        std::string all_funcs_str;
        std::smatch func_match;
        std::string func_name;
        std::string func_body;

        while (std::regex_search(search_start, content.cend(), func_match, func_pattern)) {
            all_funcs_str += func_match.str();
            search_start = func_match.suffix().first;
        }
        // std::cout << "完整匹配: " << all_funcs_str << std::endl << std::endl;

        if (search_start == content.cbegin()) {  // 从未匹配到任何结果
            fusion_var_map_loaded = true;
            return;
        }

        // 正则表达式匹配以下两种模式:
        // 拆分为三个子正则，分别匹配三种模式（便于针对性提取变量）
        // 1. Alloca模式：auto 左变量 = ir_builder->Alloca(...)
        std::regex alloca_pattern(R"(auto\s+(\w+)\s*=\s*ir_builder->Alloca\s*\([^;]+;)",
                                  std::regex::icase);

        // 2. ExpressCreator模式：auto 左变量 = ir_builder->ExpressCreator<...>({右变量列表})
        std::regex expr_creator_pattern(
            R"((?:auto\s+(\w+)\s*=\s*)?ir_builder->ExpressCreator<op::([^>]+)>\s*\(\s*\{([^}]*)\}\s*[^\)]*\);)",
            std::regex::icase);

        // 3. Broadcast模式: auto 左变量 = ir_builder->Create<...>(右变量, ...)
        std::regex broadcast_pattern(
            R"((\w+)\s*=\s*ir_builder->Create<ir::view::Broadcast>\s*\(\s*([\w\[\]]+)\s*,[^;]+;)",
            std::regex::icase);

        std::smatch var_match;
        std::string left_num;
        std::string right_num;
        search_start = all_funcs_str.cbegin();
        var_counter_2 = 0;
        fusion_var_map.clear();

        while (search_start != all_funcs_str.cend()) {
            // 存储所有可能的匹配结果
            std::vector<std::pair<std::smatch, std::regex*>> matches;

            // 尝试匹配所有 3 个模式
            std::smatch var_match_alloca, var_match_expr, var_match_broadcast;
            if (std::regex_search(search_start, all_funcs_str.cend(), var_match_alloca,
                                  alloca_pattern)) {
                // std::cout << "[Alloca] 匹配位置: " << var_match_alloca.position()
                //         << ", 内容: " << var_match_alloca.str() << std::endl;
                matches.emplace_back(var_match_alloca, &alloca_pattern);
            }
            if (std::regex_search(search_start, all_funcs_str.cend(), var_match_expr,
                                  expr_creator_pattern)) {
                // std::cout << "[ExpressCreator] 匹配位置: " << var_match_expr.position()
                //         << ", 内容: " << var_match_expr.str() << std::endl;
                matches.emplace_back(var_match_expr, &expr_creator_pattern);
            }
            if (std::regex_search(search_start, all_funcs_str.cend(), var_match_broadcast,
                                  broadcast_pattern)) {
                // std::cout << "[Broadcast] 匹配位置: " << var_match_broadcast.position()
                //         << ", 内容: " << var_match_broadcast.str() << std::endl;
                matches.emplace_back(var_match_broadcast, &broadcast_pattern);
            }

            // 如果没有匹配到任何模式，移动 search_start
            if (matches.empty()) {
                ++search_start;
                continue;
            }

            // 找到匹配位置最小的那个（即最早出现的匹配）
            auto best_match =
                std::min_element(matches.begin(), matches.end(), [&](const auto& a, const auto& b) {
                    return a.first.position() < b.first.position();
                });

            // 提取匹配结果
            const auto& var_match = best_match->first;
            const auto& pattern = best_match->second;
            // std::cout << ">>> 最终选择匹配: 位置=" << var_match.position()
            //           << ", 内容: " << var_match.str() << std::endl;

            // 根据匹配的模式类型进行处理
            if (pattern == &alloca_pattern) {
                // 模式1：Alloca（只提取左边变量）
                std::string left_var = var_match[1].str();
                left_num = "%" + std::to_string(var_counter_2++);
                fusion_var_map[left_num] = left_var;
            } else if (pattern == &expr_creator_pattern) {
                // 模式2：ExpressCreator（提取左边变量 + 右边变量）
                std::string left_var = var_match[1].str();
                std::string op_type_full = var_match[2].str();
                std::string op_type = op_type_full.substr(0, op_type_full.find("Creator"));

                std::string input_vars_str = var_match[3].str();
                std::vector<std::string> input_vars = split_vars(input_vars_str);

                // 根据算子类型过滤有效变量
                std::vector<std::string> valid_inputs;
                if (op_type == "Shape" || op_type == "Sum" || op_type == "ReduceProd" ||
                    op_type == "UnaryInstrinsic") {
                    if (!input_vars.empty() && IsValidVariable(input_vars[0])) {
                        valid_inputs.push_back(input_vars[0]);
                    }
                } else if (op_type == "Div" || op_type == "Mul" || op_type == "Add" ||
                           op_type == "Sub") {
                    for (size_t i = 0; i < input_vars.size() && i < 2; ++i) {
                        // if (IsValidVariable(input_vars[i])) {
                        //     valid_inputs.push_back(input_vars[i]);
                        // }
                        valid_inputs.push_back(input_vars[i]);
                    }

                } else if (op_type == "Fill") {
                    valid_inputs.push_back(input_vars[1]);
                    left_var = input_vars[0];
                }

                // 分配编号
                for (const auto& var : valid_inputs) {
                    right_num = "%" + std::to_string(var_counter_2++);
                    fusion_var_map[right_num] = var;
                }

                left_num = "%" + std::to_string(var_counter_2++);
                fusion_var_map[left_num] = left_var;

            } else if (pattern == &broadcast_pattern) {
                // 模式3：Broadcast（提取左边变量 + 右边变量）
                std::string left_var = var_match[1].str();
                std::string right_var = var_match[2].str();
                right_num = "%" + std::to_string(var_counter_2++);
                left_num = "%" + std::to_string(var_counter_2++);
                fusion_var_map[right_num] = right_var;
                fusion_var_map[left_num] = left_var;
            }

            search_start = var_match.suffix().first;
        }

        fusion_var_map_loaded = true;
    }

    // 分割变量列表的通用函数（支持任意数量变量）
    std::vector<std::string> split_vars(const std::string& vars_str) {
        std::vector<std::string> result;
        std::stringstream ss(vars_str);
        std::string var;

        while (std::getline(ss, var, ',')) {
            size_t start = var.find_first_not_of(" \t\n\r");
            size_t end = var.find_last_not_of(" \t\n\r");
            if (start == std::string::npos || end == std::string::npos) {
                continue;
            }
            result.push_back(var.substr(start, end - start + 1));
        }
        return result;
    }

    // 验证是否为有效变量（排除数字、字符串等字面量）
    bool IsValidVariable(const std::string& var) {
        if (var.empty()) return false;
        if (!isalpha(var[0]) && var[0] != '_') return false;
        return var.find_first_not_of(
                   "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[]->") ==
               std::string::npos;
    }

    void PrintFusionVariableMap() const {
        if (!fusion_var_map_loaded) {
            std::cout << "Fusion variable map is not loaded yet." << std::endl;
            return;
        }

        if (fusion_var_map.empty()) {
            std::cout << "Fusion variable map is empty." << std::endl;
            return;
        }

        std::cout << "Fusion Variable Mapping:" << std::endl;
        std::cout << "-------------------------" << std::endl;

        std::vector<std::pair<std::string, std::string>> sorted_pairs(fusion_var_map.begin(),
                                                                      fusion_var_map.end());

        auto comparator = [](const auto& a, const auto& b) {
            int num_a = std::stoi(a.first.substr(1));
            int num_b = std::stoi(b.first.substr(1));
            return num_a < num_b;
        };

        std::sort(sorted_pairs.begin(), sorted_pairs.end(), comparator);

        for (const auto& pair : sorted_pairs) {
            std::cout << "Original variable: " << pair.first
                      << " -> Mapped variable: " << pair.second << std::endl;
        }
    }

    std::string GetVariableName(std::shared_ptr<ir::Tensor> tensor) {
        if (var_name_dict.find(tensor) != var_name_dict.end()) {
            std::string original_num = var_name_dict[tensor];
            if (fusion_var_map_loaded) {
                auto it = fusion_var_map.find(original_num);
                if (it != fusion_var_map.end() && !it->second.empty()) {
                    return it->second;
                }
            }
            return original_num;
        }

        std::string original_num;
        if (!tensor->name.empty()) {
            original_num = "%" + tensor->name;
        } else {
            original_num = "%" + std::to_string(var_counter++);
        }
        var_name_dict[tensor] = original_num;

        if (fusion_var_map_loaded) {
            auto it = fusion_var_map.find(original_num);
            if (it != fusion_var_map.end() && !it->second.empty()) {
                return it->second;
            }
        }

        return original_num;
    }

    std::string GetVariableNoneName(std::shared_ptr<ir::Tensor> tensor) {
        std::string original_num = "%" + std::to_string(var_counter++);
        var_name_dict[tensor] = original_num;

        if (fusion_var_map_loaded) {
            auto it = fusion_var_map.find(original_num);
            if (it != fusion_var_map.end() && !it->second.empty()) {
                return it->second;
            }
        }

        return original_num;
    }
};

}  // namespace galois::framework
