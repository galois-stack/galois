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

    void SetFusionFile(const std::string& file_path) {
        fusion_file_path = file_path;
    }

    void Traverse(std::shared_ptr<ir::Tensor> tensor) {
        var_counter = 0;
        node_counter = 0;
        var_name_dict.clear();  // 清空变量名映射
        operator_level = 0;     // 重置层级计数器
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
    int indent_level = 0;    // 缩进级别
    int var_counter = 0;     // 寄存器计数器
    int var_counter_2 = 0;     // 寄存器计数器
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
            std::regex::icase
        );

        std::string::const_iterator search_start = content.cbegin();
        std::string all_funcs_str;
        std::smatch func_match;
        std::string func_name;
        std::string func_body;

        while (std::regex_search(search_start, content.cend(), func_match, func_pattern)) {
            all_funcs_str += func_match.str();
            search_start = func_match.suffix().first;
        }
        std::cout << "完整匹配: " << all_funcs_str << std::endl << std::endl;

        if (search_start == content.cbegin()) {  // 从未匹配到任何结果
            fusion_var_map_loaded = true;
            return;
        }

        // 正则表达式匹配以下两种模式:
        // 拆分为三个子正则，分别匹配三种模式（便于针对性提取变量）
        // 1. Alloca模式：auto 左变量 = ir_builder->Alloca(...)
        std::regex alloca_pattern(
            R"(auto\s+(\w+)\s*=\s*ir_builder->Alloca\s*\([^;]+;)",
            std::regex::icase
        );

        // 2. ExpressCreator模式：auto 左变量 = ir_builder->ExpressCreator<...>({右变量列表})
        std::regex expr_creator_pattern(
            R"(ir_builder->ExpressCreator<op::([^>]+)>\s*\(\s*\{([^}]*)\}\s*[^\)]*\);)",
            std::regex::icase
        );

        // 3. Broadcast模式：auto 左变量 = ir_builder->Create<...>(右变量, ...)
        std::regex broadcast_pattern(
            R"((\w+)\s*=\s*[\s\S]*?ir_builder->Create<ir::view::Broadcast>\s*\(\s*([\w\[\]]+)\s*,)",
            std::regex::icase
        );
        
        std::smatch var_match;
        std::string left_num;
        std::string right_num;
        search_start = all_funcs_str.cbegin();
        var_counter_2 = 0;
        fusion_var_map.clear();

        while (search_start != all_funcs_str.cend()) {
            // 模式1：匹配Alloca（只提取左边变量）
            if (std::regex_search(search_start, all_funcs_str.cend(), var_match, alloca_pattern)) {
                std::cout << "var_match: " << var_match.str() << std::endl;
                std::cout << "search_start 位置: " << (search_start - all_funcs_str.cbegin()) << std::endl;
                if (var_match.size() ) {
                    std::string left_var = var_match[1].str(); // 等号左边变量
                    left_num = "%" + std::to_string(var_counter_2++);
                    fusion_var_map[left_num] = left_var;
                    search_start = var_match.suffix().first;
                    std::cout << "search_start 位置: " << (search_start - all_funcs_str.cbegin()) << std::endl;
                    continue; // 处理完后直接进入下一次循环，重新检查三种模式
                }
            }

            // 模式2：匹配ExpressCreator（提取左边变量 + 右边所有变量）
            else if (std::regex_search(search_start, all_funcs_str.cend(), var_match, expr_creator_pattern)) {
                std::cout << "var_match: " << var_match.str() << std::endl;
                std::cout << "search_start 位置: " << (search_start - all_funcs_str.cbegin()) << std::endl;
                if (var_match.size()) {
                    std::string op_type_full = var_match[1].str();
                    std::string op_type = op_type_full;
                    size_t creator_pos = op_type.find("Creator");
                    if (creator_pos != std::string::npos) {
                        op_type = op_type.substr(0, creator_pos); // 简化为Shape、Fill等
                    }
                    std::cout << "op_type_full: " << op_type_full << std::endl;
                    std::cout << "op_type: " << op_type << std::endl;

                    std::string input_vars_str = var_match[2].str();
                    std::vector<std::string> input_vars = split_vars(input_vars_str); // 分割变量

                    std::vector<std::string> valid_inputs;
                    if (op_type == "Shape" || op_type == "Sum" || op_type == "ReduceProd" || op_type == "UnaryInstrinsic") {
                        // 1个输入算子
                        if (!input_vars.empty() && IsValidVariable(input_vars[0])) {
                            valid_inputs.push_back(input_vars[0]);
                        }
                    } else if (op_type == "Fill" || op_type == "Div" || op_type == "Mul" || op_type == "Add" || op_type == "Sub") {
                        // 2个输入算子
                        for (size_t i = 0; i < input_vars.size() && i < 2; ++i) {
                            if (IsValidVariable(input_vars[i])) {
                                valid_inputs.push_back(input_vars[i]);
                            }
                        }
                    }

                    for (const auto& var : valid_inputs) {
                        right_num = "%" + std::to_string(var_counter_2++);
                        fusion_var_map[right_num] = var; 
                    }

                    search_start = var_match.suffix().first;
                    // // 调试：输出下一次搜索的起始位置内容（前50个字符）
                    // std::string remaining(all_funcs_str.begin() + (search_start - all_funcs_str.cbegin()), 
                    //                     all_funcs_str.begin() + (search_start - all_funcs_str.cbegin()) + 50);
                    // std::cout << "下一次搜索起始内容: " << remaining << std::endl;
                    std::cout << "search_start 位置: " << (search_start - all_funcs_str.cbegin()) << std::endl;
                    continue; // 处理完后直接进入下一次循环，重新检查三种模式
                }
            }

            // 模式3：匹配Broadcast（提取左边变量 + 右边变量）
            else if (std::regex_search(search_start, all_funcs_str.cend(), var_match, broadcast_pattern)) {
                std::cout << "var_match: " << var_match.str() << std::endl;
                if (var_match.size()) {
                    std::string left_var = var_match[1].str(); // 等号左边变量
                    std::string right_var = var_match[2].str(); // 等号右边变量
                    left_num = "%" + std::to_string(var_counter_2++);
                    right_num = "%" + std::to_string(var_counter_2++);
                    fusion_var_map[left_num] = left_var;
                    fusion_var_map[right_num] = right_var;
                    search_start = var_match.suffix().first;
                    continue; // 处理完后直接进入下一次循环，重新检查三种模式
                }
            }

            // 若均不匹配，移动搜索位置（避免死循环）
            else{
                ++search_start;
            }
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
        // 变量名规则：以字母或下划线开头，可包含字母、数字、下划线、[]、->
        if (var.empty()) return false;
        if (!isalpha(var[0]) && var[0] != '_') return false; // 首字符必须是字母或下划线
        // 后续字符允许字母、数字、下划线、[]、->
        return var.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[]->") == std::string::npos;
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
        
        // 遍历映射并打印每个键值对
        for (const auto& pair : fusion_var_map) {
            std::cout << "Original variable: " << pair.first 
                    << " -> Mapped variable: " << pair.second << std::endl;
        }
    }

    std::string GetVariableName(std::shared_ptr<ir::Tensor> tensor) {
        if (var_name_dict.find(tensor) != var_name_dict.end()) {
            return var_name_dict[tensor];
        }

        std::string name;

        if (fusion_var_map_loaded && !tensor->name.empty()) {
            auto it = fusion_var_map.find(tensor->name);
            if (it != fusion_var_map.end()) {
                name = it->second;
                var_name_dict[tensor] = name;
                return name;
            }
        }

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
