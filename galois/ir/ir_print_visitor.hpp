#pragma once

#include <memory>
#include <sstream>
#include <string>

#include "galois/ir/ir.hpp"

namespace galois::ir {

class IRPrinter : public ir::Visitor {
   protected:
    IRPrinter() = default;

   public:
    static std::shared_ptr<IRPrinter> Create() {
        auto self = std::shared_ptr<IRPrinter>(new IRPrinter);
        return self;
    }

    std::string Print(std::shared_ptr<Tensor> tensor) {
        var_counter = 0;
        var_name_dict.clear();  // 清空变量名映射
        output.str("");         // 清空输出
        tensor->ApplyVisitor(this->shared_from_this());
        return output.str();
    }

    void Dump(std::shared_ptr<Tensor> tensor) {
        auto str = this->Print(tensor);
        std::cout << str;
    }

    void Visit(std::shared_ptr<Operator> ir_operator) override {
        output << "operator " << ir_operator->name << "(";
        for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
            if (i > 0) {
                output << ", ";
            }
            output << ir_operator->inputs[i]->type->name << " "
                   << GetVariableName(ir_operator->inputs[i]);
        }

        output << ")->" << ir_operator->GetOperatorType()->output_type->name << "\n";
        ir_operator->block->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<Block> ir_block) override {
        this->Indent();
        output << "{\n";
        ++indent_level;
        for (auto& tensor : *ir_block) {
            this->Indent();
            tensor->ApplyVisitor(this->shared_from_this());
        }
        --indent_level;
        this->Indent();
        output << "}\n";
    }

    void Visit(std::shared_ptr<Alloca> ir_alloca) override {
        output << GetVariableName(ir_alloca) << " = Alloca " << ir_alloca->type->name << ";\n";
    }

    void Visit(std::shared_ptr<Free> ir_free) override {
        output << "Free " << GetVariableName(ir_free->Tensor()) << ";\n";
    }

    void Visit(std::shared_ptr<Grid> ir_grid) override {
        output << "grid [";
        for (int i = 0; i < ir_grid->shape.size(); ++i) {
            if (i > 0) {
                output << "x";
            }
            output << ir_grid->shape[i];
        }
        output << "]\n";

        ir_grid->block->ApplyVisitor(this->shared_from_this());
    }

    void Visit(std::shared_ptr<Accessor> ir_accessor) override {
        output << GetVariableName(ir_accessor) << " = Accessor "
               << GetVariableName(ir_accessor->Tensor()) << ";\n";
    }

    void Visit(std::shared_ptr<Indexing> ir_index) override {
        output << GetVariableName(ir_index) << " = Indexing " << GetVariableName(ir_index->Tensor())
               << ";\n";
    }

    void Visit(std::shared_ptr<Write> ir_write) override {
        output << "Write " << GetVariableName(ir_write->Tensor()) << ", "
               << GetVariableName(ir_write->Variable()) << ";\n";
    }

    void Visit(std::shared_ptr<Return> ir_return) override {
        output << "return " << GetVariableName(ir_return->Tensor()) << ";\n";
    }

    void Visit(std::shared_ptr<ArithmeticInstruction> ir_arith) override {
        std::string op;
        switch (ir_arith->operation) {
            case galois::ir::ArithmeticInstruction::Add:
                op = "Add";
                break;
            case ArithmeticInstruction::Sub:
                op = "Sub";
                break;
            case ArithmeticInstruction::Mul:
                op = "Mul";
                break;
            case ArithmeticInstruction::Div:
                op = "Div";
                break;
        }
        output << GetVariableName(ir_arith) << " = " << op << " "
               << GetVariableName(ir_arith->GetOperand(0)) << ", "
               << GetVariableName(ir_arith->GetOperand(1)) << ";\n";
    }

    void Visit(std::shared_ptr<CompareInstruction> ir_compare) override {
        std::string op;
        switch (ir_compare->operation) {
            case galois::ir::CompareInstruction::Equal:
                op = "Equal";
                break;
            case CompareInstruction::NotEqual:
                op = "NotEqual";
                break;
            case CompareInstruction::Less:
                op = "Less";
                break;
            case CompareInstruction::LessEqual:
                op = "LessEqual";
                break;
            case CompareInstruction::Greater:
                op = "Greater";
                break;
            case CompareInstruction::GreaterEqual:
                op = "GreaterEqual";
                break;
        }
        output << GetVariableName(ir_compare) << " = " << op << " "
               << GetVariableName(ir_compare->GetOperand(0)) << ", "
               << GetVariableName(ir_compare->GetOperand(1)) << ";\n";
    }

    void Visit(std::shared_ptr<SelectInstruction> ir_select) override {
        output << GetVariableName(ir_select) << " = Select "
               << GetVariableName(ir_select->Condition()) << ", "
               << GetVariableName(ir_select->TrueValue()) << ", "
               << GetVariableName(ir_select->FalseValue()) << ";\n";
    }

    void Visit(std::shared_ptr<ConstantFloat> ir_constant_float) override {
        output << GetVariableName(ir_constant_float) << " = "
               << TypeToString(ir_constant_float->type) << " " << ir_constant_float->value << ";\n";
    }

    void Visit(std::shared_ptr<ConstantInt> ir_constant_int) override {
        output << GetVariableName(ir_constant_int) << " = " << TypeToString(ir_constant_int->type)
               << " " << ir_constant_int->value << ";\n";
    }

    void Visit(std::shared_ptr<Call> ir_call) override {
        output << GetVariableName(ir_call) << " = Call @" << ir_call->Operator()->name << "(";
        for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
            if (i > 0) output << ", ";
            output << GetVariableName(ir_call->Input(i));
        }
        output << ");\n";
    }

    void Visit(std::shared_ptr<UnaryIntrinsic> ir_unary_intrinsic) override {
        output << GetVariableName(ir_unary_intrinsic) << " = " << ir_unary_intrinsic->intrinsic_name
               << " " << TypeToString(ir_unary_intrinsic->type) << " "
               << GetVariableName(ir_unary_intrinsic->Operand()) << ";\n";
    }

    void Visit(std::shared_ptr<Prefetch> ir_prefetch) override {
        output << "Prefetch " << GetVariableName(ir_prefetch->Address()) << ", " << ir_prefetch->rw
               << ", " << ir_prefetch->locality << ", " << ir_prefetch->cache_type << ";\n";
    }
    void Visit(std::shared_ptr<Tensor> ir_tensor) override {}
    void Visit(std::shared_ptr<Input> ir_input) override {}
    void Visit(std::shared_ptr<Instruction> ir_instruction) override {}

   private:
    std::ostringstream output;
    int indent_level = 0;  // 缩进级别
    int var_counter = 0;   // 寄存器计数器
    std::unordered_map<std::shared_ptr<Tensor>, std::string> var_name_dict;

    void Indent() {
        output << std::string(indent_level * 2, ' ');  // 每级缩进2个空格
    }
    std::string TypeToString(std::shared_ptr<TensorType> type) {
        if (!type) return "void";
        return type->name;
    }
    // 获取变量名
    std::string GetVariableName(std::shared_ptr<Tensor> tensor) {
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

}  // namespace galois::ir
