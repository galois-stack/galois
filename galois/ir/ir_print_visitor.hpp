#include <memory>
#include <sstream>
#include <string>

#include "tensor.hpp"

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
        varCounter = 0;
        varNameMap.clear();  // 清空变量名映射
        output.str("");      // 清空输出
        tensor->ApplyVisitor(this->shared_from_this());
        return output.str();
    }

    void Visit(std::shared_ptr<Operator> ir_operator) override {
        output << "operator " << ir_operator->name << "(";
        for (size_t i = 0; i < ir_operator->inputs.size(); ++i) {
            if (i > 0) {
                output << ", ";
            }
            output << ir_operator->inputs[i]->type->name << " "
                   << getVarName(ir_operator->inputs[i]);
        }

        output << ")->" << ir_operator->GetOperatorType()->output_type->name << " {\n";

        indentLevel++;
        ir_operator->block->ApplyVisitor(this->shared_from_this());
        indentLevel--;

        indent();
        output << "}\n";
    }

    void Visit(std::shared_ptr<Block> ir_block) override {
        for (auto& tensor : *ir_block) {
            // 调试信息使用
            // output << "                        ; Visiting: " << tensor->tag << "\n";
            tensor->ApplyVisitor(this->shared_from_this());
        }
    }

    void Visit(std::shared_ptr<Alloca> ir_alloca) override {
        indent();
        output << getVarName(ir_alloca) << "  = Alloca " << ir_alloca->type->name << ";\n";
    }

    void Visit(std::shared_ptr<Grid> ir_grid) override {
        indent();
        output << "grid [";
        for (int i = 0; i < ir_grid->shape.size(); ++i) {
            if (i > 0) {
                output << "x";
            }
            output << ir_grid->shape[i];
        }
        output << "]{\n";

        indentLevel++;
        ir_grid->block->ApplyVisitor(this->shared_from_this());
        indentLevel--;

        indent();
        output << "}\n";
    }

    void Visit(std::shared_ptr<Accessor> ir_accessor) override {
        indent();
        output << getVarName(ir_accessor) << " = Accessor " << getVarName(ir_accessor->Tensor())
               << ";\n";
    }

    void Visit(std::shared_ptr<Write> ir_write) override {
        indent();
        output << "Write " << getVarName(ir_write->Tensor()) << ", "
               << getVarName(ir_write->Variable()) << ";\n";
    }

    void Visit(std::shared_ptr<Return> ir_return) override {
        indent();
        output << "return " << getVarName(ir_return->Tensor()) << ";\n";
    }

    void Visit(std::shared_ptr<ArithmeticInstruction> ir_arith) override {
        indent();

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
        output << getVarName(ir_arith) << " = " << op << " " << getVarName(ir_arith->GetOperand(0))
               << ", " << getVarName(ir_arith->GetOperand(1)) << ";\n";
    }

    void Visit(std::shared_ptr<ConstantFloat> ir_constant_float) override {
        indent();
        output << getVarName(ir_constant_float) << " = ConstantFloat "
               << TypeToString(ir_constant_float->type) << " ";

        // ToDo是否需要输出？？
        // if (ir_constant_float->special_value == ConstantFloat::SpecialValue::None) {
        //     output << (ir_constant_float->is_negative ? "-" : "") << ir_constant_float->value;
        // } else {
        //     switch (ir_constant_float->special_value) {
        //         case ConstantFloat::SpecialValue::Smallest:
        //             output << "smallest";
        //             break;
        //         case ConstantFloat::SpecialValue::Largest:
        //             output << "largest";
        //             break;
        //         case ConstantFloat::SpecialValue::NaN:
        //             output << "nan";
        //             break;
        //         case ConstantFloat::SpecialValue::Inf:
        //             output << (ir_constant_float->is_negative ? "-inf" : "inf");
        //             break;
        //         default:
        //             output << "unknown";
        //     }
        // }
        output << ";\n";
    }

    void Visit(std::shared_ptr<ConstantInt> ir_constant_int) override {
        indent();
        output << getVarName(ir_constant_int) << " = ConstantInt "
               << TypeToString(ir_constant_int->type) << " ";

        output << ";\n";
    }

    void Visit(std::shared_ptr<Call> ir_call) override {
        indent();
        output << getVarName(ir_call) << " = Call @" << ir_call->Operator()->name << "(";
        for (int64_t i = 0; i < ir_call->InputSize(); ++i) {
            if (i > 0) output << ", ";
            output << getVarName(ir_call->Input(i));
        }
        output << ");\n";
    }

    void Visit(std::shared_ptr<UnaryIntrinsic> ir_unary_intrinsic) override {
        indent();
        output << getVarName(ir_unary_intrinsic) << " = " << ir_unary_intrinsic->intrinsic_name
               << " " << TypeToString(ir_unary_intrinsic->type) << " "
               << getVarName(ir_unary_intrinsic->Operand()) << ";\n";
    }

    void Visit(std::shared_ptr<Tensor> ir_tensor) override {}
    void Visit(std::shared_ptr<Input> ir_input) override {}
    void Visit(std::shared_ptr<Instruction> ir_instruction) override {}

   private:
    std::ostringstream output;
    int indentLevel = 0;  // 缩进级别
    int varCounter = 0;   // 寄存器计数器
    std::unordered_map<std::shared_ptr<Tensor>, std::string> varNameMap;

    void indent() {
        output << std::string(indentLevel * 2, ' ');  // 每级缩进2个空格
    }
    std::string TypeToString(std::shared_ptr<TensorType> type) {
        if (!type) return "void";
        return type->name;
    }
    // 获取变量名
    std::string getVarName(std::shared_ptr<Tensor> tensor) {
        if (varNameMap.find(tensor) != varNameMap.end()) {
            return varNameMap[tensor];
        }
        std::string name;
        if (!tensor->name.empty()) {
            return "%" + tensor->name;
        } else {
            name = "%" + std::to_string(varCounter++);
        }
        varNameMap[tensor] = name;
        return name;
    }
};

}  // namespace galois::ir