#include "galois/op/fill.hpp"

#include "galois_test.hpp"

TEST(GaloisTests, TestFill) {
    // 1.创建张量类型
    auto ir_ts_type = ir::f32->Tile(4, 1)->Tile(2, 1);

    // 2.创建IR构建器和Fill算子，填充值为5
    auto ir_builder = ir::Builder::Create();

    // 4.创建操作类型和操作符
    auto ir_operator_type = ir::OperatorType::Create({ir_ts_type}, ir::VoidType::Create());
    auto [ir_operator, operator_scope] =
        ir_builder->CreateOperator(ir_operator_type, "fill_tensor");
    auto ir_value = ir_builder->GetConstant(ir::f32, 5.0);
    ir_builder->Express<op::FillCreator>({ir_operator->inputs[0], ir_value});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun =
        reinterpret_cast<void (*)(float *)>(prajna_compiler->GetSymbolValue("::fill_tensor"));

    auto input = new float[8];
    //
    tmp_fun(input);
    for (int i = 0; i < 8; i++) {
        GALOIS_ASSERT(input[i] == 5.0);
    }

    // 释放动态数组
    delete[] input;
    tmp_fun = nullptr;
}
