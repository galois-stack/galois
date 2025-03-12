#include "galois/op/fill.hpp"

#include "galois_test.hpp"

TEST(GaloisTests, TestFill) {

    // 1.创建张量类型
    auto ir_ts_type = ir::f32->Tile(4, 1)->Tile(2, 1);
   
   // 2.创建IR构建器和Fill算子，填充值为5
   auto ir_builder = ir::Builder::Create();

   auto ir_fill_op_creator = op::FillCreator::Create(ir_ts_type,5);

    // 3.推导类型
    auto ir_ts_type_c = ir_fill_op_creator->InferType({ir_ts_type});

    // 4.创建操作类型和操作符  
    auto ir_operator_type = ir::OperatorType::Create({ir_ts_type},ir_ts_type_c);
    auto [ir_operator, operator_scope] =
        ir_builder->CreateOperator(ir_operator_type, "fill_tensor");

    // 对张量进行填充
    ir_fill_op_creator->AffineExpress(ir_operator->inputs, ir_builder);
    // 释放operator_scope
    operator_scope = nullptr;  

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<void (*)(float *)>(
    prajna_compiler->GetSymbolValue("::fill_tensor"));

    auto input = new float[8];
   
    // 
    tmp_fun(input); 
    for(int i = 0; i < 8; i++) {
        GALOIS_ASSERT(input[i] == 5.0);
    }

    // 释放动态数组
    delete[] input;  
    tmp_fun = nullptr; 

}