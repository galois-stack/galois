#include "galois/op/sum.hpp"

#include "galois_test.hpp"

TEST(GaloisTests, TestSum) {
    auto ir_input_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_sum_creator = op::SumCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_sum_creator, {ir_input_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    auto input = new float[4];
    input[0] = 0.0f;
    input[1] = 1.0f;
    input[2] = 2.0f;
    input[3] = 3.0f;
    auto sum = *tmp_fun(input);
    fmt::print("sum : {}\n", sum);
}
