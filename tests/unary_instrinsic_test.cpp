#include "galois/op/unary_intrinsic.hpp"
#include "galois_test.hpp"

TEST(GaloisTests, TestUnaryIntrinsic) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_intrin_creator = op::UnaryInstrinsicCreator::Create("sin");
    auto ir_output_type = ir_intrin_creator->InferType({ir_input_type});
    auto ir_operator_type = ir::OperatorType::Create({ir_input_type}, ir_output_type);
    auto [ir_operator, operator_scope] = ir_builder->CreateOperator(ir_operator_type, "tmp");
    ir_intrin_creator->AffineExpress(ir_operator->inputs, ir_builder);

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *)>(prajna_compiler->GetSymbolValue("::tmp"));

    float input = 3.1415f / 2.0f;
    auto value = *tmp_fun(&input);
    fmt::print("sin(3.1415) = {}\n", value);
}
