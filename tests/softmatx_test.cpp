#include "galois/op/softmax.hpp"
#include "galois_test.hpp"

TEST(GaloisTests, TestSoftmax) {
    auto ir_vec_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_softmax_creator = op::SoftmaxCreator::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_softmax_creator, {ir_vec_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    auto input = std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f};
    auto value = tmp_fun(input.data());

    for (int i = 0; i < 4; ++i) {
        fmt::print("{}, ", value[i]);
    }
    fmt::print("\n");
}
