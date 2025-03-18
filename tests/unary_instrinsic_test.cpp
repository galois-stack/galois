#include <cstdint>
#include "galois/op/unary_intrinsic.hpp"
#include "tests/galois_test.hpp"
#include <vector> 


void BM_UnaryIntrinsic(benchmark::State &state) {

    auto length = int64_t(state.range(0));
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_intrin_creator = op::UnaryInstrinsicCreator::Create("sin");
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_intrin_creator, {ir_input_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    std::vector<float> input_vec(length); 
    for(auto _ : state) {
        tmp_fun(input_vec.data());
    }    
    
    // 设置基准测试数据
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * length);


    
}

BENCHMARK(BM_UnaryIntrinsic)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);


TEST(GaloisTests, TestUnaryIntrinsic) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_intrin_creator = op::UnaryInstrinsicCreator::Create("sin");
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_intrin_creator, {ir_input_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun = reinterpret_cast<float *(*)(float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_operator->fullname));

    float input = 3.1415f / 2.0f;
    auto value = *tmp_fun(&input);
    fmt::print("sin(3.1415) = {}\n", value);
}
