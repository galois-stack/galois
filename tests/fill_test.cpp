#include "galois/op/fill.hpp"

#include "tests/galois_test.hpp"

// 定义基准测试函数 BM_Fill，benchmark::State &state 用于控制基准测试的运行
void BM_Fill(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();
    auto ir_fill_creator = op::FillCreator::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator(ir_fill_creator, {ir_input_type, ir_value_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    // 获取编译后的 Fill 函数地址，转换为函数指针 fill_fun。
    auto fill_fun = reinterpret_cast<void (*)(float *, float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_fill_creator->fullname));

    std::vector<float> input_vec(length);
    float value = 1.0f;
    for (auto _ : state) {
        fill_fun(input_vec.data(), &value);
    }
    // 统计基准测试处理的字节数和元素数量
    //  state.iterations() 记录了基准测试运行的次数。SetBytesProcessed(...) 计算 处理的总字节数。
    //  SetItemsProcessed(...) 计算 填充的总元素个数。
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}

BENCHMARK(BM_Fill)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestFill) {
    // 创建张量类型
    int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();
    auto ir_fill_creator = op::FillCreator::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator(ir_fill_creator, {ir_input_type, ir_value_type});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto fill_fun = reinterpret_cast<void (*)(float *, float *)>(
        prajna_compiler->GetSymbolValue("::" + ir_fill_creator->fullname));

    std::vector<float> input_vec(length);
    float value = 5.0f;
    fill_fun(input_vec.data(), &value);
    for (int i = 0; i < length; i++) {
        GALOIS_ASSERT(input_vec[i] == value);
    }
}
