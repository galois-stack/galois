#include "galois/op/fill.hpp"

#include "tests/galois_test.hpp"

//定义基准测试函数 BM_Fill，benchmark::State &state 用于控制基准测试的运行
void BM_Fill(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_vec_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();

    auto ir_fill_creator = op::FillCreator::Create();

    auto ir_operator_type = ir::OperatorType::Create({ir_vec_type}, ir::VoidType::Create());
    auto [ir_operator, operator_scope] =
        ir_builder->CreateOperator(ir_operator_type, ir_fill_creator->fullname);
    auto ir_value = ir_builder->GetConstant(ir::f32, 5.0);
    ir_builder->Express<op::FillCreator>({ir_operator->inputs[0], ir_value});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen = std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    //获取编译后的 Fill 函数地址，转换为函数指针 tmp_fun。
    auto tmp_fun = reinterpret_cast<void(*)(float *)>(prajna_compiler->GetSymbolValue("::" + ir_fill_creator->fullname));

    auto input = new float[length];
    for (auto _ : state) {
        //tmp_fun(input) 执行填充操作，将 input 数组的所有元素设置为 5.0。
        tmp_fun(input);
    }
    //统计基准测试处理的字节数和元素数量
    // state.iterations() 记录了基准测试运行的次数。SetBytesProcessed(...) 计算 处理的总字节数。 SetItemsProcessed(...) 计算 填充的总元素个数。
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
     // 释放动态数组
     delete[] input;
     tmp_fun = nullptr;
}

BENCHMARK(BM_Fill)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);


TEST(GaloisTests, TestFill) {
    // 创建张量类型
    auto ir_ts_type = ir::f32->Tile(4, 1)->Tile(2, 1);

    // 创建IR构建器和Fill算子，填充值为5
    auto ir_builder = ir::Builder::Create();

    auto ir_fill_creator = op::FillCreator::Create();

    // 创建操作类型和操作符
    auto ir_operator_type = ir::OperatorType::Create({ir_ts_type}, ir::VoidType::Create());
    auto [ir_operator, operator_scope] =
        ir_builder->CreateOperator(ir_operator_type, ir_fill_creator->fullname);
    auto ir_value = ir_builder->GetConstant(ir::f32, 5.0);
    ir_builder->Express<op::FillCreator>({ir_operator->inputs[0], ir_value});

    auto prajna_compiler = CreateCompiler();
    auto llvm_codegen =
        std::make_shared<codegen::cpu::PrajnaCodegen>(prajna_compiler->_symbol_table);
    llvm_codegen->EmitOperatorFunction(ir_operator);
    prajna_compiler->GenLlvm(llvm_codegen->pir_builder->module);
    auto tmp_fun =
        reinterpret_cast<void (*)(float *)>(prajna_compiler->GetSymbolValue("::" + ir_fill_creator->fullname));

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
