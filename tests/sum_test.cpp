#include "galois/op/sum.hpp"

#include "tests/galois_test.hpp"

// 定义基准测试函数 BM_Sum::State &state 用于控制基准测试的运行
void BM_Sum(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SumCreator>({ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto sum_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    auto input = new float[length];

    for (auto _ : state) {
        sum_fun(input);
    }

    // state.iterations() 记录了基准测试运行的次数。SetBytesProcessed(...) 计算 处理的总字节数。
    // SetItemsProcessed(...) 计算 填充的总元素个数。
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}

BENCHMARK(BM_Sum)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestSum) {
    auto ir_input_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SumCreator>({ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto sum_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    auto input = new float[4];
    input[0] = 0.0f;
    input[1] = 1.0f;
    input[2] = 2.0f;
    input[3] = 3.0f;
    auto sum = *sum_fun(input);
    fmt::print("sum : {}\n", sum);

    // 释放动态数组
    delete[] input;
    sum_fun = nullptr;
}
