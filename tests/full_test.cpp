#include "tests/galois_test.hpp"

// 定义基准测试函数 BM_Full，benchmark::State &state 用于控制基准测试的运行
void BM_Full(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();

    auto ir_full_creator = op::FullCreator::Create(ir_input_type);

    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_full_creator, {ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto full_fun = jit_engine->EmitOperatorSymbol<float (*)(float *)>(ir_operator);

    float value = 5.0f;
    for (auto _ : state) {
        full_fun(&value);
    }
    // 统计基准测试处理的字节数和元素数量
    //  state.iterations() 记录了基准测试运行的次数。SetBytesProcessed(...) 计算 处理的总字节数。
    //  SetItemsProcessed(...) 计算 填充的总元素个数。
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}

BENCHMARK(BM_Full)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestFull) {
    int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();

    auto ir_full_creator = op::FullCreator::Create(ir_input_type);

    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_full_creator, {ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto full_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    float value = 5.0f;
    auto result = full_fun(&value);
    for (int i = 0; i < length; i++) {
        GALOIS_ASSERT(result[i] == 5.0);
    }
    delete[] result;
}
