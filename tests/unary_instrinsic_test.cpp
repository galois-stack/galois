#include <cstdint>
#include <vector>

#include "galois/op/unary_intrinsic.hpp"
#include "tests/galois_test.hpp"

void BM_UnaryIntrinsic(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::UnaryInstrinsicCreator>({ir_input_type}, "sin");

    auto jit_engine = jit::Engine::Create();
    auto sin_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input_vec(length);
    for (auto _ : state) {
        sin_fun(input_vec.data());
    }

    // 设置基准测试数据
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * length);
}

BENCHMARK(BM_UnaryIntrinsic)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestUnaryIntrinsic) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::UnaryInstrinsicCreator>({ir_input_type}, "sin");

    auto jit_engine = jit::Engine::Create();
    auto sin_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    float input = 3.1415f / 2.0f;
    auto value = *sin_fun(&input);
    fmt::print("sin(3.1415) = {}\n", value);
}
