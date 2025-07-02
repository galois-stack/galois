#include "galois/op/softmax.hpp"
#include "tests/galois_test.hpp"

void BM_Softmax(benchmark::State &state) {
    auto length = int64_t(state.range(0));
    auto ir_vec_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SoftmaxCreator>({ir_vec_type});

    auto jit_engine = jit::Engine::Create();
    auto softmax_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    auto input = std::vector<float>(length, 1.0f);

    for (auto _ : state) {
        softmax_fun(input.data());
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * length * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * length);
}
BENCHMARK(BM_Softmax)->Arg(1 << 10)->Arg(1 << 20)->Arg(1 << 30);

TEST(GaloisTests, TestSoftmax) {
    auto ir_vec_type = ir::f32->Tile(4);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SoftmaxCreator>({ir_vec_type});

    auto jit_engine = jit::Engine::Create();
    auto softmax_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    auto input = std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f};
    auto value = softmax_fun(input.data());

    for (int i = 0; i < 4; ++i) {
        fmt::print("{}, ", value[i]);
    }
    fmt::print("\n");
}

TEST(GaloisTests, TestSoftmax_bandwidth) {
    int64_t length = 102400000;
    auto ir_vec_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SoftmaxCreator>({ir_vec_type});

    auto jit_engine = jit::Engine::Create();
    auto softmax_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input(length);
    auto t0 = std::chrono::steady_clock::now();
    auto value = softmax_fun(input.data());
    auto t1 = std::chrono::steady_clock::now();
    
    auto bandwidth = double(length * 2 * sizeof(float)) / (t1 - t0).count();
    fmt::print("{} GB/sec \n", bandwidth);
}

