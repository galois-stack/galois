#include "galois/op/copy.hpp"

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestCopy) {
    // 创建张量类型
    int64_t length = 1024000000;
    auto ir_input_type = ir::f32->Tile(length);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::CopyCreator>({ir_input_type, ir_input_type});
    auto jit_engine = jit::Engine::Create();
    auto copy_fun = jit_engine->EmitOperatorSymbol<void (*)(float *, float *)>(ir_operator);

    std::vector<float> vec_src(length);
    std::vector<float> vec_dst(length);

    auto t0 = std::chrono::steady_clock::now();
    copy_fun(vec_src.data(), vec_dst.data());
    auto t1 = std::chrono::steady_clock::now();

    auto bandwidth = double(length * 2 * sizeof(float)) / (t1 - t0).count();
    fmt::print("{} GB/sec \n", bandwidth);
}
