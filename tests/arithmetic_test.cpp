#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestAdd) {
    // 创建张量类型
    int64_t length = 102400000;
    auto ir_input_type = ir::f32->Tile(length);
    auto ir_value_type = ir::f32;

    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::AddCreator>({ir_input_type, ir_input_type});
    auto jit_engine = jit::Engine::Create();
    auto add_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    std::vector<float> input_vec0(length);
    std::vector<float> input_vec1(length);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto p_re = add_fun(input_vec0.data(), input_vec1.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    auto bandwidth = double(length * 2 * sizeof(float)) / (t1 - t0).count();
    fmt::print("{} GB/sec \n", bandwidth);

    free(p_re);
}
