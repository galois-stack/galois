#include "galois/op/compare.hpp"  // 假设你把 CompareCreator 写在这里
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestLess) {
    // 创建张量类型：输入为 float32 向量，输出为 bool 向量
    int64_t length = 102400000;
    auto ir_input_type = ir::f32->Tile(length);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::LessCreator>({ir_input_type, ir_input_type});
    auto jit_engine = jit::Engine::Create();

    // 输入两个 float 指针，输出 bool 指针
    auto less_fun = jit_engine->EmitOperatorSymbol<bool*(*)(float*, float*)>(ir_operator);

    std::vector<float> input_vec0(length);
    std::vector<float> input_vec1(length);

    // 初始化：input0 全为 0，input1 全为 1，所以 input0 < input1 恒为 true
    std::fill(input_vec0.begin(), input_vec0.end(), 0.0f);
    std::fill(input_vec1.begin(), input_vec1.end(), 1.0f);

    auto t0 = std::chrono::steady_clock::now();
    auto p_re = less_fun(input_vec0.data(), input_vec1.data());
    auto t1 = std::chrono::steady_clock::now();

    auto bandwidth = double(length * 2 * sizeof(float)) / (t1 - t0).count();
    fmt::print("Compare (Less) bandwidth: {} GB/sec\n", bandwidth);

    // 验证每个 bool 值都为 true（即 1）
    for (int64_t i = 0; i < length; ++i) {
        ASSERT_TRUE(p_re[i]);
    }

    free(p_re);
}
