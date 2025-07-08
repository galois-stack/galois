#include <cstdint>
#include <vector>

#include "galois/op/compare.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestCompareEqual) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::EqualCreator>({ir_input_type, ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto equal_fun = jit_engine->EmitOperatorSymbol<bool *(*)(float *, float *)>(ir_operator);

    float input1 = 3.14f;
    float input2 = 3.14f;
    auto result = *equal_fun(&input1, &input2);
    EXPECT_TRUE(result);
    fmt::print("Equal({}, {}) = {}\n", input1, input2, result);

    input2 = 2.71f;
    result = *equal_fun(&input1, &input2);
    EXPECT_FALSE(result);
    fmt::print("Equal({}, {}) = {}\n", input1, input2, result);
}

TEST(GaloisTests, TestCompareLess) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::LessCreator>({ir_input_type, ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto less_fun = jit_engine->EmitOperatorSymbol<bool *(*)(float *, float *)>(ir_operator);

    float input1 = 2.0f;
    float input2 = 3.0f;
    auto result = *less_fun(&input1, &input2);
    EXPECT_TRUE(result);
    fmt::print("Less({}, {}) = {}\n", input1, input2, result);

    input1 = 4.0f;
    result = *less_fun(&input1, &input2);
    EXPECT_FALSE(result);
    fmt::print("Less({}, {}) = {}\n", input1, input2, result);
}

TEST(GaloisTests, TestCompareInteger) {
    auto ir_input_type = ir::i32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::GreaterCreator>({ir_input_type, ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto greater_fun = jit_engine->EmitOperatorSymbol<bool *(*)(int32_t *, int32_t *)>(ir_operator);

    int32_t input1 = 10;
    int32_t input2 = 5;
    auto result = *greater_fun(&input1, &input2);
    EXPECT_TRUE(result);
    fmt::print("Greater({}, {}) = {}\n", input1, input2, result);

    input1 = 3;
    result = *greater_fun(&input1, &input2);
    EXPECT_FALSE(result);
    fmt::print("Greater({}, {}) = {}\n", input1, input2, result);
}

TEST(GaloisTests, TestCompareTensor) {
    int rows = 2, cols = 3;
    int length = rows * cols;
    auto ir_input_type = ir::f32->Tile(rows, cols);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::LessEqualCreator>({ir_input_type, ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto less_equal_fun = jit_engine->EmitOperatorSymbol<bool *(*)(float *, float *)>(ir_operator);

    std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> input2 = {1.5f, 1.5f, 3.5f, 3.5f, 5.5f, 5.5f};
    std::vector<bool> output(length);

    bool *result = less_equal_fun(input1.data(), input2.data());
    for (int i = 0; i < length; ++i) {
        bool expected = input1[i] <= input2[i];
        EXPECT_EQ(result[i], expected)
            << "Mismatch at index " << i << ": input1=" << input1[i] << ", input2=" << input2[i];
        output[i] = result[i];
    }

    fmt::print("LessEqual tensor comparison output: ");
    for (bool v : output) fmt::print("{} ", v);
    fmt::print("\n");
}
