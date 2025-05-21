#include "galois/op/sigmoid.hpp"

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestSigmoid) {
    auto ir_input_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SigmoidCreator>({ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto sigmoid_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    float input = 0.0f;
    auto value = *sigmoid_fun(&input);
    fmt::print("sigmoid(0) = {}\n", value);

    input = 3.1415f / 2.0f;
    value = *sigmoid_fun(&input);
    fmt::print("sigmoid(3.1415/2) = {}\n", value);

    input = -1.0f;
    value = *sigmoid_fun(&input);
    fmt::print("sigmoid(-1) = {}\n", value);
}

TEST(GaloisTests, TestSigmoid3x3) {
    int rows = 2, cols = 3;
    int length = rows * cols;
    auto ir_input_type = ir::f32->Tile(rows, cols);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SigmoidCreator>({ir_input_type});

    auto jit_engine = jit::Engine::Create();
    auto sigmoid_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input = {0.0f, 1.0f, -1.0f, 3.0f, -2.0f, 0.5f};
    std::vector<float> output(length);

    float *result = sigmoid_fun(input.data());
    for (int i = 0; i < length; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input[i]));
        EXPECT_NEAR(result[i], expected, 1e-5)
            << "Mismatch at index " << i << ": input=" << input[i];
        output[i] = result[i];
    }

    fmt::print("sigmoid(mat) output: ");
    for (float v : output) fmt::print("{} ", v);
    fmt::print("\n");
}
