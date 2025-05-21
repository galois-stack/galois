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
