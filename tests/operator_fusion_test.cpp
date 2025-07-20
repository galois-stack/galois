#include "galois/op/operator_fusion.hpp"

#include <boost/scope/scope_exit.hpp>
#include <cmath>

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestOperatorFusion) {
    constexpr int64_t length = 8;
    auto ir_input_type = ir::f32->Tile(length);
    std::vector<std::string> operations = {"add", "sub"};

    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::OperatorFusionCreator>(
        {ir_input_type, ir_input_type, ir_input_type}, operations);

    auto jit_engine = jit::Engine::Create();
    auto operatorfusion_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *, float *)>(ir_operator);

    std::array<float, length> input1 = {3.1415f, 2.7182f, 1.1415f, 0.7182f,
                                        4.1415f, 5.7182f, 6.1415f, 7.7182f};
    std::array<float, length> input2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, length> input3 = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    std::vector<float> real_result(length);
    for (size_t i = 0; i < length; ++i) {
        real_result[i] = (input1[i] + input2[i]) - input3[i];
    }

    float *result = operatorfusion_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < length; ++i) {
        ASSERT_NEAR(result[i], real_result[i], 1e-5);
    }
}