#include "galois/op/padding.hpp"

#include "gtest/gtest.h"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestPadding) {
    auto ir_input_type = ir::f32->Tile(1, 1);
    Eigen::VectorXi64 shape(2);
    shape[0] = 32;
    shape[1] = 32;

    auto ir_builder = ir::Builder::Create();
    auto ir_padding_creator = op::PaddingCreator::Create(shape);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_padding_creator, {ir_input_type});
    auto jit_engine = jit::Engine::Create();
    auto padding_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input(1 * 1);
    input[0] = 42.0f;
    auto padded_value = padding_fun(input.data());
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i == 0 && j == 0) {
                GALOIS_ASSERT(padded_value[0] == input[0]);
            } else {
                GALOIS_ASSERT(padded_value[i * 8 + j] == 0.0f);
            }
        }
    }
}

TEST(GaloisTests, TestPaddingIdentity) {
    auto ir_input_type = ir::f32->Tile(32, 32);
    Eigen::VectorXi64 shape(2);
    shape[0] = 32;
    shape[1] = 32;

    auto ir_builder = ir::Builder::Create();
    auto ir_padding_creator = op::PaddingCreator::Create(shape);
    auto ir_operator = ir_builder->CreateOperatorByCreator(ir_padding_creator, {ir_input_type});
    auto jit_engine = jit::Engine::Create();
    auto padding_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input(ir_input_type->Size());
    auto padded_value = padding_fun(input.data());
}
