#include "galois/op/transpose.hpp"

#include "gtest/gtest.h"
#include "tests/galois_test.hpp"

TEST(GaloisTests, Transpose_2x3) {
    auto ir_input_type = ir::f32->Tile(2, 3);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::TransposeCreator>({ir_input_type}, 0, 1);
    auto jit_engine = jit::Engine::Create();
    auto transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    float* output = transpose_fun(input.data());
    boost::scope::scope_exit free_mem([output] { free(output); });
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}

TEST(GaloisTests, Transpose_3x2) {
    auto ir_input_type = ir::f32->Tile(3, 2);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::TransposeCreator>({ir_input_type}, 0, 1);
    auto jit_engine = jit::Engine::Create();
    auto transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> input = {1, 2, 3, 4, 5, 6};
    float* output = transpose_fun(input.data());
    boost::scope::scope_exit free_mem([output] { free(output); });
    std::vector<float> expected = {1, 3, 5, 2, 4, 6};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}

TEST(GaloisTests, Transpose_2x2x2_dim1_dim2) {
    auto ir_input_type = ir::f32->Tile(2, 2, 2);
    auto ir_builder = ir::Builder::Create();

    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::TransposeCreator>({ir_input_type}, 1, 2);
    auto jit_engine = jit::Engine::Create();
    auto transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    float* output = transpose_fun(input.data());
    boost::scope::scope_exit free_mem([output] { free(output); });
    std::vector<float> expected = {1, 3, 2, 4, 5, 7, 6, 8};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}

TEST(GaloisTests, Transpose_7x8x9_dim1_dim2) {
    auto ir_input_type = ir::f32->Tile(7, 8, 9);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::TransposeCreator>({ir_input_type}, 1, 2);
    auto jit_engine = jit::Engine::Create();
    auto transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> input(7 * 8 * 9);
    for (int i = 0; i < 7 * 8 * 9; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    std::vector<float> expected(7 * 8 * 9);
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 9; ++k) {
                expected[i * 9 * 8 + k * 8 + j] = input[i * 8 * 9 + j * 9 + k];
            }
        }
    }

    float* output = transpose_fun(input.data());
    boost::scope::scope_exit free_mem([output] { free(output); });

    for (size_t idx = 0; idx < expected.size(); ++idx) {
        GALOIS_ASSERT(output[idx] == expected[idx]);
    }
}
