#include "galois/op/concatenate.hpp"

#include "gtest/gtest.h"
#include "tests/galois_test.hpp"

TEST(GaloisTests, Concatenate_2x2Dim0) {
    auto ir_input_type = ir::f32->Tile(2, 2);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
        {ir_input_type, ir_input_type, ir_input_type}, 0);
    auto jit_engine = jit::Engine::Create();
    auto concat_fun =
        jit_engine->EmitOperatorSymbol<float* (*)(float*, float*, float*)>(ir_operator);

    std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> input2 = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> input3 = {9.0f, 10.0f, 11.0f, 12.0f};

    float* output = concat_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit free_mem([output] { free(output); });

    std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}

TEST(GaloisTests, Concatenate_2x2Dim1) {
    auto ir_input_type = ir::f32->Tile(2, 2);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
        {ir_input_type, ir_input_type, ir_input_type}, 1);
    auto jit_engine = jit::Engine::Create();
    auto concat_fun =
        jit_engine->EmitOperatorSymbol<float* (*)(float*, float*, float*)>(ir_operator);

    std::vector<float> input1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> input2 = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> input3 = {9.0f, 10.0f, 11.0f, 12.0f};

    float* output = concat_fun(input1.data(), input2.data(), input3.data());
    boost::scope::scope_exit free_mem([output] { free(output); });

    std::vector<float> expected = {1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}
