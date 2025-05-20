#include "galois/op/flatten.hpp"

#include "gtest/gtest.h"
#include "tests/galois_test.hpp"

TEST(GaloisTests, Flatten_4x4) {
    auto ir_input_type = ir::f32->Tile(4, 4);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::FlattenCreator>({ir_input_type});
    auto jit_engine = jit::Engine::Create();
    auto flatten_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float* output = flatten_fun(input.data());
    std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    for (size_t i = 0; i < expected.size(); ++i) {
        GALOIS_ASSERT(output[i] == expected[i]);
    }
}
