#include "galois/op/relu.hpp"

#include "galois/ir/ir.hpp"
#include "galois/jit/engine.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestBasedReLU) {
    int64_t length = 128;
    auto ir_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();

    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ReluCreator>({ir_type});

    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    std::vector<float> float_vec(length);
    for (int i = 0; i < length; ++i) {
        float_vec[i] = static_cast<float>(i - 64);  // Mix of positive, negative, and zero
    }

    auto output_vec = model_fun(float_vec.data());

    for (int i = 0; i < length; i++) {
        float expected = std::max(0.0f, float_vec[i]);
        GALOIS_ASSERT(output_vec[i] == expected);
    }

    free(output_vec);
}

TEST(GaloisTests, TestReLUScalar) {
    auto ir_type = ir::f32;
    auto ir_builder = ir::Builder::Create();

    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ReluCreator>({ir_type});

    auto jit_engine = jit::Engine::Create();
    auto model_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_operator);

    // Test positive value
    float input1 = 3.14f;
    auto result1 = *model_fun(&input1);
    GALOIS_ASSERT(result1 == 3.14f);

    // Test negative value
    float input2 = -2.71f;
    auto result2 = *model_fun(&input2);
    GALOIS_ASSERT(result2 == 0.0f);

    // Test zero
    float input3 = 0.0f;
    auto result3 = *model_fun(&input3);
    GALOIS_ASSERT(result3 == 0.0f);
}