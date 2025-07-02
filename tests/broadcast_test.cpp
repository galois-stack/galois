#include "galois/op/broadcast.hpp"

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestBroadCast_Tile_1x1_Tile_2x2) {
    auto ir_input_type = ir::f32->Tile(1, 1);
    auto ir_output_type = ir::f32->Tile(2, 2);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::BroadCastCreator>(
        {ir_input_type}, ir_output_type->shape);

    auto jit_engine = jit::Engine::Create();
    auto broadcast_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    float input = 3.1415f;
    float *result = broadcast_fun(&input);
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < ir_output_type->Size(); ++i) {
        ASSERT_NEAR(result[i], input, 1e-5);
    }
}

TEST(GaloisTests, TestBroadCast_Scalar_Tile_1024x1) {
    auto ir_input_type = ir::f32;
    auto ir_output_type = ir::f32->Tile(1024, 1);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::BroadCastCreator>(
        {ir_input_type}, ir_output_type->shape);

    auto jit_engine = jit::Engine::Create();
    auto broadcast_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    float input = 3.1415f;
    float *result = broadcast_fun(&input);
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < ir_output_type->Size(); ++i) {
        ASSERT_NEAR(result[i], input, 1e-5);
    }
}

TEST(GaloisTests, TestBroadCast_Tile_1x2_Tile_4x2) {
    auto ir_input_type = ir::f32->Tile(1, 2);
    auto ir_output_type = ir::f32->Tile(4, 2);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::BroadCastCreator>(
        {ir_input_type}, ir_output_type->shape);

    auto jit_engine = jit::Engine::Create();
    auto broadcast_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::array<float, 2> input = {3.1415f, 2.7182f};
    float *result = broadcast_fun(input.data());
    boost::scope::scope_exit guard([&] { free(result); });

    for (size_t i = 0; i < ir_output_type->shape[0]; ++i) {
        for (size_t j = 0; j < ir_output_type->shape[1]; ++j) {
            ASSERT_NEAR(result[i * ir_output_type->shape[1] + j], input[j], 1e-5);
        }
    }
}
