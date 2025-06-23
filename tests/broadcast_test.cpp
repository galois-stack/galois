#include "galois/op/broadcast.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestBroadCast) {
    int rows_act = 2, cols_act = 2;

    auto ir_act_1_type = ir::f32->Tile(1, 1);
    auto ir_act_2_type = ir::f32->Tile(rows_act, cols_act);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::BroadCastCreator>(
        {ir_act_1_type, ir_act_2_type});

    auto jit_engine = jit::Engine::Create();
    auto broadcast_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    std::vector<float> input(1);
    std::vector<float> target(rows_act*cols_act);
    std::vector<float> output = {1.0f, 1.0f, 
                                1.0f, 1.0f};

    for (int i = 0; i < rows_act; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    float *result = broadcast_fun(input.data(), target.data());
    
    for (int i = 0; i < rows_act * cols_act; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
        << "Mismatch at index " << i << ": output=" << output[i];
    }

    free(result);
}