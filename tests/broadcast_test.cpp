#include "galois/op/broadcast.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestBroadCast) {

    int rows_in = 1, cols_in = 1;
    int rows_out = 2, cols_out = 2;

    auto ir_input_type = ir::f32->Tile(rows_in, cols_in);
    auto ir_output_type = ir::f32->Tile(rows_out, cols_out);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::BroadCastCreator>(
        {ir_input_type}, ir_output_type->shape);

    auto jit_engine = jit::Engine::Create();
    auto broadcast_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    std::vector<float> input(rows_in * cols_in);
    for (int i = 0; i < rows_in * cols_in; ++i) {
        input[i] = static_cast<float>(i + 1);
    }

    std::vector<float> output = {1.0f, 1.0f, 
                                1.0f, 1.0f};

    float *result = broadcast_fun(input.data());
    
    for (int i = 0; i < rows_out * cols_out; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
        << "Mismatch at index " << i << ": output=" << output[i];
    }

    free(result);
}