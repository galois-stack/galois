#include "galois/op/convolution.hpp"
#include "tests/galois_test.hpp"

void convolution2d(float* output, float* input, float* weight, int rows_act, int rows_weight, int stride, int padding) {
    int cols_act = rows_act;
    int rows_out = (rows_act + 2 * padding - rows_weight) / stride + 1;
    int cols_out = rows_out;
    
    for (int i = 0; i < rows_out; ++i) {
        for (int j = 0; j < cols_out; ++j) {
            int start_row = i * stride - padding;
            int start_col = j * stride - padding;
            
            float sum = 0.0f;
            for (int ki = 0; ki < rows_weight; ++ki) {
                for (int kj = 0; kj < rows_weight; ++kj) {
                    int input_row = start_row + ki;
                    int input_col = start_col + kj;
                    
                    bool is_valid = (input_row >= 0) && (input_row < rows_act) && 
                                    (input_col >= 0) && (input_col < cols_act);

                    int input_idx = is_valid ? input_row * cols_act + input_col : 0;
                    int weight_idx = ki * rows_weight + kj;
                    
                    sum += (is_valid ? *(input + input_idx) : 0.0f) * *(weight + weight_idx);
                }
            }

            *(output + i * cols_out + j) = sum;
        }
    }
}

TEST(GaloisTests, TestConvolution3x3) {
    int rows_act = 4, cols_act = 4;
    int length_act = rows_act * cols_act;
    int rows_weight = 3;
    int cols_weight = rows_weight;
    int length_weight = rows_weight * cols_weight;
    int stride = 1;
    int padding = 0;
    int rows_out = (rows_act + 2 * padding - rows_weight) / stride + 1;
    int length_out = rows_out * rows_out;

    auto ir_act_type = ir::f32->Tile(rows_act, cols_act);
    auto ir_weight_type = ir::f32->Tile(rows_weight, cols_weight);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConvolutionCreator>(
        {ir_act_type, ir_weight_type});

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    std::vector<float> input(length_act);
    std::vector<float> weight(length_weight);
    std::vector<float> output(length_out);

    for (int i = 0; i < length_act; ++i) {
        input[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < length_weight; ++i) {
        weight[i] = 1.0f;
    }

    float *result = conv_fun(input.data(), weight.data());
    convolution2d(output.data(), input.data(), weight.data(), rows_act, rows_weight, stride, padding);
    
    for (int i = 0; i < length_out; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
        << "Mismatch at index " << i << ": output=" << output[i];
    }

    free(result);
}