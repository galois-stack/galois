#include "tests/galois_test.hpp"
#include "galois/op/convolution_nd.hpp"

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
    auto ir_operator =
        ir_builder->CreateOperatorByCreator<op::ConvolutionCreator>({ir_act_type, ir_weight_type});

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
    convolution2d(output.data(), input.data(), weight.data(), rows_act, rows_weight, stride,
                  padding);

    for (int i = 0; i < length_out; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
            << "Mismatch at index " << i << ": output=" << output[i];
    }

    free(result);
}

TEST(GaloisTests, TestConvolution2D_3x3) {
    int rows_act = 4, cols_act = 4;
    int length_act = rows_act * cols_act;
    int rows_weight = 3;
    int cols_weight = rows_weight;
    int length_weight = rows_weight * cols_weight;
    int stride = 1;
    int padding = 0;
    int rows_out = (rows_act + 2 * padding - rows_weight) / stride + 1;
    int length_out = rows_out * rows_out;

    // For ConvolutionNDCreator<2>, we need:
    // Input format: [C_in, D1, D2] -> [1, 4, 4] for single channel
    // Weight format: [C_out, C_in, K1, K2] -> [1, 1, 3, 3] for single input/output channel
    int input_channels = 1;
    int output_channels = 1;
    
    auto ir_act_type = ir::f32->Tile(input_channels, rows_act, cols_act);  // [1, 4, 4]
    auto ir_weight_type = ir::f32->Tile(output_channels, input_channels, rows_weight, cols_weight);  // [1, 1, 3, 3]
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConvolutionNDCreator<2>>({ir_act_type, ir_weight_type}, std::array<int, 2>{stride, stride});

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    // Prepare input data in [C_in, D1, D2] format
    std::vector<float> input_nd(input_channels * length_act);
    std::vector<float> weight_nd(output_channels * input_channels * length_weight);
    std::vector<float> output(length_out);

    // Fill input data (single channel, so just copy the 2D data)
    for (int i = 0; i < length_act; ++i) {
        input_nd[i] = static_cast<float>(i + 1);
    }
    
    // Fill weight data (single input/output channel, so just copy the 2D kernel)
    for (int i = 0; i < length_weight; ++i) {
        weight_nd[i] = 1.0f;
    }

    float *result = conv_fun(input_nd.data(), weight_nd.data());
    
    // Use original 2D data for reference calculation
    std::vector<float> input_2d(length_act);
    std::vector<float> weight_2d(length_weight);
    for (int i = 0; i < length_act; ++i) {
        input_2d[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < length_weight; ++i) {
        weight_2d[i] = 1.0f;
    }
    
    convolution2d(output.data(), input_2d.data(), weight_2d.data(), rows_act, rows_weight, stride, padding);
    
    for (int i = 0; i < length_out; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
        << "Mismatch at index " << i << ": expected=" << output[i] << ", got=" << result[i];
    }

    free(result);
}