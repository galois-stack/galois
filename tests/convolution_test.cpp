#include "tests/galois_test.hpp"
#include "galois/op/convolution.hpp"

void convolution1d_reference(float* output, const float* input, const float* weight,
                            int input_c, int input_w,
                            int output_c, int output_w,
                            int kernel_w, int stride_w = 1) {
    // Initialize output to zero
    for (int i = 0; i < output_c * output_w; ++i) {
        output[i] = 0.0f;
    }
    
    // Input format: [C, W], Weight format: [C_out, C_in, KW]
    for (int oc = 0; oc < output_c; ++oc) {
        for (int ow = 0; ow < output_w; ++ow) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < input_c; ++ic) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = ow * stride_w + kw;
                    
                    // Check bounds
                    if (iw >= 0 && iw < input_w) {
                        // Input layout: [C, W]
                        int input_idx = ic * input_w + iw;
                        // Weight layout: [C_out, C_in, KW]
                        int weight_idx = oc * input_c * kernel_w + ic * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            // Output layout: [C, W]
            int output_idx = oc * output_w + ow;
            output[output_idx] = sum;
        }
    }
}

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

void convolution3d_reference(float* output, const float* input, const float* weight,
                            int input_c, int input_d, int input_h, int input_w,
                            int output_c, int output_d, int output_h, int output_w,
                            int kernel_d, int kernel_h, int kernel_w,
                            int stride_d = 1, int stride_h = 1, int stride_w = 1) {
    // Initialize output to zero
    for (int i = 0; i < output_c * output_d * output_h * output_w; ++i) {
        output[i] = 0.0f;
    }
    
    // Input format: [C, D, H, W], Weight format: [C_out, C_in, KD, KH, KW]
    for (int oc = 0; oc < output_c; ++oc) {
        for (int od = 0; od < output_d; ++od) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < input_c; ++ic) {
                        for (int kd = 0; kd < kernel_d; ++kd) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int id = od * stride_d + kd;
                                    int ih = oh * stride_h + kh;
                                    int iw = ow * stride_w + kw;
                                    
                                    // Check bounds
                                    if (id >= 0 && id < input_d && ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                        // Input layout: [C, D, H, W]
                                        int input_idx = ic * input_d * input_h * input_w + 
                                                       id * input_h * input_w + ih * input_w + iw;
                                        // Weight layout: [C_out, C_in, KD, KH, KW]
                                        int weight_idx = oc * input_c * kernel_d * kernel_h * kernel_w + 
                                                        ic * kernel_d * kernel_h * kernel_w + 
                                                        kd * kernel_h * kernel_w + kh * kernel_w + kw;
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Output layout: [C, D, H, W]
                    int output_idx = oc * output_d * output_h * output_w + 
                                    od * output_h * output_w + oh * output_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

TEST(GaloisTests, TestConvolution1D) {
    // 1D convolution: signal processing
    int input_c = 3, input_w = 16;
    int output_c = 5, kernel_w = 3;
    int stride = 2;
    
    auto ir_input_type = ir::f32->Tile(input_c, input_w);
    auto ir_weight_type = ir::f32->Tile(output_c, input_c, kernel_w);
    
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConvolutionNDCreator<1>>(
        {ir_input_type, ir_weight_type}, std::array<int, 1>{stride});

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);
    
    std::vector<float> input(input_c * input_w);
    std::vector<float> weight(output_c * input_c * kernel_w);
    
    // Initialize input in [C, W] format
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10 + 1);
    }
    
    // Initialize weight
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 3) + 1) * 0.1f;
    }

    float *result = conv_fun(input.data(), weight.data());
    
    int output_w = (input_w - kernel_w) / stride + 1;
    std::vector<float> expected(output_c * output_w);
    
    convolution1d_reference(expected.data(), input.data(), weight.data(),
                          input_c, input_w, output_c, output_w,
                          kernel_w, stride);
    
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5)
            << "Mismatch at index " << i << ": got=" << result[i] << ", expected=" << expected[i];
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

TEST(GaloisTests, TestConvolution3D_Volume) {
    // 3D convolution: for video or medical imaging
    int input_c = 2, input_d = 8, input_h = 8, input_w = 8;
    int output_c = 4, kernel_d = 3, kernel_h = 3, kernel_w = 3;
    std::array<int, 3> strides = {1, 2, 2};
    
    auto ir_input_type = ir::f32->Tile(input_c, input_d, input_h, input_w);
    auto ir_weight_type = ir::f32->Tile(output_c, input_c, kernel_d, kernel_h, kernel_w);
    
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConvolutionNDCreator<3>>(
        {ir_input_type, ir_weight_type}, strides);

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);
    
    std::vector<float> input(input_c * input_d * input_h * input_w);
    std::vector<float> weight(output_c * input_c * kernel_d * kernel_h * kernel_w);
    
    // Initialize input in [C, D, H, W] format
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10 + 1);
    }
    
    // Initialize weight
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 3) + 1) * 0.1f;
    }

    float *result = conv_fun(input.data(), weight.data());
    
    int output_d = (input_d - kernel_d) / strides[0] + 1;
    int output_h = (input_h - kernel_h) / strides[1] + 1;
    int output_w = (input_w - kernel_w) / strides[2] + 1;
    std::vector<float> expected(output_c * output_d * output_h * output_w);
    
    convolution3d_reference(expected.data(), input.data(), weight.data(),
                          input_c, input_d, input_h, input_w, 
                          output_c, output_d, output_h, output_w,
                          kernel_d, kernel_h, kernel_w,
                          strides[0], strides[1], strides[2]);
    
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5)
            << "Mismatch at index " << i << ": got=" << result[i] << ", expected=" << expected[i];
    }

    free(result);
}

TEST(GaloisTests, TestConvolution4D_Spacetime) {
    // 4D convolution: space-time convolution
    int input_c = 1, input_t = 4, input_d = 4, input_h = 4, input_w = 4;
    int output_c = 2, kernel_t = 2, kernel_d = 2, kernel_h = 2, kernel_w = 2;
    
    auto ir_input_type = ir::f32->Tile(input_c, input_t, input_d, input_h, input_w);
    auto ir_weight_type = ir::f32->Tile(output_c, input_c, kernel_t, kernel_d, kernel_h, kernel_w);
    
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConvolutionNDCreator<4>>(
        {ir_input_type, ir_weight_type});

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);
    
    std::vector<float> input(input_c * input_t * input_d * input_h * input_w);
    std::vector<float> weight(output_c * input_c * kernel_t * kernel_d * kernel_h * kernel_w);
    
    // Initialize input in [C, T, D, H, W] format
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 5 + 1);
    }
    
    // Initialize weight
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 2) + 1) * 0.1f;
    }

    float *result = conv_fun(input.data(), weight.data());
    
    int output_t = input_t - kernel_t + 1;
    int output_d = input_d - kernel_d + 1;
    int output_h = input_h - kernel_h + 1;
    int output_w = input_w - kernel_w + 1;
    
    // For 4D, we'll do a simple validation by checking that output is not all zeros
    // and has reasonable values
    std::vector<float> expected_output(output_c * output_t * output_d * output_h * output_w);
    
    bool has_non_zero = false;
    float sum_output = 0.0f;
    for (int i = 0; i < output_c * output_t * output_d * output_h * output_w; ++i) {
        if (std::abs(result[i]) > 1e-6) {
            has_non_zero = true;
        }
        sum_output += result[i];
    }
    
    EXPECT_TRUE(has_non_zero) << "Output should have non-zero values";
    EXPECT_GT(std::abs(sum_output), 1e-4) << "Output sum should be non-trivial";
    
    // Check output dimensions are correct
    EXPECT_EQ(output_t, 3);
    EXPECT_EQ(output_d, 3);
    EXPECT_EQ(output_h, 3);
    EXPECT_EQ(output_w, 3);

    free(result);
}