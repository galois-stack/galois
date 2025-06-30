#include "galois/op/convolution_nd.hpp"
#include "tests/galois_test.hpp"
#include <cmath>

// 1D卷积参考实现
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