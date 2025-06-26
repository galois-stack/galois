#include "galois/op/convolution_3d.hpp"
#include "tests/galois_test.hpp"
#include <cmath>

void convolution3d_reference(float* output, const float* input, const float* weight,
                            int input_c, int input_h, int input_w,
                            int output_c, int output_h, int output_w,
                            int kernel_h, int kernel_w,
                            int stride_h = 1, int stride_w = 1,
                            int padding_h = 0, int padding_w = 0) {
    // Initialize output to zero
    for (int i = 0; i < output_c * output_h * output_w; ++i) {
        output[i] = 0.0f;
    }
    
    // Input format: [C, H, W], Output format: [C, H, W], Weight format: [C_out, C_in, KH, KW]
    for (int oc = 0; oc < output_c; ++oc) {
        for (int oh = 0; oh < output_h; ++oh) {
            for (int ow = 0; ow < output_w; ++ow) {
                float sum = 0.0f;
                
                for (int ic = 0; ic < input_c; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh - padding_h;
                            int iw = ow * stride_w + kw - padding_w;
                            
                            // Check bounds
                            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                // Input layout: [C, H, W]
                                int input_idx = ic * input_h * input_w + ih * input_w + iw;
                                // Weight layout: [C_out, C_in, KH, KW]
                                int weight_idx = oc * input_c * kernel_h * kernel_w + 
                                               ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                // Output layout: [C, H, W]
                int output_idx = oc * output_h * output_w + oh * output_w + ow;
                output[output_idx] = sum;
            }
        }
    }
}

TEST(GaloisTests, TestConvolution3D_Simple) {
    int input_c = 2, input_h = 4, input_w = 4;
    int output_c = 3;
    int kernel_h = 3, kernel_w = 3;
    
    // Input format: [C, H, W], Weight format: [C_out, C_in, KH, KW]
    auto ir_input_type = ir::f32->Tile(input_c, input_h, input_w);
    auto ir_weight_type = ir::f32->Tile(output_c, input_c, kernel_h, kernel_w);
    
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::Convolution3DCreator>(
        {ir_input_type, ir_weight_type});

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    std::vector<float> input(input_c * input_h * input_w);
    std::vector<float> weight(output_c * input_c * kernel_h * kernel_w);
    
    // Initialize input in [C, H, W] format
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10 + 1);
    }
    
    // Initialize weight
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 3) + 1) * 0.1f;
    }

    float *result = conv_fun(input.data(), weight.data());
    
    int output_h = input_h - kernel_h + 1;
    int output_w = input_w - kernel_w + 1;
    std::vector<float> expected(output_c * output_h * output_w);
    
    convolution3d_reference(expected.data(), input.data(), weight.data(),
                          input_c, input_h, input_w, output_c, output_h, output_w,
                          kernel_h, kernel_w);
    
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5)
            << "Mismatch at index " << i << ": got=" << result[i] << ", expected=" << expected[i];
    }

    free(result);
}

TEST(GaloisTests, TestConvolution3D_WithStride) {
    int input_c = 2, input_h = 6, input_w = 6;
    int output_c = 2;
    int kernel_h = 3, kernel_w = 3;
    int stride_h = 2, stride_w = 2;
    
    // Input format: [C, H, W], Weight format: [C_out, C_in, KH, KW]
    auto ir_input_type = ir::f32->Tile(input_c, input_h, input_w);
    auto ir_weight_type = ir::f32->Tile(output_c, input_c, kernel_h, kernel_w);
    
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::Convolution3DCreator>(
        {ir_input_type, ir_weight_type}, stride_h, stride_w, 0, 0);

    auto jit_engine = jit::Engine::Create();
    auto conv_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, float *)>(ir_operator);

    std::vector<float> input(input_c * input_h * input_w);
    std::vector<float> weight(output_c * input_c * kernel_h * kernel_w);
    
    // Initialize input in [C, H, W] format
    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i % 10 + 1);
    }
    
    // Initialize weight
    for (int i = 0; i < weight.size(); ++i) {
        weight[i] = static_cast<float>((i % 3) + 1) * 0.1f;
    }

    float *result = conv_fun(input.data(), weight.data());
    
    int output_h = (input_h - kernel_h) / stride_h + 1;
    int output_w = (input_w - kernel_w) / stride_w + 1;
    std::vector<float> expected(output_c * output_h * output_w);
    
    convolution3d_reference(expected.data(), input.data(), weight.data(),
                          input_c, input_h, input_w, output_c, output_h, output_w,
                          kernel_h, kernel_w, stride_h, stride_w);
    
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-5)
            << "Mismatch at index " << i << ": got=" << result[i] << ", expected=" << expected[i];
    }

    free(result);
}