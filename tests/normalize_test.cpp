#include "galois/op/normalize.hpp"

#include "tests/galois_test.hpp"

float calculate_mean(float *matrix, int rows, int cols) {
    float sum = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sum += *(matrix + i * cols + j);
        }
    }
    return sum / (rows * cols);
}

float calculate_variance(float *matrix, int rows, int cols, float mean) {
    float sum_squared_diff = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = *(matrix + i * cols + j) - mean;
            sum_squared_diff += diff * diff;
        }
    }
    return sum_squared_diff / (rows * cols);
}

void normalize_matrix(float *matrix, float *output, int rows, int cols, float *gama, float *beta) {
    float mean = calculate_mean(matrix, rows, cols);
    float variance = calculate_variance(matrix, rows, cols, mean);
    float epsilon = 1e-5;
    float std_dev = std::sqrt(variance + epsilon);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(output + i * cols + j) =
                ((*(matrix + i * cols + j) - mean) / std_dev) * (*gama) + (*beta);
        }
    }
}

TEST(GaloisTests, TestNormalize3x3) {
    int rows = 3, cols = 3;
    int length = rows * cols;
    auto ir_input_type = ir::f32->Tile(rows, cols);
    auto ir_gama_type = ir::f32;
    auto ir_beta_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::NormalizeCreator>(
        {ir_input_type, ir_gama_type, ir_beta_type});

    auto jit_engine = jit::Engine::Create();
    auto normalize_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(float *, float *, float *)>(ir_operator);

    std::vector<float> input = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f};
    float gama = 0.5f;
    float beta = 0.3f;
    std::vector<float> output(length);

    float *result = normalize_fun(input.data(), &gama, &beta);
    normalize_matrix(input.data(), output.data(), rows, cols, &gama, &beta);

    for (int i = 0; i < length; ++i) {
        EXPECT_NEAR(result[i], output[i], 1e-5)
            << "Mismatch at index " << i << ": input=" << input[i];
        output[i] = result[i];
    }

    free(result);
}