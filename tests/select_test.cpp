#include "galois/op/select.hpp"

#include <cstdint>
#include <vector>

#include "galois/op/compare.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestSelectScalarBool) {
    auto ir_condition_type = ir::bool_;
    auto ir_value_type = ir::f32;
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
        {ir_condition_type, ir_value_type, ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto select_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(bool *, float *, float *)>(ir_operator);

    // Test condition = true
    bool condition = true;
    float true_value = 3.14f;
    float false_value = 2.71f;
    auto result = *select_fun(&condition, &true_value, &false_value);
    EXPECT_FLOAT_EQ(result, true_value);

    // Test condition = false
    condition = false;
    result = *select_fun(&condition, &true_value, &false_value);
    EXPECT_FLOAT_EQ(result, false_value);
}

TEST(GaloisTests, TestSelectTensorElementwise) {
    // Test element-wise selection with tensor inputs
    int length = 6;
    auto ir_condition_type = ir::bool_->Tile(length);
    auto ir_value_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
        {ir_condition_type, ir_value_type, ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto select_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(bool *, float *, float *)>(ir_operator);

    // Test data: [true, false, true, false, true, false]
    bool condition[6] = {true, false, true, false, true, false};
    std::vector<float> true_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> false_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<float> expected = {1.0f, 20.0f, 3.0f, 40.0f, 5.0f, 60.0f};

    float *result = select_fun(condition, true_values.data(), false_values.data());

    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(GaloisTests, TestSelectMatrix) {
    // Test selection with 2D tensor (matrix)
    int rows = 2, cols = 3;
    int length = rows * cols;
    auto ir_condition_type = ir::bool_->Tile(rows, cols);
    auto ir_value_type = ir::f32->Tile(rows, cols);
    auto ir_builder = ir::Builder::Create();
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
        {ir_condition_type, ir_value_type, ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto select_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(bool *, float *, float *)>(ir_operator);

    // Test data: condition matrix [[true, false, true], [false, true, false]]
    bool condition[6] = {true, false, true, false, true, false};
    std::vector<float> true_values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
    std::vector<float> false_values = {11.1f, 22.2f, 33.3f, 44.4f, 55.5f, 66.6f};
    std::vector<float> expected = {1.1f, 22.2f, 3.3f, 44.4f, 5.5f, 66.6f};

    float *result = select_fun(condition, true_values.data(), false_values.data());

    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(GaloisTests, TestSelectWithComparison) {
    int length = 4;
    auto ir_value_type = ir::f32->Tile(length);
    auto ir_builder = ir::Builder::Create();

    // Create comparison operator: x > threshold
    auto ir_compare_operator =
        ir_builder->CreateOperatorByCreator<op::GreaterCreator>({ir_value_type, ir_value_type});

    auto ir_condition_type = ir::bool_->Tile(length);
    auto ir_select_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
        {ir_condition_type, ir_value_type, ir_value_type});

    auto jit_engine = jit::Engine::Create();
    auto compare_fun =
        jit_engine->EmitOperatorSymbol<bool *(*)(float *, float *)>(ir_compare_operator);
    auto select_fun =
        jit_engine->EmitOperatorSymbol<float *(*)(bool *, float *, float *)>(ir_select_operator);

    std::vector<float> input = {1.0f, 3.0f, 2.0f, 4.0f};
    std::vector<float> threshold = {2.5f, 2.5f, 2.5f, 2.5f};
    std::vector<float> high_values = {10.0f, 30.0f, 20.0f, 40.0f};
    std::vector<float> low_values = {100.0f, 300.0f, 200.0f, 400.0f};

    bool *condition = compare_fun(input.data(), threshold.data());

    float *result = select_fun(condition, high_values.data(), low_values.data());

    // Expected: input > 2.5 ? high_values : low_values
    // [1.0 > 2.5 = false, 3.0 > 2.5 = true, 2.0 > 2.5 = false, 4.0 > 2.5 = true]
    // Result should be: [100.0, 30.0, 200.0, 40.0]
    std::vector<float> expected = {100.0f, 30.0f, 200.0f, 40.0f};

    for (int i = 0; i < length; ++i) {
        EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(GaloisTests, TestSelectTypeInference) {
    // Test type inference for different data types
    auto ir_builder = ir::Builder::Create();

    {
        auto ir_condition_type = ir::bool_;
        auto ir_value_type = ir::f64;
        auto ir_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
            {ir_condition_type, ir_value_type, ir_value_type});

        EXPECT_EQ(ir_operator->GetOperatorType()->output_type->DataType(), ir::f64);
    }

    {
        auto ir_condition_type = ir::bool_->Tile(10);
        auto ir_value_type = ir::i64->Tile(10);
        auto ir_operator = ir_builder->CreateOperatorByCreator<op::SelectCreator>(
            {ir_condition_type, ir_value_type, ir_value_type});

        EXPECT_EQ(ir_operator->GetOperatorType()->output_type->DataType(), ir::i64);
        EXPECT_EQ(ir_operator->GetOperatorType()->output_type->shape.size(), 1);
        EXPECT_EQ(ir_operator->GetOperatorType()->output_type->shape[0], 10);
    }
}