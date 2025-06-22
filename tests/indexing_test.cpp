#include "galois/op/fill.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestIndexing2DArray) {
    // Test indexing a 2D array
    auto ir_tensor_type = ir::f32->Tile(3, 4);  // 3x4 matrix
    auto ir_scalar_type = ir::f32;

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type = ir::OperatorType::Create({ir_tensor_type}, ir_scalar_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "indexing_test");

    // Create indexing with constant indices [1, 2]
    auto ir_index0 = ir_builder->GetInt64Constant(1);
    auto ir_index1 = ir_builder->GetInt64Constant(2);
    auto ir_input = ir_operator->inputs[0];
    auto ir_indexing = ir_builder->Create<ir::Indexing>(
        ir_input, std::vector<std::shared_ptr<ir::Tensor>>{ir_index0, ir_index1});
    // Return the indexed value
    auto ir_return = ir_builder->Create<ir::Return>(ir_indexing);

    // Test compilation
    auto jit_engine = jit::Engine::Create();
    auto index_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *)>(ir_operator);

    // Test execution
    std::vector<float> input_vec(12);                     // 3x4 = 12 elements
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);  // [1.0, 2.0, ..., 12.0]

    auto result = index_fun(input_vec.data());  // result是一个指向inpu_vec.data, 所以不需要释放
    // For row-major order: input_vec[1*4 + 2] = input_vec[6] = 7.0f
    EXPECT_EQ(*result, 7.0f);
}

TEST(GaloisTests, TestIndexingByTensor) {
    // Test indexing a 2D array
    auto ir_tensor_type = ir::f32->Tile(3, 4);  // 3x4 matrix
    auto ir_index_type = ir::i64->Tile(2)->Tile(3);
    auto ir_output_type = ir::f32->Tile(3);

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "indexing_test");
    auto ir_input = ir_operator->inputs[0];
    auto ir_ts_indices = ir_operator->inputs[1];
    auto ir_output = ir_builder->Alloca(ir_output_type);
    {
        auto [ir_grid, _] = ir_builder->CreateGrid(ir_output_type->shape);
        auto ir_index = ir_builder->CreateIdentityAccessor(ir_ts_indices);
        auto ir_output_accessor = ir_builder->CreateIdentityAccessor(ir_output);

        auto ir_index0 = ir_builder->CreateAccessor(ir_index);
        ir_index0->shift_vector[0] = 0;
        auto ir_index1 = ir_builder->CreateAccessor(ir_index);
        ir_index1->shift_vector[0] = 1;
        auto ir_indexing = ir_builder->Create<ir::Indexing>(
            ir_input, std::vector<std::shared_ptr<ir::Tensor>>{ir_index0, ir_index1});
        ir_builder->Write(ir_indexing, ir_output_accessor);
    }
    ir_builder->Return(ir_output);

    // Test compilation
    auto jit_engine = jit::Engine::Create();
    auto index_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    // // Test execution
    std::vector<float> input_vec(ir_tensor_type->Size());
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);  // [1.0, 2.0, ..., 12.0]

    std::vector<int64_t> index_vec(ir_index_type->NormalizeSize());
    index_vec[0] = 0;
    index_vec[1] = 0;

    index_vec[2] = 1;
    index_vec[3] = 1;

    index_vec[4] = 2;
    index_vec[5] = 3;

    auto result = index_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 1.0f);
    EXPECT_EQ(result[1], 6.0f);
    EXPECT_EQ(result[2], 12.0f);
}
