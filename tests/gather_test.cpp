#include "galois/op/gather.hpp"

#include "tests/galois_test.hpp"

TEST(GaloisTests, TestGather1D) {
    auto ir_tensor_type = ir::f32->Tile(8);          // 1D array with 8 elements
    auto ir_index_type = ir::i64->Tile(1)->Tile(4);  // [1, 4] - 1D coordinates, 4 sample points
    auto ir_output_type = ir::f32->Tile(4);          // [4] output points

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "gather1d_test");

    auto ir_input = ir_operator->inputs[0];
    auto ir_indices = ir_operator->inputs[1];

    auto gather_creator = galois::op::GatherCreator::Create();
    gather_creator->Express({ir_input, ir_indices}, ir_builder);

    auto jit_engine = jit::Engine::Create();
    auto gather_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    // Test data: [1, 2, 3, 4, 5, 6, 7, 8]
    std::vector<float> input_vec(8);
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);

    // Indices: [2], [0], [7], [4] -> should gather elements at positions 2, 0, 7, 4
    std::vector<int64_t> index_vec = {2, 0, 7, 4};

    auto result = gather_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 3.0f);  // input[2] = 3
    EXPECT_EQ(result[1], 1.0f);  // input[0] = 1
    EXPECT_EQ(result[2], 8.0f);  // input[7] = 8
    EXPECT_EQ(result[3], 5.0f);  // input[4] = 5
}

TEST(GaloisTests, TestGather2D) {
    auto ir_tensor_type = ir::f32->Tile(3, 4);       // 3x4 matrix
    auto ir_index_type = ir::i64->Tile(2)->Tile(3);  // [2, 3] nested structure
    auto ir_output_type = ir::f32->Tile(3);          // [3] shape

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "gather_test");

    auto ir_input = ir_operator->inputs[0];
    auto ir_indices = ir_operator->inputs[1];

    auto gather_creator = galois::op::GatherCreator::Create();
    gather_creator->Express({ir_input, ir_indices}, ir_builder);

    auto jit_engine = jit::Engine::Create();
    auto gather_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    std::vector<float> input_vec(ir_tensor_type->Size());
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);  // [1.0, 2.0, ..., 12.0]

    std::vector<int64_t> index_vec(ir_index_type->NormalizeSize());
    index_vec[0] = 0;
    index_vec[1] = 0;

    index_vec[2] = 1;
    index_vec[3] = 1;

    index_vec[4] = 2;
    index_vec[5] = 3;

    auto result = gather_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 1.0f);
    EXPECT_EQ(result[1], 6.0f);
    EXPECT_EQ(result[2], 12.0f);
}

TEST(GaloisTests, TestGather3D) {
    auto ir_tensor_type = ir::f32->Tile(2, 3, 4);    // 2x3x4 tensor
    auto ir_index_type = ir::i64->Tile(3)->Tile(4);  // [3, 4] - 3D coordinates, 4 sample points
    auto ir_output_type = ir::f32->Tile(4);          // [4] output points

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "gather3d_test");

    auto ir_input = ir_operator->inputs[0];
    auto ir_indices = ir_operator->inputs[1];

    auto gather_creator = galois::op::GatherCreator::Create();
    gather_creator->Express({ir_input, ir_indices}, ir_builder);

    auto jit_engine = jit::Engine::Create();
    auto gather_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    // Test data: 2x3x4 = 24 elements
    std::vector<float> input_vec(24);
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);

    // 3D indices: [0,0,0], [0,1,2], [1,0,3], [1,2,1]
    std::vector<int64_t> index_vec = {
        0, 0, 0,  // Point 0: [0,0,0] -> input[0][0][0] = 1
        0, 1, 2,  // Point 1: [0,1,2] -> input[0][1][2] = 7
        1, 0, 3,  // Point 2: [1,0,3] -> input[1][0][3] = 16
        1, 2, 1   // Point 3: [1,2,1] -> input[1][2][1] = 22
    };

    auto result = gather_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 1.0f);   // input[0][0][0] = 1
    EXPECT_EQ(result[1], 7.0f);   // input[0][1][2] = 7
    EXPECT_EQ(result[2], 16.0f);  // input[1][0][3] = 16
    EXPECT_EQ(result[3], 22.0f);  // input[1][2][1] = 22
}

TEST(GaloisTests, TestGather4D) {
    auto ir_tensor_type = ir::f32->Tile(2, 2, 2, 3);  // 2x2x2x3 tensor
    auto ir_index_type = ir::i64->Tile(4)->Tile(3);   // [4, 3] - 4D coordinates, 3 sample points
    auto ir_output_type = ir::f32->Tile(3);           // [3] output points

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "gather4d_test");

    auto ir_input = ir_operator->inputs[0];
    auto ir_indices = ir_operator->inputs[1];

    auto gather_creator = galois::op::GatherCreator::Create();
    gather_creator->Express({ir_input, ir_indices}, ir_builder);

    auto jit_engine = jit::Engine::Create();
    auto gather_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    // Test data: 2x2x2x3 = 24 elements
    std::vector<float> input_vec(24);
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);

    // 4D indices: [0,0,0,0], [0,1,1,2], [1,1,1,1]
    std::vector<int64_t> index_vec = {
        0, 0, 0, 0,  // Point 0: [0,0,0,0] -> input[0][0][0][0] = 1
        0, 1, 1, 2,  // Point 1: [0,1,1,2] -> input[0][1][1][2] = 12
        1, 1, 1, 1   // Point 2: [1,1,1,1] -> input[1][1][1][1] = 23
    };

    auto result = gather_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 1.0f);   // input[0][0][0][0] = 1
    EXPECT_EQ(result[1], 12.0f);  // input[0][1][1][2] = 12
    EXPECT_EQ(result[2], 23.0f);  // input[1][1][1][1] = 23
}

TEST(GaloisTests, TestGatherLargeIndices) {
    auto ir_tensor_type = ir::f32->Tile(5, 5);        // 5x5 matrix
    auto ir_index_type = ir::i64->Tile(2)->Tile(10);  // [2, 10] - 2D coordinates, 10 sample points
    auto ir_output_type = ir::f32->Tile(10);          // [10] output points

    auto ir_builder = ir::Builder::Create();
    auto ir_operator_type =
        ir::OperatorType::Create({ir_tensor_type, ir_index_type}, ir_output_type);
    auto [ir_operator, _] = ir_builder->CreateOperator(ir_operator_type, "gather_large_test");

    auto ir_input = ir_operator->inputs[0];
    auto ir_indices = ir_operator->inputs[1];

    auto gather_creator = galois::op::GatherCreator::Create();
    gather_creator->Express({ir_input, ir_indices}, ir_builder);

    auto jit_engine = jit::Engine::Create();
    auto gather_fun = jit_engine->EmitOperatorSymbol<float *(*)(float *, int64_t *)>(ir_operator);

    // Test data: 5x5 = 25 elements
    std::vector<float> input_vec(25);
    std::iota(input_vec.begin(), input_vec.end(), 1.0f);

    // 10 different 2D coordinates
    std::vector<int64_t> index_vec = {
        0, 0,  // [0,0] -> 1
        0, 4,  // [0,4] -> 5
        1, 1,  // [1,1] -> 7
        2, 2,  // [2,2] -> 13
        3, 3,  // [3,3] -> 19
        4, 4,  // [4,4] -> 25
        4, 0,  // [4,0] -> 21
        3, 1,  // [3,1] -> 17
        2, 3,  // [2,3] -> 14
        1, 4   // [1,4] -> 10
    };

    auto result = gather_fun(input_vec.data(), index_vec.data());

    EXPECT_EQ(result[0], 1.0f);   // input[0][0] = 1
    EXPECT_EQ(result[1], 5.0f);   // input[0][4] = 5
    EXPECT_EQ(result[2], 7.0f);   // input[1][1] = 7
    EXPECT_EQ(result[3], 13.0f);  // input[2][2] = 13
    EXPECT_EQ(result[4], 19.0f);  // input[3][3] = 19
    EXPECT_EQ(result[5], 25.0f);  // input[4][4] = 25
    EXPECT_EQ(result[6], 21.0f);  // input[4][0] = 21
    EXPECT_EQ(result[7], 17.0f);  // input[3][1] = 17
    EXPECT_EQ(result[8], 14.0f);  // input[2][3] = 14
    EXPECT_EQ(result[9], 10.0f);  // input[1][4] = 10
}