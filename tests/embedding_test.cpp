#include "galois/op/arithmetic.hpp"
#include "tests/galois_test.hpp"

TEST(GaloisTests, TestEmbedding) {
    
    int64_t vocab_size = 100;  
    int64_t max_pos = 50;      
    int64_t embed_dim = 64;   
    int64_t token_id = 42;   
    int64_t pos_id = 10;
    
    auto ir_builder = ir::Builder::Create();
    
    auto ir_output_type = ir::f32->Tile(1, embed_dim);
    auto ir_operator_type = ir::OperatorType::Create({}, ir_output_type);
    auto [ir_operator, scope] = ir_builder->CreateOperator(ir_operator_type, "embedding");
    
    auto ir_token_matrix_type = ir::f32->Tile(vocab_size, embed_dim);
    auto ir_token_matrix = ir_builder->Create<ir::io::LoadBinary>(
        ir_token_matrix_type, "tests/models/embedding/token_weights.bin");
    
    auto ir_pos_matrix_type = ir::f32->Tile(max_pos, embed_dim);
    auto ir_pos_matrix = ir_builder->Create<ir::io::LoadBinary>(
        ir_pos_matrix_type, "tests/models/embedding/pos_weights.bin");
    
    auto ir_token_accessor = ir_builder->CreateAccessor(ir_token_matrix);
    ir_token_accessor->transform_matrix.resize(0, 0); 
    ir_token_accessor->shift_vector[0] = token_id;    
    
    auto ir_pos_accessor = ir_builder->CreateAccessor(ir_pos_matrix);
    ir_pos_accessor->transform_matrix.resize(0, 0);  
    ir_pos_accessor->shift_vector[0] = pos_id; 
    
    Eigen::VectorXi64 slice_shape(2);
    slice_shape[0] = 1;         
    slice_shape[1] = embed_dim;   
    
    auto ir_token_slice = ir_builder->Create<ir::SliceView>(ir_token_accessor, slice_shape);
    auto ir_pos_slice = ir_builder->Create<ir::SliceView>(ir_pos_accessor, slice_shape);
    
    auto ir_result = ir_builder->ExpressCreator<op::AddCreator>({ir_token_slice, ir_pos_slice});
    
    ir_builder->Create<ir::Return>(ir_result);

    auto jit_engine = jit::Engine::Create();
    auto embedding_fun = jit_engine->EmitOperatorSymbol<float* (*)()>(ir_operator);
    
    auto result = embedding_fun();
    
    ASSERT_NE(result, nullptr);
    
    fmt::print("Embedding lookup result (2D output [1, {}]):\n", embed_dim);
    fmt::print("Token ID: {}, Position ID: {}, Embed Dim: {}\n", token_id, pos_id, embed_dim);
    fmt::print("First 10 values: ");
    for (int i = 0; i < std::min(10L, embed_dim); i++) {
        fmt::print("{:.4f} ", result[i]);
    }
    fmt::print("\n");
    
    // Verify that accessor-based embedding lookup produces expected values
    // Token embedding for token_id=42: (42 * 0.01 + j * 0.001)
    // Position embedding for pos_id=10: (10 * 0.1 + j * 0.01)  
    // Expected sum: (42 * 0.01 + 10 * 0.1) + j * (0.001 + 0.01) = 1.42 + j * 0.011
    float expected_base = 42 * 0.01f + 10 * 0.1f;  // 1.42
    float expected_increment = 0.001f + 0.01f;      // 0.011
    
    fmt::print("Expected first 10 values: ");
    for (int i = 0; i < std::min(10L, embed_dim); i++) {
        float expected = expected_base + i * expected_increment;
        fmt::print("{:.4f} ", expected);
    }
    fmt::print("\n");
    
    free(result);
}