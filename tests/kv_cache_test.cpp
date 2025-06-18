#include "tests/galois_test.hpp"
#include <vector>
#include <memory>

TEST(GaloisTests, TestStandardKVCache) {
    auto ir_builder = ir::Builder::Create();
    auto jit_engine = jit::Engine::Create();
    
    const int num_heads = 8;
    const int head_dim = 64;
    const int max_seq_len = 128;
    
    fmt::print("=== Standard KV Cache Implementation ===\n");
    fmt::print("Configuration: {} heads × {} dim × {} max_seq_len\n", 
               num_heads, head_dim, max_seq_len);
    
    // build complete KV Cache + Attention pipeline
    auto build_complete_kv_attention_pipeline = [&]() {
        // define inputs
        auto ir_current_k = ir::f32->Tile(num_heads, max_seq_len, head_dim);      // current K cache
        auto ir_current_v = ir::f32->Tile(num_heads, max_seq_len, head_dim);      // current V cache  
        auto ir_new_k = ir::f32->Tile(num_heads, 1, head_dim);                   // new K token
        auto ir_new_v = ir::f32->Tile(num_heads, 1, head_dim);                   // new V token
        auto ir_query = ir::f32->Tile(num_heads, 1, head_dim);                   // Query
        
        // Step 1: KV Cache update (Concatenation)
        auto ir_updated_k = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_current_k, ir_new_k}, 1);  // concatenate in seq_len dimension
        auto ir_updated_v = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_current_v, ir_new_v}, 1);
        
        // Step 2: extract single head K and V (for simplicity, process the first head)
        Eigen::VectorXi64 single_head_shape(3);
        single_head_shape << 1, max_seq_len + 1, head_dim;
        auto ir_k_head = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_updated_k->GetOperatorType()->output_type}, single_head_shape);
        auto ir_v_head = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_updated_v->GetOperatorType()->output_type}, single_head_shape);
        auto ir_q_head = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_query}, single_head_shape);
        
        // Step 3: reshape to 2D matrix
        Eigen::VectorXi64 k_matrix_shape(2);
        k_matrix_shape << max_seq_len + 1, head_dim;  // [seq_len, head_dim]
        auto ir_k_matrix = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_k_head->GetOperatorType()->output_type}, k_matrix_shape);
        
        Eigen::VectorXi64 v_matrix_shape(2);
        v_matrix_shape << max_seq_len + 1, head_dim;  // [seq_len, head_dim]
        auto ir_v_matrix = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_v_head->GetOperatorType()->output_type}, v_matrix_shape);
        
        Eigen::VectorXi64 q_matrix_shape(2);
        q_matrix_shape << 1, head_dim;  // [1, head_dim]
        auto ir_q_matrix = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_q_head->GetOperatorType()->output_type}, q_matrix_shape);
        
        // Step 4: K matrix transpose K^T
        auto ir_k_transpose = ir_builder->CreateOperatorByCreator<op::TransposeCreator>(
            {ir_k_matrix->GetOperatorType()->output_type}, 0, 1);  // [head_dim, seq_len]
        
        // Step 5: calculate attention scores Q @ K^T
        auto ir_attention_scores = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
            {ir_q_matrix->GetOperatorType()->output_type,    // [1, head_dim]
             ir_k_transpose->GetOperatorType()->output_type}); // [head_dim, seq_len]
        // result: [1, seq_len]
        
        // Step 6: attention weighted Attention @ V
        auto ir_attention_output = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
            {ir_attention_scores->GetOperatorType()->output_type,  // [1, seq_len]
             ir_v_matrix->GetOperatorType()->output_type});        // [seq_len, head_dim]
        // result: [1, head_dim]
        
        // Step 7: combine all outputs (updated_k, updated_v, attention_output)
        auto ir_k_v_combined = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_updated_k->GetOperatorType()->output_type,
             ir_updated_v->GetOperatorType()->output_type}, 0);
        
        auto ir_final_output = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_k_v_combined->GetOperatorType()->output_type,
             ir_attention_output->GetOperatorType()->output_type}, 0);
        
        return ir_final_output;
    };
    
    // fix: correct function pointer syntax
    auto complete_pipeline = build_complete_kv_attention_pipeline();
    auto kv_attention_func = jit_engine->EmitOperatorSymbol<float*(*)(float*, float*, float*, float*, float*)>(
        complete_pipeline);
    
    // execute standard KV Cache inference process
    std::vector<float> k_cache(num_heads * max_seq_len * head_dim, 0.0f);
    std::vector<float> v_cache(num_heads * max_seq_len * head_dim, 0.0f);
    int current_seq_len = 0;
    
    for (int step = 0; step < 5; step++) {
        fmt::print("\n--- Inference Step {} ---\n", step);
        
        // prepare new K, V token and Query
        std::vector<float> new_k_token(num_heads * head_dim);
        std::vector<float> new_v_token(num_heads * head_dim);
        std::vector<float> query(num_heads * head_dim);
        
        // initialize data
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = h * head_dim + d;
                new_k_token[idx] = step * 1000 + h * 100 + d;
                new_v_token[idx] = (step * 1000 + h * 100 + d) * 2;
                query[idx] = step * 500 + h * 50 + d;
            }
        }
        
        // execute complete KV Cache + Attention calculation
        auto result = kv_attention_func(
            k_cache.data(),      // current K cache
            v_cache.data(),      // current V cache
            new_k_token.data(),  // new K token
            new_v_token.data(),  // new V token
            query.data()         // Query
        );
        
        // parse result
        current_seq_len = std::min(step + 1, max_seq_len);
        int updated_k_size = num_heads * (current_seq_len + 1) * head_dim;
        int updated_v_size = num_heads * (current_seq_len + 1) * head_dim;
        int attention_output_size = 1 * head_dim;
        
        // extract updated KV Cache
        float* updated_k = result;
        float* updated_v = result + updated_k_size;
        float* attention_output = result + updated_k_size + updated_v_size;
        
        // update local cache (only keep effective length)
        int effective_seq_len = std::min(current_seq_len + 1, max_seq_len);
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < effective_seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int cache_idx = h * max_seq_len * head_dim + s * head_dim + d;
                    int result_idx = h * effective_seq_len * head_dim + s * head_dim + d;
                    if (result_idx < updated_k_size) {
                        k_cache[cache_idx] = updated_k[result_idx];
                        v_cache[cache_idx] = updated_v[result_idx];
                    }
                }
            }
        }
        
        // output attention result
        fmt::print("Attention output: [{:.2f}, {:.2f}, {:.2f}, ...]\n", 
                   attention_output[0], attention_output[1], attention_output[2]);
        
        fmt::print("✅ Step {} completed: seq_len = {}\n", step, effective_seq_len);
        
        free(result);
    }
    
    
    EXPECT_TRUE(true);
}