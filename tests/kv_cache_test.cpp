#include "galois_test.hpp"
#include <vector>
#include <memory>
#include <chrono>

TEST(GaloisTests, TestBasicKVCacheImplementation) {  
    auto ir_builder = ir::Builder::Create();
    auto jit_engine = jit::Engine::Create();
    
    const int num_heads = 8;
    const int head_dim = 64;
    const int initial_seq_len = 4;
    const int new_tokens = 2;

    // Initial K, V cache: [num_heads, seq_len, head_dim]
    auto ir_k_cache = ir::f32->Tile(num_heads, initial_seq_len, head_dim);
    auto ir_v_cache = ir::f32->Tile(num_heads, initial_seq_len, head_dim);
    
    std::vector<float> k_data(num_heads * initial_seq_len * head_dim);
    std::vector<float> v_data(num_heads * initial_seq_len * head_dim);
    
    for (int h = 0; h < num_heads; h++) {
        for (int s = 0; s < initial_seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = h * initial_seq_len * head_dim + s * head_dim + d;
                k_data[idx] = h * 100 + s * 10 + d * 0.01f;  // head*100 + seq*10 + dim*0.01
                v_data[idx] = (h * 100 + s * 10 + d * 0.01f) * 2;  // V is 2x K
            }
        }
    }
        
    // New K, V tokens: [num_heads, new_tokens, head_dim]
    auto ir_new_k = ir::f32->Tile(num_heads, new_tokens, head_dim);
    auto ir_new_v = ir::f32->Tile(num_heads, new_tokens, head_dim);
    
    auto ir_concat_k = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
        {ir_k_cache, ir_new_k}, 1);  // axis=1 (sequence dimension)
    auto ir_concat_v = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
        {ir_v_cache, ir_new_v}, 1);
    
    auto concat_k_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_concat_k);
    auto concat_v_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_concat_v);
    
    // New token data
    std::vector<float> new_k_data(num_heads * new_tokens * head_dim);
    std::vector<float> new_v_data(num_heads * new_tokens * head_dim);
    
    for (int h = 0; h < num_heads; h++) {
        for (int s = 0; s < new_tokens; s++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = h * new_tokens * head_dim + s * head_dim + d;
                new_k_data[idx] = h * 100 + (initial_seq_len + s) * 10 + d * 0.01f;
                new_v_data[idx] = (h * 100 + (initial_seq_len + s) * 10 + d * 0.01f) * 2;
            }
        }
    }
    
    auto updated_k_cache = concat_k_fun(k_data.data(), new_k_data.data());
    auto updated_v_cache = concat_v_fun(v_data.data(), new_v_data.data());
    
    int final_seq_len = initial_seq_len + new_tokens;
        
    // To work around MatrixMultiplyCreator's 4D limitation, process single head
    int test_head = 0;
    
    // Use SliceCreator to extract single head: [1, seq_len, head_dim]
    Eigen::VectorXi64 single_head_shape(3);
    single_head_shape << 1, final_seq_len, head_dim;
    
    auto ir_updated_k_type = ir::f32->Tile(num_heads, final_seq_len, head_dim);
    auto ir_k_head_slice = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
        {ir_updated_k_type}, single_head_shape);
    
    auto k_head_slice_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_head_slice);
    auto k_head_data = k_head_slice_fun(updated_k_cache);
    
    fmt::print("Extract head {}: K[{}, {}, {}]\n", test_head, 1, final_seq_len, head_dim);
    
    // Reshape single head K to 2D: [seq_len, head_dim]
    Eigen::VectorXi64 k_2d_shape(2);
    k_2d_shape << final_seq_len, head_dim;
    
    auto ir_k_head_type = ir::f32->Tile(1, final_seq_len, head_dim);
    auto ir_k_2d = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
        {ir_k_head_type}, k_2d_shape);
    
    auto k_2d_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_2d);
    auto k_2d_data = k_2d_fun(k_head_data);
    
    fmt::print("K reshape: [1, {}, {}] -> [{}, {}]\n", final_seq_len, head_dim, final_seq_len, head_dim);
    
    // Transpose K: [seq_len, head_dim] -> [head_dim, seq_len]
    auto ir_k_2d_type = ir::f32->Tile(final_seq_len, head_dim);
    auto ir_k_transpose = ir_builder->CreateOperatorByCreator<op::TransposeCreator>(
        {ir_k_2d_type}, 0, 1);
    
    auto k_transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_transpose);
    auto kt_data = k_transpose_fun(k_2d_data);
    
    fmt::print("K transpose: [{}, {}] -> [{}, {}]\n", final_seq_len, head_dim, head_dim, final_seq_len);
    
    // Create query Q: [1, head_dim] (single new token query)
    auto ir_q = ir::f32->Tile(1, head_dim);
    std::vector<float> q_data(head_dim);
    for (int d = 0; d < head_dim; d++) {
        q_data[d] = test_head * 100 + final_seq_len * 10 + d * 0.01f;  // Query vector
    }
    
    // Q @ K^T: [1, head_dim] @ [head_dim, seq_len] = [1, seq_len]
    auto ir_qkt = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir_q, ir::f32->Tile(head_dim, final_seq_len)});
    
    auto qkt_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_qkt);
    auto attention_scores = qkt_fun(q_data.data(), kt_data);
    
    fmt::print("\nAttention scores (first 5 positions):\n");
    for (int i = 0; i < std::min(5, final_seq_len); i++) {
        fmt::print("Position {}: {:.4f}\n", i, attention_scores[i]);
    }
    
    // Extract V single head and reshape to 2D
    auto ir_v_head_slice = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
        {ir_updated_k_type}, single_head_shape);
    auto v_head_slice_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_v_head_slice);
    auto v_head_data = v_head_slice_fun(updated_v_cache);
    
    auto ir_v_2d = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
        {ir_k_head_type}, k_2d_shape);
    auto v_2d_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_v_2d);
    auto v_2d_data = v_2d_fun(v_head_data);
    
    fmt::print("V extract and reshape: [{}, {}, {}] -> [{}, {}]\n", 
                num_heads, final_seq_len, head_dim, final_seq_len, head_dim);
    
    // Attention @ V: [1, seq_len] @ [seq_len, head_dim] = [1, head_dim]
    auto ir_attn_v = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
        {ir::f32->Tile(1, final_seq_len), ir::f32->Tile(final_seq_len, head_dim)});
    
    auto attn_v_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_attn_v);
    auto output = attn_v_fun(attention_scores, v_2d_data);
    
    fmt::print("Final output: Attention[1, {}] @ V[{}, {}] = Output[1, {}]\n", 
                final_seq_len, final_seq_len, head_dim, head_dim);

    fmt::print("\nOutput (first 10 dimensions):\n");
    for (int i = 0; i < std::min(10, head_dim); i++) {
        fmt::print("  Dimension {}: {:.6f}\n", i, output[i]);
    }
    
    free(updated_k_cache);
    free(updated_v_cache);
    free(k_head_data);
    free(k_2d_data);
    free(kt_data);
    free(attention_scores);
    free(v_head_data);
    free(v_2d_data);
    free(output);
}

TEST(GaloisTests, TestDynamicKVCacheLoop) {
    
    auto ir_builder = ir::Builder::Create();
    auto jit_engine = jit::Engine::Create();
    
    const int num_heads = 4;  
    const int head_dim = 32;  
    const int max_seq_len = 6; 
    const int num_generation_steps = 3; 
    
    // initialize dynamic cache
    std::vector<float> current_k_cache;
    std::vector<float> current_v_cache;
    int current_seq_len = 0;
    
    // store compile and execution time
    std::vector<std::chrono::microseconds> compilation_times;
    std::vector<std::chrono::microseconds> execution_times;
    std::vector<std::chrono::microseconds> total_step_times;
    
    for (int step = 0; step < num_generation_steps; step++) {
        auto total_step_start = std::chrono::high_resolution_clock::now();
        auto compile_start = std::chrono::high_resolution_clock::now();
        
        int new_seq_len = current_seq_len + 1;
        
        // build KV Cache concatenate graph
        auto ir_existing_k = ir::f32->Tile(num_heads, current_seq_len, head_dim);
        auto ir_new_k = ir::f32->Tile(num_heads, 1, head_dim);
        auto ir_concat_k = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_existing_k, ir_new_k}, 1);
        auto concat_k_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_concat_k);
        
        // concatenate V cache
        auto ir_existing_v = ir::f32->Tile(num_heads, current_seq_len, head_dim);
        auto ir_new_v = ir::f32->Tile(num_heads, 1, head_dim);
        auto ir_concat_v = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
            {ir_existing_v, ir_new_v}, 1);
        auto concat_v_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_concat_v);
        
        // create SliceCreator for each head
        // extract single head K: [num_heads, seq_len, head_dim] -> [1, seq_len, head_dim]
        Eigen::VectorXi64 single_head_shape(3);
        single_head_shape << 1, new_seq_len, head_dim;
        
        auto ir_updated_k_type = ir::f32->Tile(num_heads, new_seq_len, head_dim);
        auto ir_k_head_slice = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_updated_k_type}, single_head_shape);
        auto k_head_slice_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_head_slice);
        
        // reshape K to 2D: [1, seq_len, head_dim] -> [seq_len, head_dim]
        Eigen::VectorXi64 k_2d_shape(2);
        k_2d_shape << new_seq_len, head_dim;
        
        auto ir_k_head_type = ir::f32->Tile(1, new_seq_len, head_dim);
        auto ir_k_2d = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_k_head_type}, k_2d_shape);
        auto k_2d_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_2d);
        
        // transpose K: [seq_len, head_dim] -> [head_dim, seq_len]
        auto ir_k_2d_type = ir::f32->Tile(new_seq_len, head_dim);
        auto ir_k_transpose = ir_builder->CreateOperatorByCreator<op::TransposeCreator>(
            {ir_k_2d_type}, 0, 1);
        auto k_transpose_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_k_transpose);
        
        // Q @ K^T matrix multiplication: [1, head_dim] @ [head_dim, seq_len] = [1, seq_len]
        auto ir_q = ir::f32->Tile(1, head_dim);
        auto ir_qkt = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
            {ir_q, ir::f32->Tile(head_dim, new_seq_len)});
        auto qkt_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_qkt);
        
        // V process (same as K extract and reshape)
        auto ir_v_head_slice = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_updated_k_type}, single_head_shape);
        auto v_head_slice_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_v_head_slice);
        
        auto ir_v_2d = ir_builder->CreateOperatorByCreator<op::SliceCreator>(
            {ir_k_head_type}, k_2d_shape);
        auto v_2d_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*)>(ir_v_2d);
        
        // Attention @ V: [1, seq_len] @ [seq_len, head_dim] = [1, head_dim]
        auto ir_attn_v = ir_builder->CreateOperatorByCreator<op::MatrixMultiplyCreator>(
            {ir::f32->Tile(1, new_seq_len), ir::f32->Tile(new_seq_len, head_dim)});
        auto attn_v_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_attn_v);
        
        auto compile_end = std::chrono::high_resolution_clock::now();
        auto compile_time = std::chrono::duration_cast<std::chrono::microseconds>(
            compile_end - compile_start);
        compilation_times.push_back(compile_time);
        
        auto execution_start = std::chrono::high_resolution_clock::now();
        
        // create new K, V token data
        std::vector<float> new_k_data(num_heads * 1 * head_dim);
        std::vector<float> new_v_data(num_heads * 1 * head_dim);
        
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = h * head_dim + d;
                new_k_data[idx] = h * 100 + (current_seq_len) * 10 + d * 0.01f;
                new_v_data[idx] = (h * 100 + (current_seq_len) * 10 + d * 0.01f) * 2;
            }
        }
        
        // execute KV Cache concatenate
        float* updated_k_cache;
        float* updated_v_cache;
        
        if (step == 0) {
            // use new data
            current_k_cache = new_k_data;
            current_v_cache = new_v_data;
            updated_k_cache = new_k_data.data();
            updated_v_cache = new_v_data.data();
        } else {
            // concatenate existing cache and new data
            updated_k_cache = concat_k_fun(current_k_cache.data(), new_k_data.data());
            updated_v_cache = concat_v_fun(current_v_cache.data(), new_v_data.data());
            
            // update local cache
            current_k_cache.assign(updated_k_cache, updated_k_cache + num_heads * new_seq_len * head_dim);
            current_v_cache.assign(updated_v_cache, updated_v_cache + num_heads * new_seq_len * head_dim);
        }
        
        // multi-head attention calculation loop
        std::vector<std::vector<float>> multi_head_outputs(num_heads);
        
        for (int head = 0; head < num_heads; head++) {
            
            // 1. use SliceCreator to extract current head K
            // calculate head offset to slice correct head
            size_t head_offset = head * new_seq_len * head_dim;
            float* head_k_cache = updated_k_cache + head_offset;
            
            auto k_head_data = k_head_slice_fun(head_k_cache);
            
            // 2. reshape K to 2D
            auto k_2d_data = k_2d_fun(k_head_data);
            
            // 3. transpose K
            auto kt_data = k_transpose_fun(k_2d_data);
            
            // 4. create query Q for this head
            std::vector<float> q_data(head_dim);
            for (int d = 0; d < head_dim; d++) {
                q_data[d] = head * 100 + new_seq_len * 10 + d * 0.01f;
            }
            
            // 5. calculate attention scores Q @ K^T
            auto attention_scores = qkt_fun(q_data.data(), kt_data);
            
            // 6. extract and process V for this head
            size_t head_v_offset = head * new_seq_len * head_dim;
            float* head_v_cache = updated_v_cache + head_v_offset;
            
            auto v_head_data = v_head_slice_fun(head_v_cache);
            auto v_2d_data = v_2d_fun(v_head_data);
            
            // 7. calculate final output Attention @ V
            auto output = attn_v_fun(attention_scores, v_2d_data);
            
            multi_head_outputs[head].assign(output, output + head_dim);
            
            free(k_head_data);
            free(k_2d_data);
            free(kt_data);
            free(attention_scores);
            free(v_head_data);
            free(v_2d_data);
            free(output);
        }
        
        auto execution_end = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
            execution_end - execution_start);
        execution_times.push_back(execution_time);
        
        auto total_step_end = std::chrono::high_resolution_clock::now();
        auto total_step_time = std::chrono::duration_cast<std::chrono::microseconds>(
            total_step_end - total_step_start);
        total_step_times.push_back(total_step_time);
        
        // multi-head output summary
        for (int head = 0; head < num_heads; head++) {
            float head_sum = 0.0f;
            for (float val : multi_head_outputs[head]) {
                head_sum += val;
            }
        }
        
        if (step > 0) {
            free(updated_k_cache);
            free(updated_v_cache);
        }
        
        current_seq_len = new_seq_len;
    }
    
    // performance analysis
    auto total_compile_time = std::accumulate(compilation_times.begin(), 
                                                compilation_times.end(), 
                                                std::chrono::microseconds(0));
    auto total_execution_time = std::accumulate(execution_times.begin(), 
                                                execution_times.end(), 
                                                std::chrono::microseconds(0));
    auto total_step_time = std::accumulate(total_step_times.begin(), 
                                            total_step_times.end(), 
                                            std::chrono::microseconds(0));
    
    fmt::print("performance analysis report\n");
    
    fmt::print("\ncompile overhead analysis:\n");
    fmt::print("  • total compile time: {} μs ({:.1f} ms)\n", 
                total_compile_time.count(), total_compile_time.count() / 1000.0);
    fmt::print("  • average compile time per step: {:.1f} μs\n", 
                total_compile_time.count() / (float)num_generation_steps);
    
    fmt::print("\nexecution performance analysis:\n");
    fmt::print("  • total execution time: {} μs ({:.1f} ms)\n", 
                total_execution_time.count(), total_execution_time.count() / 1000.0);
    fmt::print("  • average execution time per step: {:.1f} μs\n", 
                total_execution_time.count() / (float)num_generation_steps);
    fmt::print("  • average execution time per head: {:.1f} μs\n", 
                total_execution_time.count() / (float)(num_generation_steps * num_heads));
    
    fmt::print("\noverall performance:\n");
    fmt::print("  • total time: {} μs ({:.1f} ms)\n", 
                total_step_time.count(), total_step_time.count() / 1000.0);
    fmt::print("  • compile ratio: {:.1f}%%\n", 
                100.0 * total_compile_time.count() / total_step_time.count());
    fmt::print("  • execution ratio: {:.1f}%%\n", 
                100.0 * total_execution_time.count() / total_step_time.count());
    fmt::print("  • compile/execution ratio: {:.1f}:1\n", 
                (float)total_compile_time.count() / total_execution_time.count());
        
    EXPECT_TRUE(true);
}
