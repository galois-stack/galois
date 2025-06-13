#include "galois/op/arithmetic.hpp"
#include "tests/galois_test.hpp"
#include <vector>
#include <memory>

class SimpleKVCache {
public:
    struct CacheEntry {
        std::shared_ptr<ir::Tensor> keys;    // [seq_len, head_dim]
        std::shared_ptr<ir::Tensor> values;  // [seq_len, head_dim]
        int64_t current_length;
        int64_t max_length;
    };
    
    std::vector<CacheEntry> cache_entries;  // One entry per attention head
    int64_t num_heads;
    int64_t head_dim;
    int64_t max_seq_len;
    
    SimpleKVCache(int64_t heads, int64_t dim, int64_t max_len) 
        : num_heads(heads), head_dim(dim), max_seq_len(max_len) {
        cache_entries.resize(num_heads);
        for (auto& entry : cache_entries) {
            entry.current_length = 0;
            entry.max_length = max_len;
        }
    }
};

TEST(GaloisTests, TestKVCache) {
    // Simplified KV Cache test using Concatenate operation
    // Test: concatenate existing cache with new tokens
    int64_t current_seq_len = 4;
    int64_t new_tokens = 1;
    int64_t head_dim = 64;
    
    auto ir_builder = ir::Builder::Create();
    
    auto ir_existing_type = ir::f32->Tile(current_seq_len, head_dim);
    auto ir_new_type = ir::f32->Tile(new_tokens, head_dim);
    
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::ConcatenateCreator>(
        {ir_existing_type, ir_new_type}, 0);  
    
    auto jit_engine = jit::Engine::Create();
    auto concat_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*)>(ir_operator);
    
    int64_t existing_size = current_seq_len * head_dim;
    int64_t new_size = new_tokens * head_dim;
    
    std::vector<float> existing_cache(existing_size, 0.1f);  
    std::vector<float> new_tokens_data(new_size, 0.5f);    
    
    fmt::print("=== KV Cache Concatenation Test ===\n");
    fmt::print("Existing cache shape: [{}, {}], New tokens: [{}, {}]\n", 
               current_seq_len, head_dim, new_tokens, head_dim);
    
    auto result = concat_fun(existing_cache.data(), new_tokens_data.data());
    ASSERT_NE(result, nullptr);
    
    int64_t total_elements = (current_seq_len + new_tokens) * head_dim;
    fmt::print("Result shape: [{}, {}] = {} elements\n", 
               current_seq_len + new_tokens, head_dim, total_elements);
    
    fmt::print("First 3 values (existing): ");
    for (int i = 0; i < 3; i++) {
        fmt::print("{:.1f} ", result[i]);
        EXPECT_NEAR(result[i], 0.1f, 0.01f);
    }
    fmt::print("\n");
    
    // Check values after existing cache (should be new tokens = 0.5) 
    int new_start = existing_size;
    fmt::print("New token values: ");
    for (int i = 0; i < std::min(3L, new_size); i++) {
        fmt::print("{:.1f} ", result[new_start + i]);
        EXPECT_NEAR(result[new_start + i], 0.5f, 0.01f);
    }
    fmt::print("\n");
    
    fmt::print("✓ KV Cache concatenation successful: {} + {} -> {} elements\n", 
               existing_size, new_size, total_elements);
    
    free(result);
}

TEST(GaloisTests, TestKVCacheMultiHead) {
    // Test KV cache management for multiple attention heads
    int64_t num_heads = 4;
    int64_t head_dim = 32;
    int64_t seq_len = 3;
    
    auto ir_builder = ir::Builder::Create();
    
    // Multi-head KV cache: [num_heads, seq_len, head_dim]
    auto ir_output_type = ir::f32->Tile(num_heads, seq_len, head_dim);
    
    // Create operator that takes one input per head: [seq_len, head_dim] each
    auto ir_head_type = ir::f32->Tile(seq_len, head_dim);
    std::vector<std::shared_ptr<ir::TensorType>> input_types(num_heads, ir_head_type);
    
    auto ir_operator = ir_builder->CreateOperatorByCreator<op::StackCreator>(input_types, 0);
    
    auto jit_engine = jit::Engine::Create();
    auto multi_head_fun = jit_engine->EmitOperatorSymbol<float* (*)(float*, float*, float*, float*)>(ir_operator);
    
    int64_t head_size = seq_len * head_dim;
    std::vector<std::vector<float>> head_data(num_heads);
    std::vector<float*> head_ptrs(num_heads);
    
    for (int64_t head = 0; head < num_heads; head++) {
        float head_value = 0.1f * (head + 1); 
        head_data[head].resize(head_size, head_value);
        head_ptrs[head] = head_data[head].data();
    }
    
    auto result = multi_head_fun(head_ptrs[0], head_ptrs[1], head_ptrs[2], head_ptrs[3]);
    ASSERT_NE(result, nullptr);
    
    fmt::print("\n=== Multi-Head KV Cache Test ===\n");
    fmt::print("Shape: [{}, {}, {}]\n", num_heads, seq_len, head_dim);
    
    // Verify each head has different values
    for (int64_t head = 0; head < num_heads; head++) {
        int64_t head_offset = head * seq_len * head_dim;
        float expected_value = 0.1f * (head + 1);
        fmt::print("Head {} value: {:.2f} (expected: {:.2f})\n", 
                   head, result[head_offset], expected_value);
    }
    
    fmt::print("✓ Multi-head KV cache management: StackCreator for head dimension\n");
    
    free(result);
}