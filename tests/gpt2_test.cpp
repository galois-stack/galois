#include "tests/galois_test.hpp"
#include "gpt2_helper.hpp"

TEST(GaloisTests, TestGPT2)
{
    GPT2Tokenizer tokenizer("tests/data/gpt2/vocab.json");
    
    std::string text = "Hello how are you";
    std::vector<int> tokens = tokenizer.Encode(text);
    std::string decoded_text = tokenizer.Decode(tokens);
    ASSERT_EQ(decoded_text, text);
}