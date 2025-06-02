#include "galois_test.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct GPT2Config {
    int n_layers = 12;
    int n_heads = 12;
    int n_embd = 768;
    int vocab_size = 50257;
    int block_size = 1024;
    int bias = true;
};

class GPT2Tokenizer {
public:
    GPT2Tokenizer() = default;
    GPT2Tokenizer(const std::string& vocab_path){
        LoadVocab(vocab_path);
    };

    void LoadVocab(const std::string& vocab_path)
    {
        std::ifstream vocab_stream(vocab_path);
        GALOIS_ASSERT(vocab_stream.is_open());
        
        json vocab_json;
        vocab_stream >> vocab_json;
        vocab_stream.close();

        for (auto& [token, id] : vocab_json.items()) {
            vocab[token] = id.get<int>();
            inverse_vocab[id.get<int>()] = token;
        }
    };

    std::vector<int> Encode(const std::string& text)
    {
        std::vector<int> tokens;
        
        auto it = vocab.find(text);
        if(it != vocab.end())
        {
            tokens.push_back(it->second);
            return tokens;
        }
        
        std::string space_prefixed = "Ġ" + text;
        it = vocab.find(space_prefixed);
        if(it != vocab.end())
        {
            tokens.push_back(it->second);
            return tokens;
        }

        std::istringstream iss(text);
        std::string word;
        bool is_first_word = true;
        while(iss >> word)
        {
            std::string token_to_find = word;

            if(!is_first_word)
            {
                token_to_find = "Ġ" + token_to_find;
            }
            is_first_word = false;

            auto word_it = vocab.find(token_to_find);
            if(word_it != vocab.end())
            {
                tokens.push_back(word_it->second);
            }
            else
            {
                word_it = vocab.find(word);
                if(word_it != vocab.end())
                {
                    tokens.push_back(word_it->second);
                }
                else
                {
                    tokens.push_back(unk_token_id);
                }
            }
        }

        return tokens;
    };

    std::string Decode(const std::vector<int>& tokens)
    {
        std::string result;
        bool is_first_token = true;

        for(int token_id : tokens){
            auto it = inverse_vocab.find(token_id);
            if(it != inverse_vocab.end())
            {
                std::string token = it->second;

                if(token == "<|endoftext|>" || token == "<|startoftext|>") continue;

                std::string decoded_token = DecodeToken(token);

                if(is_first_token)
                {
                    if(!decoded_token.empty() && decoded_token[0] == ' ')
                    {
                        decoded_token = decoded_token.substr(1);
                    }
                    is_first_token = false;
                }

                result += decoded_token;
            }
        }

        return result;
    };
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inverse_vocab;
    int bos_token_id = 50256;
    int eos_token_id = 50257;
    int unk_token_id = 50256;

    std::unordered_map<std::string, char> byte_decoder;

    std::string DecodeToken(const std::string& token)
    {
        if (!token.empty() && static_cast<unsigned char>(token[0]) == 0xC4 && 
        token.length() > 1 && static_cast<unsigned char>(token[1]) == 0xA0)
        {
            std::string result = " ";
            if(token.length() > 2)
            {
                result += token.substr(2);
            }
            return result;
       }

       if(token.length() > 0)
       {
        if(token[0] == '\xC4' || token.find("Ġ") == 0)
        {
            std::string result = " ";
            size_t start_pos = 1;
            if(token.length() > 1 && token[0] == '\xC4' && token[1] == '\xA0')
            {
                start_pos = 2;
            }
            if(token.length() > start_pos)
            {
                result += token.substr(start_pos);
            }
            return result;
        }
       }

       return token;
    }
};