#pragma once

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "galois/assert.hpp"

#define RANGE(container) container.begin(), container.end()

#define POSITIONS(token) token.first_position, token.last_position

namespace galois {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class ScopeGuard {
   public:
    static std::unique_ptr<ScopeGuard> Create(std::function<void()> func) {
        auto self = std::unique_ptr<ScopeGuard>(new ScopeGuard);
        self->_todo = func;
        return self;
    }

    ~ScopeGuard() {
        if (_todo) {
            _todo();
        }
    }

   private:
    std::function<void()> _todo;
};

template <typename DstType_, typename SrcType_>
auto Cast(std::shared_ptr<SrcType_> ir_src) -> std::shared_ptr<DstType_> {
    auto ir_dst = std::dynamic_pointer_cast<DstType_>(ir_src);
    return ir_dst;
}

template <typename DstType_, typename SrcType_>
bool Is(std::shared_ptr<SrcType_> ir_src) {
    return Cast<DstType_, SrcType_>(ir_src) != nullptr;
}

template <typename TensorType>
inline auto Clone(TensorType t) -> std::decay_t<TensorType> {
    return t;
}

inline std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

inline bool IsPowerOfTwo(int64_t x) { return (x & (x - 1)) == 0; }

inline void* auto_aligned_alloc(size_t len) {
    // 可用的对齐值，从高到底
    const size_t alignments[] = {32, 16, 8, 4, 2};
    const size_t num_alignments = sizeof(alignments) / sizeof(alignments[0]);

    for (size_t i = 0; i < num_alignments; ++i) {
        size_t alignment = alignments[i];
        // 检查长度是否是对齐值的倍数
        if (len % alignment == 0) {
            void* ptr = std::aligned_alloc(alignment, len);
            if (ptr) {
                return ptr;
            }
        }
    }
    void* ptr = std::malloc(len);
    GALOIS_ASSERT(ptr != nullptr);
    return ptr;
}

}  // namespace galois