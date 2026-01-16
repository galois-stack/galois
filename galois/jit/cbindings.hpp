#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>

namespace galois {

inline void* auto_aligned_alloc(size_t len) {
    // 可能需要进一步的调整
    std::array<size_t, 7> alignments = {128, 64, 32, 16, 8, 4, 2};
    for (size_t i = 0; i < alignments.size(); ++i) {
        if (len % alignments[i] == 0) {
            void* ptr = std::aligned_alloc(alignments[i], len);
            if (ptr) {
                return ptr;
            }
        }
    }

    return std::malloc(len);
}

}  // namespace galois
