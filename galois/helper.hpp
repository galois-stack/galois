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

class ScopeExit {
   public:
    static std::unique_ptr<ScopeExit> Create(std::function<void()> func) {
        auto self = std::unique_ptr<ScopeExit>(new ScopeExit);
        self->_todo = func;
        return self;
    }

    ~ScopeExit() {
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

template <typename DstType_, typename SrcType_>
bool Is(std::weak_ptr<SrcType_> ir_src) {
    return Is<DstType_>(ir_src.lock());
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

template <typename T>
std::shared_ptr<T> Lock(std::weak_ptr<T>& weak) {
    auto ptr = weak.lock();
    GALOIS_ASSERT(ptr);
    return ptr;
}

}  // namespace galois

// For Eigen
template <typename Matrix_>
inline void RemoveRow(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows() - 1;
    unsigned int numCols = matrix.cols();

    if (index < numRows)
        matrix.block(index, 0, numRows - index, numCols) = matrix.bottomRows(numRows - index);

    matrix.conservativeResize(numRows, numCols);
}

template <typename Matrix_>
inline void RemoveColumn(Matrix_& matrix, int64_t index) {
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols() - 1;

    if (index < numCols)
        matrix.block(0, index, numRows, numCols - index) = matrix.rightCols(numCols - index);

    matrix.conservativeResize(numRows, numCols);
}
