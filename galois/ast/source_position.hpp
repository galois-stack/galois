#pragma once

namespace galois::ast {

struct SourcePosition {
    int line = -1;
    int column = -1;
    std::string file;
};

}  // namespace galois::ast
