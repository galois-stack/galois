#pragma once

#include <chrono>

#include "galois/galois.hpp"
#include "galois/graph/graph.hpp"
// #include "galois/op/affine_convertor.hpp"
#include "galois/op/op.hpp"
// #include "galois/optimization/gemm_optimizer.hpp"
#include "gtest/gtest.h"
#include "prajna/bindings/core.hpp"
#include "prajna/jit/execution_engine.h"
#include "thpool.h"

using namespace galois;
using namespace galois::ir;
using namespace std;

namespace Eigen {
typedef Matrix<float, -1, -1, Eigen::RowMajor || Eigen::Aligned16> MatrixRXf;
}

inline std::shared_ptr<prajna::Compiler> CreateCompiler() {
    auto prajna_compiler = prajna::Compiler::Create(false);
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_init),
                                               "thpool_init");
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_add_work),
                                               "thpool_add_work");
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_wait),
                                               "thpool_wait");
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(thpool_destroy),
                                               "thpool_destroy");

    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(aligned_alloc),
                                               "aligned_alloc");
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(malloc), "malloc");
    prajna_compiler->jit_engine->BindCFunction(reinterpret_cast<void *>(free), "free");

    return prajna_compiler;
}
