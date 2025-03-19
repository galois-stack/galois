#pragma once

#include <chrono>

#include "benchmark/benchmark.h"
#include "galois/galois.hpp"
#include "gtest/gtest.h"

using namespace galois;
using namespace galois::ir;
using namespace std;

namespace Eigen {
typedef Matrix<float, -1, -1, Eigen::RowMajor || Eigen::Aligned16> MatrixRXf32;
typedef Matrix<int8_t, -1, -1, Eigen::RowMajor || Eigen::Aligned16> MatrixRXi8;
}  // namespace Eigen
