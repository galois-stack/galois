#include <chrono>

#include "benchmark/benchmark.h"
#include "galois/galois.hpp"
#include "gtest/gtest.h"

using namespace galois;
using namespace galois::ir;
using namespace std;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (::testing::UnitTest::GetInstance()->Run() != 0) {
        return 1;
    }

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
