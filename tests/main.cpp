#include <chrono>

#include "galois/galois.hpp"
#include "gtest/gtest.h"

using namespace galois;
using namespace galois::ir;
using namespace std;

extern "C" void peak_gflops(int64_t count);
// the definition of func is written in assembly language
// raw string literal could be very useful
asm(R"(
	.globl	"peak_gflops"               ; -- Begin function OuterProductTest
	.p2align	2
"_peak_gflops":                      ; @"peak_gflops"
		.cfi_startproc
; %bb.0:
		mov	x10, xzr
        // movk w8, #1, lsl #16
LBB1_1:                                 ;
                                        ;
                                        ;
	cmp	x10, x0
	add	x11, x10, #1
	mov	x10, x11

	fmul.4s	v1, v1, v1
	fmul.4s	v2, v2, v2
	fmul.4s	v3, v3, v3
	fmul.4s	v4, v4, v4
    fmul.4s	v5, v5, v5
	fmul.4s	v6, v6, v6
	fmul.4s	v7, v7, v7
	fmul.4s	v8, v8, v8
    fmul.4s	v10, v10, v10
	fmul.4s	v11, v11, v11
	fmul.4s	v12, v12, v12
	fmul.4s	v13, v13, v13
    fmul.4s	v14, v14, v14
	fmul.4s	v15, v15, v15
	fmul.4s	v16, v16, v16
	fmul.4s	v17, v17, v17

	b.lo	LBB1_1
	ret
	.cfi_endproc
)");

TEST(GaloisTests, TestPeakGflops) {
    int64_t count = 1000000000;
    auto t0 = std::chrono::high_resolution_clock::now();
    peak_gflops(count);
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("peak performance: {}g\n", static_cast<double>(count) * 8 * 16 / (t1 - t0).count());
}
