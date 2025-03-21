#! /bin/bash
set -e

build_configure=$1
shift 1

cmake . \
-G Ninja \
-B build_${build_configure} \
-DCMAKE_BUILD_TYPE=${build_configure} \
-DCMAKE_INSTALL_PREFIX=build_${build_configure}/install \
-DBOOST_INCLUDE_LIBRARIES="algorithm;variant;optional;fusion;spirit;multiprecision;process;dll" \
-DLLVM_INCLUDE_TESTS=OFF \
-DLLVM_ENABLE_DUMP=ON \
-DLLVM_INCLUDE_BENCHMARKS=OFF \
-DLLVM_OPTIMIZED_TABLEGEN=ON \
-DLLVM_ENABLE_PROJECTS="clang;compiler-rt" \
-DLLVM_ENABLE_ABI_BREAKING_CHECKS=OFF \
-DCOMPILER_RT_SANITIZERS_TO_BUILD="" \
-DLLVM_TOOL_OPT_BUILD=OFF \
-DLLVM_ENABLE_RTTI=ON \
-DBUILD_SHARED_LIBS=OFF \
$@

#llvm和z3的opt存在冲突所, 所以-DLLVM_TOOL_OPT_BUILD=OFF
