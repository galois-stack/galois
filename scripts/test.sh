#! /bin/bash
set -e

build_dir=build_$1

./$build_dir/bin/galois_test $@
