#! /bin/bash
set -e

build_dir=build_$1

# ./$build_dir/bin/galois_test $@
CONDA_ENV="/home/wanrui/miniconda3/envs/torch-2.5"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${CONDA_ENV}/lib/python3.12/site-packages"


gdb ./$build_dir/bin/galois_test $@
