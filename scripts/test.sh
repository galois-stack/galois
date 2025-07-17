#!/bin/bash
set -e

build_dir=build_$1

cleanup() {
    if [[ -n "$CONDA_SHLVL" ]]; then
        conda deactivate
    fi
}

trap cleanup EXIT

source ~/miniconda3/etc/profile.d/conda.sh

conda activate torch-2.5

if [[ "$CONDA_DEFAULT_ENV" != "torch-2.5" ]]; then
    echo "Error: Failed to activate Conda environment 'torch-2.5'" >&2
    exit 1
fi

CONDA_ENV="$CONDA_PREFIX"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"

./$build_dir/bin/galois_test "$@"