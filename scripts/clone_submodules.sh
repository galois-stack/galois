#! /bin/bash
set -e


# 下载子模块
git submodule update --init $@ .

pushd third_party/prajna
bash scripts/clone_submodules.sh $@
popd
