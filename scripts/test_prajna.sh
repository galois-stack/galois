set -e

build_dir=build_$1

./$build_dir/bin/galois exe tests/program/main.galois
./$build_dir/bin/galois exe tests/program/hello_world.galoisscript
