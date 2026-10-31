#!/usr/bin/env bash
# Build libcutlass_fp8.so for GB10 (sm_121a). Usage: build.sh CUTLASS_DIR [OUT_SO] [extra nvcc flags, e.g. -DFC1_PP=1 ...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cutlass=${1:?CUTLASS checkout}
out=${2:-$here/libcutlass_fp8.so}
shift $(( $# < 2 ? $# : 2 ))
nvcc=${NVCC:-/home/riomus/toolchains/cuda-13.3-complete/bin/nvcc}
# the -complete toolchain lacks CCCL (nv/target); take it from the matching 13.3 root
cccl=${CCCL_INC:-/home/riomus/toolchains/cuda133-root/usr/local/cuda-13.3/targets/sbsa-linux/include/cccl}
nice -n 10 "$nvcc" -O3 -std=c++17 -shared -Xcompiler -fPIC --expt-relaxed-constexpr \
  -gencode arch=compute_121a,code=sm_121a -DNDEBUG "$@" \
  -I"$cutlass/include" -I"$cutlass/tools/util/include" -I"$cccl" \
  "$here/cutlass_fp8.cu" -o "$out"
