#!/usr/bin/env bash
# Build the Bruce engine with nvcc.
#
#   ./build.sh              # defaults to sm_86 (RTX 3060); override for other GPUs:
#   CUDA_ARCH=sm_89 ./build.sh
#   CUDA_ARCH=sm_90 ./build.sh
#
# The engine has only been validated on Windows + sm_86.  On Linux the POSIX I/O
# and socket paths are used; report issues with the exact GPU and driver.
set -euo pipefail

cd "$(dirname "$0")"

ARCH="${CUDA_ARCH:-sm_86}"
NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "nvcc not found (set NVCC=/path/to/nvcc)" >&2
    exit 1
fi

"$NVCC" -O3 -std=c++17 -arch="$ARCH" \
    -Xcompiler -O2 \
    -o bruce \
    src/bruce.cu src/unicode.cpp src/unicode-data.cpp

echo "build ok: $(pwd)/bruce"
