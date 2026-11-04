#!/usr/bin/env bash
set -euo pipefail
CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${CPP_DIR}/build/ax650"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP_MSP_DIR="${BSP_MSP_DIR:-/data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out}"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
cmake -S "$CPP_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-g++" \
  -DAX650=ON -DBSP_MSP_DIR="$BSP_MSP_DIR"
cmake --build "$BUILD_DIR" -j"$(nproc)"
mkdir -p "$CPP_DIR/bin"; cp "$BUILD_DIR/audiosr_ax650" "$CPP_DIR/bin/audiosr_ax650"
"${TOOLCHAIN_ROOT}/bin/aarch64-none-linux-gnu-strip" --strip-unneeded "$CPP_DIR/bin/audiosr_ax650"
echo "built $CPP_DIR/bin/audiosr_ax650"
