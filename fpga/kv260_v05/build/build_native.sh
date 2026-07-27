#!/usr/bin/env bash
set -euo pipefail

source /tools/Xilinx/Vitis/2022.1/settings64.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
output_dir="${1:-${script_dir}/work/native}"
mkdir -p "${output_dir}"

g++ -O2 -std=c++17 -Wno-unknown-pragmas \
  -I/tools/Xilinx/Vitis_HLS/2022.1/include \
  "${source_dir}/native_fixed_v05.cpp" \
  "${source_dir}/snn_qp_v05_kernel.cpp" \
  -o "${output_dir}/native_fixed_v05"

sha256sum "${output_dir}/native_fixed_v05"
