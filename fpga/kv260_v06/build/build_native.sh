#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
output_dir="${1:-${script_dir}/work/native}"
hls_include="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
if [[ ! -d "${hls_include}" && -d /home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1 ]]; then
    hls_include=/home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1
fi
if [[ ! -d "${hls_include}" ]]; then
    echo "HLS include directory not found: ${hls_include}" >&2
    exit 2
fi

# The VM has Vitis headers without necessarily having the Vitis shell setup;
# source settings only when the file is present.
if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi

mkdir -p "${output_dir}"
g++ -O2 -std=c++17 -Wno-unknown-pragmas -pthread \
    -I"${hls_include}" -I"${source_dir}" \
    "${source_dir}/native_fixed_v06.cpp" \
    "${source_dir}/snn_qp_v06_kernel.cpp" \
    -o "${output_dir}/native_fixed_v06"
sha256sum "${output_dir}/native_fixed_v06"
