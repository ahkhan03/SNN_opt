#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
v05_dir="$(cd "${script_dir}/../../kv260_v05/src" && pwd)"
output_dir="${1:-${script_dir}/work/gate4/bin}"
hls_include="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
if [[ ! -d "${hls_include}" && -d /home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1 ]]; then
    hls_include=/home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1
fi
if [[ ! -d "${hls_include}" ]]; then
    echo "HLS include directory not found: ${hls_include}" >&2
    exit 2
fi
if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi

mkdir -p "${output_dir}"
common=(-O2 -std=c++17 -Wno-unknown-pragmas -pthread
        -I"${hls_include}" -I"${source_dir}" -I"${v05_dir}")

# The v0.5 source remains the oracle; only its compile-time storage extent is
# widened for route-boundary cells.  Its default 64x64 build is untouched.
g++ "${common[@]}" -DMAXN=1024 -DMAXM=1024 -DV05_INPUT_MAX_DIM=1024 \
    "${v05_dir}/native_fixed_v05.cpp" \
    "${v05_dir}/snn_qp_v05_kernel.cpp" \
    -o "${output_dir}/native_fixed_v05"

g++ "${common[@]}" \
    "${source_dir}/native_fixed_v06.cpp" \
    "${source_dir}/snn_qp_v06_kernel.cpp" \
    -o "${output_dir}/native_fixed_v06"

# This binary is used only for the host-side AUTO assertion.  Force the
# in-process BO facade even on a workstation that happens to have XRT headers.
g++ -O2 -std=c++17 -pthread -DV06_FORCE_MOCK \
    -I"${source_dir}" -I"${v05_dir}" \
    "${source_dir}/host_kv260_v06.cpp" \
    -o "${output_dir}/host_kv260_v06"

sha256sum "${output_dir}/native_fixed_v05" \
          "${output_dir}/native_fixed_v06" \
          "${output_dir}/host_kv260_v06"
