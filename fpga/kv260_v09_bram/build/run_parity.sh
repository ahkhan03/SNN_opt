#!/usr/bin/env bash
# In-cap bit-identity gate: v09b against the untouched v06 kernel.
# Bundles are named explicitly. The fleet directory also holds n=160 and
# n=320, which are over this kernel's cap and are not part of the gate.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../../.." && pwd)"
source_dir="${script_dir}/../src"
v06_dir="${root_dir}/fpga/kv260_v06/src"
v05_dir="${root_dir}/fpga/kv260_v05/src"
hls_include="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
if [[ ! -d "${hls_include}" ]]; then
    echo "HLS include directory not found: ${hls_include} (set HLS_INCLUDE)" >&2
    exit 2
fi

fix_dir="${root_dir}/fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures"
# Fleet bundles are not in this tree. Pass the directory that holds
# fleet_n10_it2000.bin and fleet_n40_it2000.bin. Do not glob every fleet
# bundle in that directory: larger n values are a later kernel.
if [[ -z "${FLEET_DIR:-}" ]]; then
    echo "FLEET_DIR is required (directory of fleet_n10_it2000.bin and fleet_n40_it2000.bin)" >&2
    exit 2
fi
fleet_dir="${FLEET_DIR}"
square_dir="${script_dir}/../evidence/fixtures"
bundles=(
    "${fix_dir}/prob_6x18.bin"
    "${fix_dir}/prob_20x60.bin"
    "${fix_dir}/prob_64x64.bin"
    "${fleet_dir}/fleet_n10_it2000.bin"
    "${fleet_dir}/fleet_n40_it2000.bin"
    "${square_dir}/square_T16_K2000.bin"
    "${square_dir}/square_T32_K2000.bin"
    "${square_dir}/square_T64_K2000.bin"
    "${square_dir}/square_T80_K2000.bin"
)
for bundle in "${bundles[@]}"; do
    if [[ ! -f "${bundle}" ]]; then
        echo "missing bundle: ${bundle}" >&2
        exit 2
    fi
done

if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi

work_dir="${PARITY_WORK_DIR:-${script_dir}/work/parity}"
mkdir -p "${work_dir}"
exe="${work_dir}/parity_v09b_vs_v06"

g++ -O2 -std=c++17 -Wno-unknown-pragmas -pthread \
    -I"${hls_include}" -I"${source_dir}" -I"${v06_dir}" -I"${v05_dir}" \
    "${source_dir}/parity_v09b_vs_v06.cpp" \
    "${v06_dir}/snn_qp_v06_kernel.cpp" \
    "${source_dir}/snn_qp_v09b_kernel.cpp" \
    -o "${exe}"

log_path="${PARITY_LOG:-${work_dir}/parity_log.txt}"
mkdir -p "$(dirname "${log_path}")"
set +e
"${exe}" "${bundles[@]}" 2>&1 | tee "${log_path}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
