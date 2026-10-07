#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../../.." && pwd)"
source_dir="${script_dir}/../src"
v05_dir="${root_dir}/fpga/kv260_v05/src"
hls_include="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
if [[ ! -d "${hls_include}" ]]; then
    echo "HLS include directory not found: ${hls_include} (set HLS_INCLUDE to the Vitis HLS 2022.1 include dir)" >&2
    exit 2
fi
if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi

work_dir="${PARITY_WORK_DIR:-${script_dir}/work/parity}"
mkdir -p "${work_dir}"
exe="${work_dir}/parity_v07_vs_v05"
# Optional: set PMSM_ROOT to a PMSM resident fixture tree to add its cells.
pmsm_root="${PMSM_ROOT:-}"
pmsm_src="${pmsm_root}/src"
include_pmsm=()
if [[ -d "${pmsm_src}" ]]; then include_pmsm=(-I"${pmsm_src}"); fi

g++ -O2 -std=c++17 -Wno-unknown-pragmas -pthread \
    -I"${hls_include}" -I"${source_dir}" -I"${v05_dir}" \
    "${include_pmsm[@]}" \
    "${source_dir}/parity_v07_vs_v05.cpp" \
    "${v05_dir}/snn_qp_v05_kernel.cpp" \
    "${source_dir}/snn_qp_v07_kernel.cpp" \
    -o "${exe}"

log_path="${PARITY_LOG:-${work_dir}/parity_log.txt}"
mkdir -p "$(dirname "${log_path}")"
if [[ -d "${pmsm_src}" ]]; then
    set +e
    "${exe}" "${pmsm_root}" 2>&1 | tee "${log_path}"
    rc=${PIPESTATUS[0]}
    set -e
else
    set +e
    "${exe}" 2>&1 | tee "${log_path}"
    rc=${PIPESTATUS[0]}
    set -e
fi
exit "${rc}"
