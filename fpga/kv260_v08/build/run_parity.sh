#!/usr/bin/env bash
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
fleet_dir="${FLEET_DIR:-}"
if [[ -z "${fleet_dir}" || ! -d "${fleet_dir}" ]]; then
    echo "FLEET_DIR must be a directory of fleet_n*_it2000.bin bundles" >&2
    exit 2
fi
shopt -s nullglob
bundles=("${fleet_dir}"/fleet_n*_it2000.bin)
if [[ ${#bundles[@]} -eq 0 ]]; then
    echo "no fleet_n*_it2000.bin bundles in ${fleet_dir}" >&2
    exit 2
fi
IFS=$'\n'
bundles=($(printf '%s\n' "${bundles[@]}" | sort -V))
unset IFS

if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi

work_dir="${PARITY_WORK_DIR:-${script_dir}/work/parity}"
mkdir -p "${work_dir}"
exe="${work_dir}/parity_v08_vs_v06"

g++ -O2 -std=c++17 -Wno-unknown-pragmas -pthread \
    -I"${hls_include}" -I"${source_dir}" -I"${v06_dir}" -I"${v05_dir}" \
    "${source_dir}/parity_v08_vs_v06.cpp" \
    "${v06_dir}/snn_qp_v06_kernel.cpp" \
    "${source_dir}/snn_qp_v08_kernel.cpp" \
    -o "${exe}"

log_path="${PARITY_LOG:-${work_dir}/parity_log.txt}"
mkdir -p "$(dirname "${log_path}")"
set +e
"${exe}" "${bundles[@]}" 2>&1 | tee "${log_path}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
