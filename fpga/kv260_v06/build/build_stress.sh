#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
v05_dir="$(cd "${script_dir}/../../kv260_v05/src" && pwd)"
output_dir="${1:-${script_dir}/work/stress}"
hls_include="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
if [[ ! -d "${hls_include}" && -d /home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1 ]]; then
    hls_include=/home/ameer/RD/dev_projects/platforms/kria/hls_include/2022.1
fi
if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi
mkdir -p "${output_dir}"

common=(-O2 -std=c++17 -Wno-unknown-pragmas -pthread
        -I/usr/include/xrt -I/usr/include
        -I"${source_dir}" -I"${v05_dir}")
sources=("${source_dir}/stress_kv260_v06.cpp")
defines=()
libraries=()

if [[ -f /usr/include/xrt/xrt_bo.h || -f /usr/include/xrt/xrt/xrt_bo.h ]]; then
    # Board build: XRT supplies the real Session backend.  The untouched v0.5
    # kernel is linked as the independent raw/telemetry oracle.
    if [[ ! -d "${hls_include}" ]]; then
        echo "HLS include directory not found: ${hls_include}" >&2
        exit 2
    fi
    common+=("-I${hls_include}")
    defines+=(-DV06_STRESS_WITH_V05 -DMAXN=1024 -DMAXM=1024)
    sources+=("${v05_dir}/snn_qp_v05_kernel.cpp")
    libraries+=(-lxrt_coreutil)
else
    # Workstation build: retain the XRT-shaped mock.  If HLS headers are
    # available, also link both native kernels so --mock checks exact parity;
    # otherwise the session's deterministic fallback record still exercises
    # sequence/wrap/mix/STOP logic.
    defines+=(-DV06_FORCE_MOCK)
    if [[ -d "${hls_include}" ]]; then
        common+=("-I${hls_include}")
        defines+=(-DV06_STRESS_WITH_V05 -DV06_MOCK_NATIVE_HOOK -DMAXN=1024 -DMAXM=1024)
        sources+=("${source_dir}/snn_qp_v06_kernel.cpp"
                  "${v05_dir}/snn_qp_v05_kernel.cpp")
    fi
fi

echo "g++ ${defines[*]} ${common[*]} ..."
g++ "${common[@]}" "${defines[@]}" "${sources[@]}" "${libraries[@]}" \
    -o "${output_dir}/stress_kv260_v06"
sha256sum "${output_dir}/stress_kv260_v06"
