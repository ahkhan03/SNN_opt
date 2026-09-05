#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
v05_src="$(cd "${script_dir}/../../kv260_v05/src" && pwd)"
output_dir="${1:-${script_dir}/work/host}"
pmsm_src="${PMSM_SRC:-/home/ameer/RD/Research/paper_workspace/snn_pmsm_resident/analysis/offdevice/kv260_res/src}"
mkdir -p "${output_dir}"

# Keep this board command literal in the recipe. The KV260 image supplies
# XRT 2.13 and libxrt_coreutil; the extra /usr/include include root lets both
# packaging layouts resolve <xrt/xrt_*.h>.
board_cxx=(g++ -std=c++17 -O2 -I/usr/include/xrt -I/usr/include
           -I"${source_dir}" -I"${v05_src}")
if [[ -f /usr/include/xrt/xrt_bo.h || -f /usr/include/xrt/xrt/xrt/xrt_bo.h ]]; then
    echo "BOARD: g++ -std=c++17 -O2 -I/usr/include/xrt ... -lxrt_coreutil"
    "${board_cxx[@]}" -o "${output_dir}/host_kv260_v06" \
        "${source_dir}/host_kv260_v06.cpp" -lxrt_coreutil
    "${board_cxx[@]}" -I"${pmsm_src}" \
        -o "${output_dir}/host_kv260_v06_stream" \
        "${source_dir}/host_kv260_v06_stream.cpp" -lxrt_coreutil
    echo "built XRT hosts in ${output_dir}"
else
    echo "XRT headers unavailable; building the in-process mock backend"
    mock_cxx=(g++ -std=c++17 -O2 -DV06_FORCE_MOCK -I"${source_dir}"
              -I"${v05_src}")
    echo "MOCK: g++ -std=c++17 -O2 -DV06_FORCE_MOCK -I${source_dir} ..."
    "${mock_cxx[@]}" -o "${output_dir}/host_kv260_v06" \
        "${source_dir}/host_kv260_v06.cpp"
    "${mock_cxx[@]}" -I"${pmsm_src}" \
        -o "${output_dir}/host_kv260_v06_stream" \
        "${source_dir}/host_kv260_v06_stream.cpp"
    echo "built mock hosts in ${output_dir}"
fi

sha256sum "${output_dir}/host_kv260_v06" \
          "${output_dir}/host_kv260_v06_stream"
