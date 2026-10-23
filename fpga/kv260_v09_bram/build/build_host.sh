#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
v05_src="$(cd "${script_dir}/../../kv260_v05/src" && pwd)"
output_dir="${1:-${script_dir}/work/host}"
built=("${output_dir}/host_kv260_v09b")
mkdir -p "${output_dir}"

# Keep this board command literal in the recipe. The KV260 image supplies
# XRT 2.13 and libxrt_coreutil; the extra /usr/include include root lets both
# packaging layouts resolve <xrt/xrt_*.h>.
board_cxx=(g++ -std=c++17 -Wall -O2 -pthread -I/usr/include/xrt -I/usr/include
           -I"${source_dir}" -I"${v05_src}")
if [[ -f /usr/include/xrt/xrt_bo.h || -f /usr/include/xrt/xrt/xrt_bo.h ]]; then
    echo "BOARD: g++ -std=c++17 -Wall -O2 -pthread -I/usr/include/xrt ... -lxrt_coreutil"
    "${board_cxx[@]}" -o "${output_dir}/host_kv260_v09b" \
        "${source_dir}/host_kv260_v09b.cpp" -lxrt_coreutil
    echo "built XRT host in ${output_dir}"
else
    echo "XRT headers unavailable; building the in-process mock backend"
    mock_cxx=(g++ -std=c++17 -Wall -O2 -pthread -DV09B_FORCE_MOCK
              -I"${source_dir}" -I"${v05_src}")
    echo "MOCK: g++ -std=c++17 -Wall -O2 -pthread -DV09B_FORCE_MOCK -I${source_dir} ..."
    "${mock_cxx[@]}" -o "${output_dir}/host_kv260_v09b" \
        "${source_dir}/host_kv260_v09b.cpp"
    echo "built mock host in ${output_dir}"
fi

echo "stream client is not part of this package"
sha256sum "${built[@]}"
