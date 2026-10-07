#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/../src"
out="${1:-${script_dir}/work/host}"
mkdir -p "$out"
g++ -std=c++17 -O2 -I"$source_dir" "$source_dir/host_kv260_v07.cpp" -o "$out/host_kv260_v07"
if [[ -f /usr/include/xrt/xrt_bo.h || -f /usr/include/xrt/xrt/xrt_bo.h ]]; then
  g++ -std=c++17 -O2 -I/usr/include/xrt -I/usr/include -I"$source_dir" -I"${script_dir}/../../kv260_v05/src" \
      "$source_dir/host_kv260_v07_xrt.cpp" -o "$out/host_kv260_v07_xrt" -lxrt_coreutil
  echo "built XRT host: $out/host_kv260_v07_xrt"
else
  echo "XRT headers unavailable; descriptor checker built (board host deferred to Vitis VM)"
fi
sha256sum "$out/host_kv260_v07"
