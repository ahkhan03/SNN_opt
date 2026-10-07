#!/usr/bin/env bash
set -euo pipefail
if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then source /tools/Xilinx/Vitis/2022.1/settings64.sh; fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/../src"
work_dir="${1:-${script_dir}/work/implementation}"
platform="${PLATFORM:-/tools/build/xsct/kr260_min/export/kr260_min/kr260_min.xpfm}"
include_dir="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
clock_hz="${CLOCK_HZ:-200000000}"
mkdir -p "$work_dir"; cd "$work_dir"
# single resident v07 top. If this clock misses timing, rerun with CLOCK_HZ=125000000.
v++ -c -t hw --platform "$platform" -k snn_qp_v07 --hls.clock "${clock_hz}:snn_qp_v07" \
  -I"$include_dir" -I"$source_dir" --save-temps -o snn_qp_v07.xo "$source_dir/snn_qp_v07_kernel.cpp"
v++ -l -t hw --platform "$platform" --clock.freqHz "${clock_hz}:snn_qp_v07_1" \
  --config "$script_dir/connectivity.cfg" --vivado.prop run.impl_1.STRATEGY=Performance_ExplorePostRoutePhysOpt \
  -o snn_qp_v07.xclbin snn_qp_v07.xo
kria_clock="${KRIA_CLOCK:-kria-clock}"
command -v "${kria_clock}" >/dev/null 2>&1 || { echo "kria-clock not found; set KRIA_CLOCK to the tool that derives the dtbo rate" >&2; exit 1; }
timing_rpts=()
while IFS= read -r rpt; do timing_rpts+=("$rpt"); done < <(find _x -name '*_timing_summary_routed.rpt' 2>/dev/null || true)
if [[ ${#timing_rpts[@]} -gt 1 ]]; then
  echo "kria-clock: expected one routed timing summary under _x, found ${#timing_rpts[@]}" >&2
  exit 1
fi
pack_timing=()
[[ ${#timing_rpts[@]} -eq 1 ]] && pack_timing=(--timing "${timing_rpts[0]}")
"${kria_clock}" pack --xclbin snn_qp_v07.xclbin --firmware-name snn_qp_v07.bit.bin "${pack_timing[@]}"
sha256sum snn_qp_v07.xo snn_qp_v07.xclbin
