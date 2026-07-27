#!/usr/bin/env bash
set -euo pipefail

source /tools/Xilinx/Vitis/2022.1/settings64.sh

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
work_dir="${1:-${script_dir}/work/implementation}"
platform="/tools/build/xsct/kr260_min/export/kr260_min/kr260_min.xpfm"
include_dir="/tools/Xilinx/Vitis_HLS/2022.1/include"
clock_hz="200000000"
dtc_bin="/tools/Xilinx/Vitis/2022.1/bin/dtc"
mkdir -p "${work_dir}"

xo="${work_dir}/snn_qp_v05.xo"
xclbin="${work_dir}/snn_qp_v05.xclbin"
bit="${work_dir}/snn_qp_v05.bit"
bitbin="${work_dir}/snn_qp_v05.bit.bin"
dtbo="${work_dir}/snn_qp_v05.dtbo"

cd "${work_dir}"
v++ -c -t hw --platform "${platform}" -k snn_qp_v05 \
  --hls.clock "${clock_hz}:snn_qp_v05" \
  -I"${include_dir}" -I"${source_dir}" --save-temps \
  -o "${xo}" "${source_dir}/snn_qp_v05_kernel.cpp"

v++ -l -t hw --platform "${platform}" \
  --clock.freqHz "${clock_hz}:snn_qp_v05_1" \
  --config "${script_dir}/connectivity.cfg" \
  --vivado.prop run.impl_1.STRATEGY=Performance_ExplorePostRoutePhysOpt \
  -o "${xclbin}" "${xo}"

xclbinutil --input "${xclbin}" \
  --dump-section "BITSTREAM:RAW:${bit}" --force
(
  cd "${work_dir}"
  bootgen -arch zynqmp -image "${script_dir}/msrp_v05.bif" \
    -o "${bitbin}" -w on
)
"${dtc_bin}" -@ -I dts -O dtb -o "${dtbo}" \
  "${script_dir}/msrp_v05.dts"
printf '%s\n' '{"shell_type":"XRT_FLAT","num_slots":"1"}' \
  > "${work_dir}/shell.json"

sha256sum "${xo}" "${xclbin}" "${bitbin}" "${dtbo}" \
  "${work_dir}/shell.json"
find "${work_dir}/_x/reports/link/imp" -maxdepth 1 -type f \
  -name '*.rpt' -print -exec sha256sum {} \;
