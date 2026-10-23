#!/usr/bin/env bash
set -euo pipefail

if [[ -f /tools/Xilinx/Vitis/2022.1/settings64.sh ]]; then
    # shellcheck disable=SC1091
    source /tools/Xilinx/Vitis/2022.1/settings64.sh
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$(cd "${script_dir}/../src" && pwd)"
work_dir="${1:-${script_dir}/work/implementation}"
platform="${PLATFORM:-/tools/build/xsct/kr260_min/export/kr260_min/kr260_min.xpfm}"
include_dir="${HLS_INCLUDE:-/tools/Xilinx/Vitis_HLS/2022.1/include}"
clock_hz="200000000"
dtc_bin="${DTC:-/tools/Xilinx/Vitis/2022.1/bin/dtc}"
mkdir -p "${work_dir}"
# The prebuilt platform must have the property from platform_config.tcl:
# CONFIG.PSU__MAXIGP0__DATA_WIDTH {32}.  It is a platform-build setting, not
# a link-time v++ option, so keep the required recipe beside this kernel.

xo="${work_dir}/snn_qp_v09b.xo"
xclbin="${work_dir}/snn_qp_v09b.xclbin"
bit="${work_dir}/snn_qp_v09b.bit"
bitbin="${work_dir}/snn_qp_v09b.bit.bin"
dtbo="${work_dir}/snn_qp_v09b.dtbo"
cd "${work_dir}"
v++ -c -t hw --platform "${platform}" -k snn_qp_v09b \
    --hls.clock "${clock_hz}:snn_qp_v09b" \
    --hls.pre_tcl "${script_dir}/hls_pre.tcl" \
    -I"${include_dir}" -I"${source_dir}" --save-temps \
    -o "${xo}" "${source_dir}/snn_qp_v09b_kernel.cpp"
v++ -l -t hw --platform "${platform}" \
    --clock.freqHz "${clock_hz}:snn_qp_v09b_1" \
    --config "${script_dir}/connectivity.cfg" \
    --vivado.prop run.impl_1.STRATEGY=Performance_ExplorePostRoutePhysOpt \
    -o "${xclbin}" "${xo}"
xclbinutil --input "${xclbin}" --dump-section "BITSTREAM:RAW:${bit}" --force
if [[ -f "${script_dir}/snn_qp_v09b.bif" ]]; then
    (cd "${work_dir}" && bootgen -arch zynqmp -image "${script_dir}/snn_qp_v09b.bif" \
        -o "${bitbin}" -w on)
fi
kria_clock="${KRIA_CLOCK:-kria-clock}"
command -v "${kria_clock}" >/dev/null 2>&1 || { echo "kria-clock not found; set KRIA_CLOCK to the tool that derives the dtbo rate" >&2; exit 1; }
if [[ -f "${bitbin}" ]]; then
    "${kria_clock}" pack --xclbin "${xclbin}" --firmware-name snn_qp_v09b.bit.bin --bitstream "${bitbin}"
else
    "${kria_clock}" pack --xclbin "${xclbin}" --firmware-name snn_qp_v09b.bit.bin
fi
if [[ -x "${dtc_bin}" ]]; then
    "${dtc_bin}" -@ -I dts -O dtb -o "${dtbo}" "${work_dir}/snn_qp_v09b.dts"
fi
printf '%s\n' '{"shell_type":"XRT_FLAT","num_slots":"1"}' > "${work_dir}/shell.json"
sha256sum "${xo}" "${xclbin}" "${bitbin}" "${dtbo}" "${work_dir}/shell.json" 2>/dev/null || true
