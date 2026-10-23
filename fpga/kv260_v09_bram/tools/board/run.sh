#!/usr/bin/env bash
# Byte-identity runner for the v09b board host.
#
# Required:
#   XCLBIN      board xclbin
#   HOST        board host binary (XRT build, not the mock)
#   FLEET_DIR   directory containing fleet_n10_it2000.bin and fleet_n40_it2000.bin
#   CLOCK_JSON  kria-clock --json-out file whose verdict is OK
#
# Optional overrides (defaults are relative to this tree):
#   FIX_DIR SQUARE_DIR RECORDS RESULTS
#
# Identity cells are oneshot and persistent, HOST_X0, warmups 0, reps 1,
# for every fixture this kernel accepts (n and m at most 128). Fleet n=160
# and n=320 are skipped. CG and STREAM are not cells: the kernel returns
# ERR_CAPACITY for those routes. A three-solve RESIDENT_WARM run on 6x18
# checks exit status only. Latency runs are not part of this script.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pkg="$(cd "${script_dir}/../.." && pwd)"
root="$(cd "${pkg}/../.." && pwd)"

: "${XCLBIN:?set XCLBIN to the board xclbin}"
: "${HOST:?set HOST to the board host binary}"
: "${FLEET_DIR:?set FLEET_DIR to the fleet bundle directory}"
: "${CLOCK_JSON:?set CLOCK_JSON to the kria-clock json-out file}"

FIX_DIR="${FIX_DIR:-${root}/fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures}"
SQUARE_DIR="${SQUARE_DIR:-${root}/fpga/kv260_v09_bram/evidence/fixtures}"
RECORDS="${RECORDS:-${pkg}/evidence/native_records}"
RESULTS="${RESULTS:-${pkg}/build/work/board_identity}"
MAX_N=128

mkdir -p "${RESULTS}"
cmp_py="${script_dir}/cmp_out.py"
fails=0

# grep -q exits on the first match, so under pipefail the SIGPIPE-killed nm
# fails the pipeline even for a genuine XRT host. Read the symbols first.
host_symbols="$(nm -C "${HOST}" 2>/dev/null || true)"
grep -q -i "xrt::" <<<"${host_symbols}" || { echo HOST_IS_MOCK; exit 1; }
python3 - "${CLOCK_JSON}" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
if doc.get("verdict") != "OK":
    sys.stderr.write("CLOCK_NOT_OK verdict=%s\n" % doc.get("verdict"))
    sys.exit(1)
PY

note_fail() {
    echo "FAIL $*"
    fails=$((fails + 1))
}

identity_cell() {
    local name="$1"
    local bundle="$2"
    local tag="$3"
    shift 3
    local out="${RESULTS}/${name}.${tag}.out"
    local js="${RESULTS}/${name}.${tag}.json"
    local log="${RESULTS}/${name}.${tag}.log"
    if ! "${HOST}" "${XCLBIN}" "${bundle}" "${out}" \
        "$@" --warmups 0 --reps 1 --start host_x0 \
        --json-out "${js}" >"${log}" 2>&1; then
        note_fail "${tag} ${name} (host exit)"
        return 0
    fi
    if ! grep -q '"backend":"xrt"' "${js}"; then
        note_fail "${tag} ${name} BACKEND_NOT_XRT"
        return 0
    fi
    if ! python3 "${cmp_py}" "${RECORDS}/${name}.native_oneshot.out" "${out}" --json "${js}"; then
        note_fail "${tag} ${name} (compare)"
    fi
}

run_named() {
    local name="$1"
    local bundle="$2"
    local width="$3"
    if (( width > MAX_N )); then
        echo "SKIP ${name}: n=${width} exceeds package cap ${MAX_N}"
        return 0
    fi
    identity_cell "${name}" "${bundle}" oneshot --one-shot
    identity_cell "${name}" "${bundle}" persistent --persistent
}

run_named prob_6x18 "${FIX_DIR}/prob_6x18.bin" 6
run_named prob_20x60 "${FIX_DIR}/prob_20x60.bin" 20
run_named prob_64x64 "${FIX_DIR}/prob_64x64.bin" 64
run_named fleet_n10_it2000 "${FLEET_DIR}/fleet_n10_it2000.bin" 10
run_named fleet_n40_it2000 "${FLEET_DIR}/fleet_n40_it2000.bin" 40
run_named fleet_n160_it2000 "${FLEET_DIR}/fleet_n160_it2000.bin" 160
run_named fleet_n320_it2000 "${FLEET_DIR}/fleet_n320_it2000.bin" 320
run_named square_T16_K2000 "${SQUARE_DIR}/square_T16_K2000.bin" 16
run_named square_T32_K2000 "${SQUARE_DIR}/square_T32_K2000.bin" 32
run_named square_T64_K2000 "${SQUARE_DIR}/square_T64_K2000.bin" 64
run_named square_T80_K2000 "${SQUARE_DIR}/square_T80_K2000.bin" 80
run_named square_T16_K30000 "${SQUARE_DIR}/square_T16_K30000.bin" 16
run_named square_T32_K30000 "${SQUARE_DIR}/square_T32_K30000.bin" 32
run_named square_T64_K30000 "${SQUARE_DIR}/square_T64_K30000.bin" 64
run_named square_T80_K30000 "${SQUARE_DIR}/square_T80_K30000.bin" 80

warm_out="${RESULTS}/prob_6x18.warm3.out"
warm_js="${RESULTS}/prob_6x18.warm3.json"
warm_log="${RESULTS}/prob_6x18.warm3.log"
if "${HOST}" "${XCLBIN}" "${FIX_DIR}/prob_6x18.bin" "${warm_out}" \
    --persistent --warmups 0 --reps 3 --start resident_warm \
    --json-out "${warm_js}" >"${warm_log}" 2>&1; then
    if grep -q '"backend":"xrt"' "${warm_js}"; then
        echo "warm3_ok prob_6x18"
    else
        note_fail "warm3 prob_6x18 BACKEND_NOT_XRT"
    fi
else
    note_fail "warm3 prob_6x18"
fi

if (( fails > 0 )); then
    echo "identity failures=${fails}"
    exit 1
fi
echo "identity failures=0"
