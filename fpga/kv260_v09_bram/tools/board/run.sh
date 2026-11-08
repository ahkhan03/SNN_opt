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
#   PYTHON        interpreter (${root}/.venv/bin/python if it exists, else python3)
#   DECAY_TARGET  normalized KKT target used by DECAY_AUTO (default 1e-4)
#   DECAY_AUTO    1 adds one automatic decay cell for each pinned fixture
#   NATIVE        native_fixed_v09b. When set, a decay cell generates its
#                 reference with the emitted args. When unset, the cell
#                 records identity=unvalidated_by_native and still checks
#                 the decision fields.
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
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${root}/.venv/bin/python" ]]; then
        PYTHON="${root}/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi
DECAY_TARGET="${DECAY_TARGET:-1e-4}"
DECAY_AUTO="${DECAY_AUTO:-0}"

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

run_host_cell() {
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
        return 1
    fi
    if ! grep -q '"backend":"xrt"' "${js}"; then
        note_fail "${tag} ${name} BACKEND_NOT_XRT"
        return 1
    fi
}

identity_cell() {
    local name="$1"
    local bundle="$2"
    local tag="$3"
    shift 3
    if ! run_host_cell "${name}" "${bundle}" "${tag}" "$@"; then
        return 0
    fi
    local out="${RESULTS}/${name}.${tag}.out"
    local js="${RESULTS}/${name}.${tag}.json"
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

# Only the automatic cells need the package. The constant-k0 matrix and the
# pinned decay cells use the host binary and stdlib Python alone, so a board
# without the package still runs them.
if [[ "${DECAY_AUTO}" == "1" ]]; then
    export PYTHONPATH="${root}/src:${root}${PYTHONPATH:+:${PYTHONPATH}}"
    if ! "${PYTHON}" -c "import numpy, snn_opt"; then
        echo "DECAY_AUTO=1 needs numpy and snn_opt on ${PYTHON}" >&2
        exit 1
    fi
fi

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

decay_cell() {
    local name="$1"
    local bundle="$2"
    local floor="$3"
    local segment="$4"
    local levels="$5"
    local final="$6"
    identity_cell "${name}" "${bundle}" oneshot --one-shot \
        --k0-floor "${floor}" --decay-segment "${segment}"
    identity_cell "${name}" "${bundle}" persistent --persistent \
        --k0-floor "${floor}" --decay-segment "${segment}"
    local mode
    for mode in oneshot persistent; do
        local js="${RESULTS}/${name}.${mode}.json"
        if [[ -f "${js}" ]]; then
            grep -q "\"k0_floor\":${floor}" "${js}" || note_fail "${name} ${mode} K0_FLOOR_JSON"
            grep -q "\"decay_segment\":${segment}" "${js}" || note_fail "${name} ${mode} DECAY_SEGMENT_JSON"
            grep -q "\"decay_levels\":${levels}" "${js}" || note_fail "${name} ${mode} DECAY_LEVELS_JSON"
            grep -q "\"k0_final_effective\":${final}" "${js}" || note_fail "${name} ${mode} FINAL_K0_JSON"
        fi
    done
}

decay_cell decay_healthy_prob_6x18 "${FIX_DIR}/prob_6x18.bin" 0.375 10 4 0.375
decay_cell decay_stiff_square_T16_K2000 "${SQUARE_DIR}/square_T16_K2000.bin" 0.002 400 4 0.001999974250793457

# One automatic cell per pinned fixture. Exit 3 is no_decay or refuse; the
# decision file says which. Only no_decay runs constant-k0. refuse fails
# the cell and does not run the host. A decay generates the native
# reference for the emitted args when NATIVE is set, and never compares
# against the pinned record for this cell name.
decay_auto_cell() {
    local name="$1"
    local bundle="$2"
    local decision="${RESULTS}/${name}.decision.json"
    local args_file="${RESULTS}/${name}.args"
    local rc=0
    "${PYTHON}" -m fpga.kv260_v09_bram.tools.board.decay_choice \
        --bundle "${bundle}" \
        --target "${DECAY_TARGET}" \
        --json-out "${decision}" \
        --args-out "${args_file}" || rc=$?
    if (( rc != 0 && rc != 3 )); then
        note_fail "auto ${name} (adapter exit ${rc})"
        return 0
    fi
    local label
    label="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("decision",""))' "${decision}")" \
        || { note_fail "auto ${name} (decision unreadable)"; return 0; }
    echo "${label} ${name}"
    if [[ "${label}" == "no_decay" ]]; then
        # Constant k0 is invocation-identical to the pinned identity cell for
        # this bundle, so the shipped native record is the reference.
        local base ref mode flag
        base="$(basename "${bundle}" .bin)"
        ref="${RECORDS}/${base}.native_oneshot.out"
        for mode in auto_oneshot auto_persistent; do
            if [[ "${mode}" == "auto_oneshot" ]]; then
                flag="--one-shot"
            else
                flag="--persistent"
            fi
            if ! run_host_cell "${name}" "${bundle}" "${mode}" "${flag}"; then
                continue
            fi
            if [[ -f "${ref}" ]]; then
                if ! python3 "${cmp_py}" "${ref}" \
                    "${RESULTS}/${name}.${mode}.out" \
                    --json "${RESULTS}/${name}.${mode}.json"; then
                    note_fail "auto ${name} ${mode} (constant-k0 record)"
                fi
            else
                echo "identity=unvalidated_by_native ${name} ${mode}"
            fi
        done
        return 0
    fi
    if [[ "${label}" != "decay" ]]; then
        local reason
        reason="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommendation"].get("reason",""))' "${decision}")" \
            || { note_fail "auto ${name} ${label} (reason unreadable)"; return 0; }
        note_fail "auto ${name} ${label} ${reason}"
        return 0
    fi
    local -a args=()
    mapfile -t args < "${args_file}"
    local native_out="${RESULTS}/${name}.native_oneshot.out"
    local have_native=0
    if [[ -n "${NATIVE:-}" ]]; then
        if ! "${NATIVE}" "${bundle}" "${native_out}" "${args[@]}"; then
            note_fail "auto ${name} (native exit)"
            return 0
        fi
        have_native=1
    else
        echo "identity=unvalidated_by_native ${name}"
    fi
    local mode flag
    for mode in auto_oneshot auto_persistent; do
        if [[ "${mode}" == "auto_oneshot" ]]; then
            flag="--one-shot"
        else
            flag="--persistent"
        fi
        if ! run_host_cell "${name}" "${bundle}" "${mode}" "${flag}" "${args[@]}"; then
            continue
        fi
        local out="${RESULTS}/${name}.${mode}.out"
        local js="${RESULTS}/${name}.${mode}.json"
        if (( have_native )); then
            if ! "${PYTHON}" "${cmp_py}" "${native_out}" "${out}" \
                --json "${js}" --decision "${decision}"; then
                note_fail "auto ${name} ${mode} (decision)"
            fi
        elif ! "${PYTHON}" "${cmp_py}" --json "${js}" --decision "${decision}"; then
            note_fail "auto ${name} ${mode} (decision)"
        fi
    done
}

if [[ "${DECAY_AUTO}" == "1" ]]; then
    decay_auto_cell decay_healthy_prob_6x18 "${FIX_DIR}/prob_6x18.bin"
    decay_auto_cell decay_stiff_square_T16_K2000 "${SQUARE_DIR}/square_T16_K2000.bin"
fi

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
