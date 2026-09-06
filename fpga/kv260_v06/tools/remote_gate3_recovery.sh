#!/usr/bin/env bash
# Gate-3 board-only killed-host recovery script.
#
# Copy the stress binary, host binary, and a small H=3,N=1 problem bundle to
# ROOT first.  This script is intentionally destructive to the running
# application (SIGKILL); it never deletes files and is run on the board, not
# automatically.
set -u -o pipefail

ROOT="${ROOT:-/root/snn_v06}"
STRESS="${STRESS:-${ROOT}/stress_kv260_v06}"
HOST="${HOST:-${ROOT}/host_kv260_v06}"
XCLBIN="${XCLBIN:-/lib/firmware/xilinx/APPNAME/snn_qp_v06.xclbin}"
PROBLEM="${PROBLEM:-${ROOT}/prob_6x18.bin}"
OUT="${OUT:-${ROOT}/gate3/recovery}"
COUNT="${COUNT:-100000}"
SEED="${SEED:-0x544333360906}"
SEQUENCE_SEED="${SEQUENCE_SEED:-4294967290}"
KILL_AFTER_SEC="${KILL_AFTER_SEC:-2}"
APP_NAME="${APP_NAME:-}"
mkdir -p "${OUT}"

if [[ ! -x "${STRESS}" || ! -x "${HOST}" || ! -f "${PROBLEM}" ]]; then
  echo "missing STRESS/HOST/PROBLEM; set ROOT or the corresponding variables" >&2
  exit 2
fi

echo "starting persistent stress count=${COUNT} seed=${SEED} sequence_seed=${SEQUENCE_SEED}"
"${STRESS}" "${XCLBIN}" "${PROBLEM}" "${OUT}/killed-host.json" \
  --count "${COUNT}" --seed "${SEED}" --sequence-seed "${SEQUENCE_SEED}" \
  >"${OUT}/killed-host.log" 2>&1 &
stress_pid=$!
echo "stress_pid=${stress_pid}" | tee "${OUT}/pid"

# Give the mailbox service time to be in-flight, then kill exactly this host.
# The stress count is deliberately large by default so it cannot normally
# drain before this point; override KILL_AFTER_SEC only when reproducing a
# particular board timing.
sleep "${KILL_AFTER_SEC}"
if kill -0 "${stress_pid}" 2>/dev/null; then
  echo "sending SIGKILL to stress_pid=${stress_pid}"
  kill -9 "${stress_pid}" 2>/dev/null || true
else
  echo "stress exited before SIGKILL; inspect ${OUT}/killed-host.log" >&2
  wait "${stress_pid}" 2>/dev/null || true
  exit 3
fi
wait "${stress_pid}" 2>/dev/null || killed_rc=$?
killed_rc=${killed_rc:-0}
echo "killed_host_rc=${killed_rc}" | tee "${OUT}/killed-host.status"

echo "resetting the orphaned persistent CU with xmutil unloadapp/loadapp"
if [[ -n "${APP_NAME}" ]]; then
  xmutil unloadapp "${APP_NAME}" >"${OUT}/xmutil-unload.log" 2>&1
  unload_rc=$?
  xmutil loadapp "${APP_NAME}" >"${OUT}/xmutil-load.log" 2>&1
  load_rc=$?
else
  xmutil unloadapp >"${OUT}/xmutil-unload.log" 2>&1
  unload_rc=$?
  xmutil loadapp >"${OUT}/xmutil-load.log" 2>&1
  load_rc=$?
fi
if [[ ${unload_rc} -ne 0 || ${load_rc} -ne 0 ]]; then
  echo "xmutil recovery failed unload=${unload_rc} load=${load_rc}" >&2
  exit 4
fi

echo "starting fresh configure/solve host"
"${HOST}" "${XCLBIN}" "${PROBLEM}" "${OUT}/fresh.fixed.bin" \
  --persistent --route auto --start host_x0 --warmups 0 --reps 1 \
  --json-out "${OUT}/fresh.json" >"${OUT}/fresh.log" 2>&1
fresh_rc=$?

if [[ ${fresh_rc} -eq 0 && -s "${OUT}/fresh.json" ]]; then
  JSON_PATH="${OUT}/fresh.json" python3 - <<'PY' 
import json, os, sys
with open(os.environ["JSON_PATH"], encoding="utf-8") as f:
    record = json.load(f)
raw = record.get("x_raw_fixed", [])
telemetry = record.get("telemetry_by_call", [])
route = record.get("route_selected")
ok = (bool(raw) and bool(telemetry) and
      all(isinstance(row, list) and len(row) == 16 for row in telemetry) and
      route in ("FULL", "CG", "STREAM", 1, 2, 3))
print(json.dumps({"fresh_route": route, "raw_words": len(raw),
                  "telemetry_calls": len(telemetry), "recovery_pass": ok},
                 sort_keys=True))
sys.exit(0 if ok else 1)
PY
  verify_rc=$?
else
  echo "fresh host failed rc=${fresh_rc}" >&2
  tail -10 "${OUT}/fresh.log" || true
  verify_rc=1
fi

echo "GATE3 KILLED-HOST SUMMARY killed_rc=${killed_rc} unload_rc=${unload_rc} load_rc=${load_rc} fresh_rc=${fresh_rc} verify_rc=${verify_rc}"
[[ ${killed_rc} -eq 137 || ${killed_rc} -eq 9 ]] && [[ ${verify_rc} -eq 0 ]]
