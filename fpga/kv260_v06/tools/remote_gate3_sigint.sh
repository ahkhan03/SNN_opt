#!/usr/bin/env bash
# Gate-3 board-only graceful-SIGINT drain script.
#
# Copy the stress binary, host binary, and a small H=3,N=1 problem bundle to
# ROOT first.  This script sends SIGINT to the stress host, waits for its
# bounded STOP drain, and then launches a fresh persistent host without
# resetting the loaded application.
set -u -o pipefail

ROOT="${ROOT:-/root/snn_v06}"
STRESS="${STRESS:-${ROOT}/stress_kv260_v06}"
HOST="${HOST:-${ROOT}/host_kv260_v06}"
XCLBIN="${XCLBIN:-/lib/firmware/xilinx/APPNAME/snn_qp_v06.xclbin}"
PROBLEM="${PROBLEM:-${ROOT}/prob_6x18.bin}"
OUT="${OUT:-${ROOT}/gate3/sigint}"
COUNT="${COUNT:-100000}"
SEED="${SEED:-0x544333360906}"
SEQUENCE_SEED="${SEQUENCE_SEED:-4294967290}"
KILL_AFTER_SEC="${KILL_AFTER_SEC:-2}"
mkdir -p "${OUT}"

if [[ ! -x "${STRESS}" || ! -x "${HOST}" || ! -f "${PROBLEM}" ]]; then
  echo "missing STRESS/HOST/PROBLEM; set ROOT or the corresponding variables" >&2
  exit 2
fi

echo "starting persistent stress count=${COUNT} seed=${SEED} sequence_seed=${SEQUENCE_SEED}"
"${STRESS}" "${XCLBIN}" "${PROBLEM}" "${OUT}/sigint.json" \
  --count "${COUNT}" --seed "${SEED}" --sequence-seed "${SEQUENCE_SEED}" \
  >"${OUT}/sigint.log" 2>&1 &
stress_pid=$!
echo "stress_pid=${stress_pid}" | tee "${OUT}/pid"

# Give the mailbox service time to be in-flight, then interrupt exactly this
# stress process.  COUNT is deliberately large so natural completion cannot
# be mistaken for the graceful-drain case.
sleep "${KILL_AFTER_SEC}"
signal_rc=0
if kill -0 "${stress_pid}" 2>/dev/null; then
  echo "sending SIGINT to stress_pid=${stress_pid}"
  kill -INT "${stress_pid}" 2>/dev/null || signal_rc=$?
else
  echo "stress exited before SIGINT; inspect ${OUT}/sigint.log" >&2
  signal_rc=1
fi

stress_rc=0
wait "${stress_pid}" 2>/dev/null || stress_rc=$?
echo "stress_rc=${stress_rc}" | tee "${OUT}/sigint.status"

if [[ ${signal_rc} -eq 0 && -s "${OUT}/sigint.json" ]]; then
  JSON_PATH="${OUT}/sigint.json" python3 - <<'PY' | tee -a "${OUT}/sigint.log"
import json
import os
import sys

with open(os.environ["JSON_PATH"], encoding="utf-8") as f:
    record = json.load(f)

checks = {
    "backend_xrt": record.get("backend") == "xrt",
    "interrupted": record.get("interrupted") is True,
    "clean_stop_pass": record.get("clean_stop_pass") is True,
    "stop_ok": record.get("stop_ok") is True,
}
print(json.dumps({"checks": checks, "record": os.environ["JSON_PATH"]},
                 sort_keys=True, separators=(",", ":")))
sys.exit(0 if all(checks.values()) else 1)
PY
  report_rc=${PIPESTATUS[0]}
else
  echo "missing SIGINT report ${OUT}/sigint.json" >&2
  report_rc=1
fi

echo "starting fresh configure/solve host without a board reset"
"${HOST}" "${XCLBIN}" "${PROBLEM}" "${OUT}/fresh.fixed.bin" \
  --persistent --route auto --start host_x0 --warmups 0 --reps 1 \
  --json-out "${OUT}/fresh.json" >"${OUT}/fresh.log" 2>&1
fresh_rc=$?

if [[ ${fresh_rc} -eq 0 && -s "${OUT}/fresh.json" ]]; then
  JSON_PATH="${OUT}/fresh.json" python3 - <<'PY' | tee -a "${OUT}/fresh.log"
import json
import os
import sys

with open(os.environ["JSON_PATH"], encoding="utf-8") as f:
    record = json.load(f)

route = record.get("route_selected")
raw = record.get("x_raw_fixed", [])
telemetry = record.get("telemetry_by_call", [])
raw_ok = (record.get("n") == 6 and isinstance(raw, list) and bool(raw) and
          all(isinstance(row, list) and len(row) == 6 for row in raw))
telemetry_ok = (isinstance(telemetry, list) and bool(telemetry) and
                all(isinstance(row, list) and len(row) == 16
                    for row in telemetry))
route_ok = route in ("FULL", "CG", "STREAM", 1, 2, 3)
checks = {"backend_xrt": record.get("backend") == "xrt",
          "route": route_ok, "raw": raw_ok, "telemetry16": telemetry_ok}
print(json.dumps({"route_selected": route,
                  "raw_rows": len(raw) if isinstance(raw, list) else 0,
                  "telemetry_calls": len(telemetry)
                  if isinstance(telemetry, list) else 0,
                  "checks": checks}, sort_keys=True,
                 separators=(",", ":")))
sys.exit(0 if all(checks.values()) else 1)
PY
  verify_rc=${PIPESTATUS[0]}
else
  echo "fresh host failed rc=${fresh_rc}" >&2
  tail -10 "${OUT}/fresh.log" || true
  verify_rc=1
fi

echo "GATE3 SIGINT SUMMARY stress_rc=${stress_rc} report_rc=${report_rc} fresh_rc=${fresh_rc} verify_rc=${verify_rc} results=${OUT}"
[[ ${signal_rc} -eq 0 ]] && [[ ${stress_rc} -eq 130 ]] && \
  [[ ${report_rc} -eq 0 ]] && [[ ${fresh_rc} -eq 0 ]] && \
  [[ ${verify_rc} -eq 0 ]]
