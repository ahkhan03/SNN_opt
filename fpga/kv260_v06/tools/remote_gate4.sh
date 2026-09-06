#!/usr/bin/env bash
# Gate-4 board half.  Copy the fixture directory produced by
# gate4_route_boundary.py to the board before running this script.  The script
# intentionally does not invoke xmutil or alter the loaded image.
set -u -o pipefail

ROOT="${ROOT:-/root/snn_v06}"
HOST="${HOST:-${ROOT}/host_kv260_v06}"
XCLBIN="${XCLBIN:-/lib/firmware/xilinx/APPNAME/snn_qp_v06.xclbin}"
FIXTURES="${FIXTURES:-${ROOT}/gate4/fixtures}"
OUT="${OUT:-${ROOT}/gate4/board_results}"
mkdir -p "${OUT}"

declare -a names=(
  full_320 full_321 cg_372 cg_373
  full_asym_fit full_asym_next cg_asym_fit cg_asym_next
)
declare -A expected=(
  [full_320]=FULL [full_321]=CG [cg_372]=CG [cg_373]=STREAM
  [full_asym_fit]=FULL [full_asym_next]=CG
  [cg_asym_fit]=CG [cg_asym_next]=STREAM
)

pass=0
fail=0
for name in "${names[@]}"; do
  problem="${FIXTURES}/${name}.bin"
  fixed="${OUT}/${name}.fixed.bin"
  json="${OUT}/${name}.json"
  log="${OUT}/${name}.log"
  if [[ ! -f "${problem}" ]]; then
    echo "FAIL ${name}: missing ${problem}" | tee "${log}"
    ((fail += 1))
    continue
  fi
  "${HOST}" "${XCLBIN}" "${problem}" "${fixed}" \
    --persistent --route auto --warmups 0 --reps 1 --start host_x0 \
    --json-out "${json}" >"${log}" 2>&1
  rc=$?
  if [[ ${rc} -ne 0 || ! -s "${json}" ]]; then
    echo "FAIL ${name}: host rc=${rc}" | tee -a "${log}"
    tail -5 "${log}" || true
    ((fail += 1))
    continue
  fi

  # Keep a compact, diff-friendly record while retaining the complete JSON
  # and fixed output beside it.  x_raw_fixed and telemetry_by_call are the
  # byte-level board evidence; route_selected is OUT_SELECTED_ROUTE's JSON
  # projection.
  NAME="${name}" EXPECTED="${expected[$name]}" JSON_PATH="${json}" \
    python3 - <<'PY' | tee -a "${log}"
import json, os
with open(os.environ["JSON_PATH"], encoding="utf-8") as f:
    record = json.load(f)
selected = record.get("route_selected")
expected = os.environ["EXPECTED"]
raw = record.get("x_raw_fixed", [])
telemetry = record.get("telemetry_by_call", [])
print(json.dumps({
    "fixture": os.environ["NAME"],
    "selected_route": selected,
    "expected_route": expected,
    "route_match": selected == expected,
    "x_raw_fixed": raw,
    "telemetry_by_call": telemetry,
}, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if selected == expected else 1)
PY
  check_rc=${PIPESTATUS[0]}
  if [[ ${check_rc} -eq 0 ]]; then
    echo "PASS ${name} selected_route=${expected[$name]}"
    ((pass += 1))
  else
    echo "FAIL ${name}: selected_route mismatch (expected ${expected[$name]})"
    ((fail += 1))
  fi
done

echo "GATE4 BOARD SUMMARY pass=${pass} fail=${fail} results=${OUT}"
[[ ${fail} -eq 0 ]]
