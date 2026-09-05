#!/usr/bin/env bash
set -u
cd /root/snn_v06
X=/lib/firmware/xilinx/APPNAME/snn_qp_v06.xclbin
./host_kv260_v06_stream $X resident_H3_N1_s11.bin h3.oneshot.fixed.bin --one-shot --warmups 2 --reps 1 --json-out h3.oneshot.smoke.json >h3.oneshot.smoke.log 2>&1 && echo smoke_oneshot_ok || { echo smoke_oneshot_FAIL; tail -3 h3.oneshot.smoke.log; }
./host_kv260_v06_stream $X resident_H3_N1_s11.bin h3.persistent.fixed.bin --persistent --warmups 2 --reps 1 --json-out h3.persistent.smoke.json >h3.persistent.smoke.log 2>&1 && echo smoke_persistent_ok || { echo smoke_persistent_FAIL; tail -3 h3.persistent.smoke.log; }
python3 - <<'PY'
import json
for f in ('h3.oneshot.smoke.json','h3.persistent.smoke.json'):
    j=json.load(open(f)); print(f, 'gate', j.get('feasibility_gate_pass'), 'maxviol_pu', j.get('max_feasibility_violation_pu'))
PY
for b in H3_N1 H10_N1; do
  ./host_kv260_v06_stream $X resident_${b}_s11.bin $b.lat.oneshot.bin --one-shot --warmups 20 --reps 1000 --json-out $b.lat.oneshot.json >$b.lat.oneshot.log 2>&1 && echo lat_oneshot_${b}_ok || { echo lat_oneshot_${b}_FAIL; tail -2 $b.lat.oneshot.log; }
  for ps in always never; do
    ./host_kv260_v06_stream $X resident_${b}_s11.bin $b.lat.persistent.$ps.bin --persistent --warmups 20 --reps 1000 --poll-sync $ps --poll yield --json-out $b.lat.persistent.$ps.json >$b.lat.persistent.$ps.log 2>&1 && echo lat_persistent_${b}_${ps}_ok || { echo lat_persistent_${b}_${ps}_FAIL; tail -2 $b.lat.persistent.$ps.log; }
  done
done
