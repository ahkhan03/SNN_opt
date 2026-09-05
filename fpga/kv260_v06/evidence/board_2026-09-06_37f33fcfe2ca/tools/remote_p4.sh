#!/usr/bin/env bash
set -u
cd /root/snn_v06
X=/lib/firmware/xilinx/APPNAME/snn_qp_v06.xclbin
for p in 6x18 20x60 64x64; do
  ./host_kv260_v06 $X prob_$p.bin $p.oneshot.out --one-shot --warmups 0 --reps 1 --start host_x0 --json-out $p.oneshot.json >$p.oneshot.log 2>&1 && echo oneshot_${p}_ok || { echo oneshot_${p}_FAIL; tail -3 $p.oneshot.log; }
  ./host_kv260_v06 $X prob_$p.bin $p.persistent.out --persistent --warmups 0 --reps 1 --start host_x0 --json-out $p.persistent.json >$p.persistent.log 2>&1 && echo persistent_${p}_ok || { echo persistent_${p}_FAIL; tail -3 $p.persistent.log; }
done
./host_kv260_v06 $X prob_6x18.bin 6x18.stream.out --persistent --route stream --warmups 0 --reps 1 --start host_x0 >6x18.stream.log 2>&1 && echo stream_ok || { echo stream_FAIL; tail -3 6x18.stream.log; }
./host_kv260_v06 $X prob_6x18.bin 6x18.cg.out --persistent --route cg --warmups 0 --reps 1 --start host_x0 >6x18.cg.log 2>&1 && echo cg_ok || { echo cg_FAIL; tail -3 6x18.cg.log; }
./host_kv260_v06 $X prob_6x18.bin 6x18.warm.out --persistent --warmups 0 --reps 3 --start resident_warm --json-out 6x18.warm.json >6x18.warm.log 2>&1 && echo warm3_ok || { echo warm3_FAIL; tail -3 6x18.warm.log; }
