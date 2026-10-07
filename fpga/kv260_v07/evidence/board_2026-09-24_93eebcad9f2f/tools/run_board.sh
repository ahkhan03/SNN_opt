#!/usr/bin/env bash
# v07 board qualification run (executed on kria as root). Restores the v06 app at the end.
set -u
W=/root/snn_v07; APP=snn_qp_v07_200m_93eebcad9f2f; V06=snn_qp_v06_200m_37f33fcfe2ca
cd $W
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
echo "## pl0_ref with v06 loaded: $(cat /sys/kernel/debug/clk/pl0_ref/clk_rate)"
echo "## v06 dtbo rate bytes: $(xxd -p /lib/firmware/xilinx/$V06/snn_qp_v06.dtbo | tr -d '\n' | grep -o -E '0ee6b280|0bebc200' | sort | uniq -c | tr '\n' ' ')"
mkdir -p /lib/firmware/xilinx/$APP && cp app/* /lib/firmware/xilinx/$APP/
xmutil unloadapp >/dev/null 2>&1; xmutil loadapp $APP 2>&1 | tail -1
sleep 2
echo "## pl0_ref with v07 loaded: $(cat /sys/kernel/debug/clk/pl0_ref/clk_rate)"
xbutil examine 2>&1 | grep -i -E "ready|device" | head -3
bash src/fpga/kv260_v07/build/build_host.sh $W/bin 2>&1 | tail -2
H=$W/bin/host_kv260_v07_xrt; X=/lib/firmware/xilinx/$APP/snn_qp_v07.xclbin
mkdir -p out
for c in a_ball_n8 b_friction_q1_seed4903 c_mixed_anchor d_cones_off_v06_parity; do
  cones=""; [ -f fix/$c.cones ] && cones="--cones fix/$c.cones"
  timeout 600 $H $X fix/$c.bin out/$c.oneshot.bin --one-shot --warmups 0 --reps 1 --start host_x0 $cones --json-out out/$c.oneshot.json > out/$c.oneshot.log 2>&1; rc=$?
  if cmp -s out/$c.oneshot.bin fix/$c.expected.bin; then echo "ONESHOT $c: board == native (byte-identical) rc=$rc"; else echo "ONESHOT $c: DIFFERS rc=$rc"; tail -2 out/$c.oneshot.log; fi
done
for p in 6x18 20x60 64x64; do
  timeout 600 $H $X fix/prob_$p.bin out/$p.oneshot.out --one-shot --warmups 0 --reps 1 --start host_x0 > out/$p.oneshot.log 2>&1; echo "ONESHOT v06-fixture $p rc=$?"
done
timeout 600 $H $X fix/prob_6x18.bin out/6x18.persistent.out --persistent --warmups 0 --reps 1 --start host_x0 > out/6x18.persistent.log 2>&1; echo "PERSISTENT 6x18 rc=$?"
timeout 600 $H $X fix/a_ball_n8.bin out/a_ball_n8.persistent.bin --persistent --warmups 0 --reps 1 --start host_x0 --cones fix/a_ball_n8.cones > out/a_ball.persistent.log 2>&1; rc=$?
cmp -s out/a_ball_n8.persistent.bin fix/a_ball_n8.expected.bin && echo "PERSISTENT a_ball_n8: board == native rc=$rc" || { echo "PERSISTENT a_ball_n8: DIFFERS rc=$rc"; tail -2 out/a_ball.persistent.log; }
dmesg | tail -5 | grep -i -E "hang|outstanding|error" || echo "## dmesg clean"
xmutil unloadapp >/dev/null 2>&1; xmutil loadapp $V06 2>&1 | tail -1
echo "## restored: $(xmutil listapps 2>/dev/null | awk '$NF ~ /0,/ {print $1}')"
