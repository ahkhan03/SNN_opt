set -u
cd ~/v08_qualification; mkdir -p results bin
APP=snn_qp_v08_k200_c2321d6f0833; X=/lib/firmware/xilinx/$APP/snn_qp_v08.xclbin
bash fpga/kv260_v08/build/build_host.sh ~/v08_qualification/bin 2>&1 | tail -2
sha256sum bin/host_kv260_v08 > results/host_sha256.txt
nm -C bin/host_kv260_v08 | grep -q -i "xrt::" || { echo HOST_IS_MOCK; exit 1; }
test -f results/clock_measured.json || { echo CLOCK_NOT_LOADED; exit 1; }
H=./bin/host_kv260_v08
for p in 6x18 20x60 64x64; do
  $H $X fixtures/prob_$p.bin results/$p.oneshot.out --one-shot --warmups 0 --reps 1 --start host_x0 > results/$p.oneshot.log 2>&1 || echo FAIL oneshot $p
  $H $X fixtures/prob_$p.bin results/$p.persistent.out --persistent --warmups 0 --reps 1 --start host_x0 > results/$p.persistent.log 2>&1 || echo FAIL persistent $p
  python3 cmp_out.py fixtures/prob_$p.native_oneshot.out results/$p.oneshot.out
  python3 cmp_out.py fixtures/prob_$p.native_oneshot.out results/$p.persistent.out
done
for r in stream cg; do
  $H $X fixtures/prob_6x18.bin results/6x18.$r.out --persistent --route $r --warmups 0 --reps 1 --start host_x0 > results/6x18.$r.log 2>&1 || echo FAIL $r
  python3 cmp_out.py fixtures/prob_6x18.native_oneshot.out results/6x18.$r.out
done
$H $X fixtures/prob_6x18.bin results/6x18.warm.out --persistent --warmups 0 --reps 3 --start resident_warm > results/6x18.warm.log 2>&1 && echo warm3_ok || echo FAIL warm3
for n in 10 40 160 320; do
  $H $X ~/kv260_v06_run/bundles/fleet_n${n}_it2000.bin results/fleet_n$n.pl.out.bin --persistent --start host_x0 --poll-sync never --warmups 5 --reps 50 --json-out results/fleet_n$n.pl.json > results/fleet_n$n.log 2>&1 || echo FAIL fleet $n
  grep -q "\"backend\":\"xrt\"" results/fleet_n$n.pl.json || echo "BACKEND_NOT_XRT fleet $n"
  python3 cmp_out.py run2_pl/fleet_n${n}_it2000.pl.out.bin results/fleet_n$n.pl.out.bin
done
python3 - <<'PY'
import json, statistics as st
for n in (10, 40, 160, 320):
    d = json.load(open(f'results/fleet_n{n}.pl.json'))
    key = 'complete_ns' if 'complete_ns' in d else 'complete_seconds'
    t = st.median(d[key]) * (1e-9 if key == 'complete_ns' else 1.0)
    print(f'fleet n={n}: PL median {t / 2000 * 1e6:.1f} us/iter')
PY
