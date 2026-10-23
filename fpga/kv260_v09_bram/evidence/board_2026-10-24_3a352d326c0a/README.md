# Board qualification of the v09b BRAM kernel (2026-10-24)

Bitstream `snn_qp_v09b_200m_3a352d326c0a` (xclbin SHA-256 prefix
`3a352d326c0a`), built from `fpga/kv260_v09_bram` at kernel commit `bf208c2`
(branch `experiment/v09-polyhedral`; harness/tools commits after it do not
touch the kernel) with Vitis 2022.1 on the recorded K26 platform at 200 MHz,
HLS clock uncertainty 2.5 ns, link strategy
`Performance_ExplorePostRoutePhysOpt`: routed **WNS +0.006 ns, TNS 0,
WHS +0.010 ns, WPWS +0.500 ns**; utilization **72,204 CLB LUT (61.65 %),
83,745 FF (35.75 %), 86 BRAM tiles (59.72 %), 0 URAM, 374 DSP (29.97 %)**
(`implementation/`, `csynth/`). The full build archive (bitstream, boot files, unscrubbed tool reports, raw
board results) is kept on the build host beside the historical Kria builds,
as for every qualified image; the files here are the load-bearing mirrors.

Board: KV260, XRT 2.13, governor `userspace`, app
`snn_qp_v09b_200m_3a352d326c0a` loaded with `kria-clock load`: pl0_ref
249,999,998 Hz, clk_wiz ratio 0.8, **kernel_hz_measured 199,999,998** (the
number to quote; `results/v09b_clock_measured.json`, verdict OK). Host
`host_kv260_v09b` built on the board with the XRT backend (SHA-256 in
`results/v09b_host_sha256.txt`; runner gates on `"backend":"xrt"` per solve).

## Parity (byte identity against the x86 oracle records)

`tools/board/run.sh` + `cmp_out.py` (raw state, all 16 telemetry words,
error, route). HOST_X0, warmups 0, reps 1, route AUTO (= FULL on every
accepted fixture), each cell in ONESHOT and PERSISTENT mode against
`evidence/native_records/*.native_oneshot.out`:

* v06 framework fixtures `prob_6x18`, `prob_20x60`, `prob_64x64`: 6/6 MATCH.
* ED fleet `fleet_n10_it2000`, `fleet_n40_it2000`: 4/4 MATCH.
  `fleet_n160`, `fleet_n320`: SKIP (n exceeds the package cap 128; the
  loader exits 2 with no record, by design).
* Square alt-investment bundles `square_T{16,32,64,80}_K{2000,30000}`
  (payload chain of the paper's 22a driver, byte-verified against an
  independent regeneration): 16/16 MATCH.
* `RESIDENT_WARM` x3 on prob_6x18 (persistent): exit-status check, ok.
* Final line: `identity failures=0`.

The x86 side of the contract (v09b oracle bit-identical to the v06 kernel on
the same 13 bundles, plus the rejection/contract battery) is
`build/run_parity.sh`; its gate ran `failures=0` on the committed state.

## Latency (square bundles, K=30000, 5 warm-ups, 50 timed reps, `--poll-sync never`)

Median `complete_ns` / 30000 (`results/v09b_latency/`), cycles at the
measured 200 MHz kernel clock. Anchors: the 2026-10-23 equal-clock accounting
(`fpga/kv260_v08/evidence/regime_2026-10-23/`).

| T | v09b us/iter | v09b cyc/iter | v08 cyc/iter | WINNER (retired semantics) cyc/iter | v09b/v08 |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.392 | 878.5 | 1205.6 | 573.8 | 1.37x |
| 32 | 5.939 | 1187.8 | 1875.7 | 947.0 | 1.58x |
| 64 | 9.827 | 1965.4 | 3668.5 | 1773.3 | 1.87x |
| 80 | 11.813 | 2362.6 | 4713.8 | 2239.8 | 2.00x |

ONESHOT and PERSISTENT medians agree within 0.2 %. At T=80 v09b is within
5.5 % of the legacy WINNER image while carrying current (v0.5.0+)
box-in-sweep semantics; the residual gap at small T is the v08-inherited
event body (a lean-event-FSM pass is the recorded next lever). The verified
cycle model predicted ~2000-2300 cyc/iter at T=80; measured 2362.6.

## Not claimed here

Energy per solve (no INA260 campaign in this round), the ED fleet latency as
a v09b arm (event-heavy shapes route to the URAM lineage), CG/STREAM routes
(removed; rejected with ERR_CAPACITY), sizes above the cap 128, and any
PMSM/stream client behaviour.
note: csynth/ mirrors are scrubbed of the VM scratch build-dir name (task-id style); the unscrubbed originals stay beside the bitstream archive in builds/snn_qp_v09b_200m_3a352d326c0a/reports/csynth/.
