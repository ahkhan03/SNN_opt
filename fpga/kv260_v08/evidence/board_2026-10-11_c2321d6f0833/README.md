# Board qualification of the v08 kernel (2026-10-11)

Bitstream `snn_qp_v08_200m_c2321d6f0833` (xclbin SHA-256 prefix `c2321d6f0833`),
built from `fpga/kv260_v08` at commit 4e0a9e3 (kernel source identical to
90fda23) with Vitis 2022.1 on the recorded K26 platform at 200 MHz, HLS clock
uncertainty 2.5 ns: WNS +0.006 ns, TNS 0, WHS +0.010 ns; routed utilization
86,181 LUT (73.6 %), 100,690 FF, 29.5 BRAM tiles, 56 URAM, 287 DSP
(`implementation/`). Board: KV260, XRT 2.13, governor `userspace`, app
`snn_qp_v08_k200_c2321d6f0833` loaded with `kria-clock load`: pl0_ref
249,999,998 Hz, ratio 0.8, kernel 200.000 MHz (`results/clock_measured.json`).
Host `host_kv260_v08` built on the board with the XRT backend (SHA-256 in
`results/host_sha256.txt`).

## Parity

`tools/run.sh` (driver) and `tools/cmp_out.py` (raw state plus all 16
telemetry words). Every comparison below printed MATCH:

* The v06 framework fixtures (`fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/fixtures`:
  6x18, 20x60, 64x64, box both sides, 50 iterations) in ONESHOT and PERSISTENT
  mode (AUTO = FULL), and 6x18 under FORCE_STREAM and FORCE_CG, against the
  native emulation records `*.native_oneshot.out`. A three-solve
  RESIDENT_WARM sequence ran without error.
* The economic-dispatch band fleet (m = 2, box [0, 1], 2000 iterations) at
  n = 10, 40, 160, 320, PERSISTENT, HOST_X0, AUTO = FULL, against the v06
  bitstream's board outputs for the same bundles: byte-identical, so v08
  reproduces v06's fixed-point trajectory event for event on the board.

## Latency (band fleet, 5 warm-ups, 50 timed solves, `--poll-sync never`)

Median complete time per solve divided by the 2000 outer iterations
(`results/fleet_n*.pl.json.gz`):

| n | v08 (us/iter) | v06 same bundles (us/iter) |
|---|---|---|
| 10 | 26.6 | 77.5 |
| 40 | 43.9 | 176.9 |
| 160 | 209.2 | 1313.6 |
| 320 | 536.0 | 4305.9 |

## Not claimed here

Energy per solve, the CG and STREAM tiers at sizes where they are selected
automatically, the PMSM stream client, and behaviour above the band fleet's
sizes on the board.
