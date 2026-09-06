# v06 decision gates 3 & 4 on the KV260, plus the stream-host re-seed fix

Bitstream `snn_qp_v06_200m_37f33fcfe2ca` (xclbin
`/lib/firmware/xilinx/snn_qp_v06_200m_37f33fcfe2ca/snn_qp_v06.xclbin`), loaded to
slot 0 via `xmutil loadapp`. Every board step was run on the KV260 directly. The
host fix and the gate harnesses were developed and reviewed off-board; the
off-board native/mock halves were re-run independently before the board runs.

## Stream-host per-pass re-seed fix (PMSM parity)

`src/host_kv260_v06_stream.cpp`: the per-pass "first" predicate is now
`period_index == 0`; `resident_predecessor` is reset to `bundle.x0` at each pass
start; period 0 of every pass is issued as `HOST_X0`, `bundle.x0`, stride 0 (the
only command that overwrites the kernel's on-device resident state). This is the
device analogue of v05's `last_x0 = q.x0` per-pass reset. The `--host-x0`
all-periods control is preserved. Verified on the KV260 (PMSM block
`10v_device_resident_v06`, re-run 2026-09-06): committed feasibility 0.00-0.72 LSB
(gated, pass); float64 first-action agreement 1.61-3.05 LSB, matching v05's
1.6-3.0 LSB band (was 0.18-0.38 pu before the fix). Full record in the paper
`snn_pmsm_resident/discussions/08`.

## Gate 4 - route-boundary parity (PASS)

`tools/gate4_route_boundary.py` (off-board: native v06 vs MAXN=1024 native v05,
raw + 16-word telemetry, plus the host AUTO predicate) and `tools/remote_gate4.sh`
(board half). 8 boundary fixtures: `full_320`->FULL, `full_321`->CG,
`cg_372`->CG, `cg_373`->STREAM, plus asymmetric `full_asym_fit`(32x624)->FULL,
`full_asym_next`(32x632)->CG, `cg_asym_fit`(40x624)->CG,
`cg_asym_next`(48x624)->STREAM.

- Off-board (controller re-run): 8/8, raw + telemetry + native-selected +
  host-predicted route all equal to the documented tier.
- **Board: 8/8, AUTO selects the documented tier on the real bitstream at every
  boundary, board raw bit-exact to native.** Per-fixture JSON in `*.json`.

## Gate 3 - mailbox correctness / latency (correctness PASS; latency above projection)

`src/stress_kv260_v06.cpp` (persistent mixed-command stress with an in-process
v05 raw/telemetry oracle) + `tools/remote_gate3_recovery.sh`. Seed
`0x544333360906`, sequence seed `4294967290` (near UINT32_MAX to cross the wrap
within a handful of solves).

- **Mailbox correctness (PASS):** 10,000/10,000 mixed commands on the real CU
  (solve 8219, refresh_a 969, configure 812); `sequence_errors=0`,
  `parity_errors=0` (every solve bit-exact to the v05 oracle across 8219 varied
  inputs); `wrap_crossed=true`; normal STOP `stop=ok`. `stress10k.slim.json`.
- **Killed-host recovery (PASS):** an orphaned persistent host (SIGKILL) is
  recovered by `xmutil unloadapp`/`loadapp`; a fresh persistent host then
  CONFIGUREs and solves correctly (route FULL, valid raw+telemetry).
  `fresh.json`.
- **Latency p50 <= 25 us (NOT met; above-projection, per the plan's own gate-3
  fallback).** The 25 us figure was a preregistered projection. The fair
  persistent device round trip is 32.3 us (`10v_device_resident_v06`, tuned
  `--poll-sync never --poll yield`, no per-solve readback), above the projection.
  The stress harness runs `poll_sync=always` with a full raw+telemetry readback
  every solve (required to verify parity), whose device round trip on this board
  is ~1.14 ms (`fresh.json` `kernel_ns`, the same always-sync path): this board
  needs the uncached `poll_sync=never` path for the fast completion poll, so the
  stress p50 (1374 us) is the always-sync verification path, not the deployment
  latency. Persistent service is supported and correct; its latency on this QP
  size is above the projection, consistent with the honest bottom line that the
  FPGA is slower and costlier than one A53 core here.

### Known harness limitation (non-blocking)

`stress_kv260_v06` does not install a working SIGINT handler on the XRT backend:
a `kill -INT` mid-run did not drain/STOP/report and the process kept running (it
was recovered as the killed-host case above). Normal end-of-run STOP works. A
follow-up would port the stream host's `SignalGuard` into the stress binary so a
graceful SIGINT writes an `interrupted` report; not required by gate 3, whose
STOP and killed-host-recovery requirements are met.
