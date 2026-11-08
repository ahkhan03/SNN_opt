# v09b host refresh, 2026-11-08

The host binary now records the quantized decay schedule and rejects a floor
in the near-floor band. The kernel source and the bitstream are unchanged:
this is a host-only change against the qualified
`snn_qp_v09b_k0d_200m_39fa7e234609` application.

## What was re-run

The bit-identity matrix was run on the board with the rebuilt host
(sha256 `f1e3e295af190580c8625e5de2a6eac2845ceca58819ec27382c0c17c93c0da1`,
built by `build/build_host.sh` on the board, XRT backend). It covers 13
fixtures in both launch modes plus the two pinned decay cells in both modes:
30 cells, compared against the records that qualified the decay image.

```
MATRIX pass=30 fail=0
```

Every cell matched the reference on raw state, all 16 telemetry words, the
mailbox error, and the selected route, and every host JSON reported
`backend:"xrt"`. The pinned decay cells still report four levels and the same
final effective step.

## Automatic floor selection

A stiff three-variable bundle was used because the framework fixtures return
`no_decay` from the recommender (their residual improves with iterations).
The adapter ran on the board and selected floor 709 quanta, segment 666, and a
twelve-level chain. The host ran that schedule in both launch modes:

- start 1227489 quanta, floor 709 quanta, segment 666, levels 12,
  final effective `4.225969314575195e-05`;
- `cmp_out.py --json ... --decision ...` reported MATCH in both modes;
- both runs exited 0 with status 0 and `backend:"xrt"`.

This confirms the decision, the host arguments, and the recorded schedule
agree on silicon.

## Clock

The kernel clock was not re-measured in this run: `kria-clock` needs a
password for the debug clock node on this board. The application is the
qualified image installed on 2026-11-08, whose record reports
`kernel_hz_measured: 199999998`; a host-only change cannot alter it.

## Records

Raw dumps, host JSONs, per-cell comparisons, and the decision record are on
the board under `/home/ubuntu/host_refresh_2026-11-08/results/`, beside the
rebuilt host in `host_build/`. The reference records are the decay-image
qualification records kept beside that build archive on the board.
