# Restricted v0.5 KV260 reference

This subtree preserves the physically qualified Kria K26 implementation of
the SNN-QP v0.5 recurrence. It is an additive reference implementation, not
an alternative public solver API and not evidence that every `snn_opt`
problem is supported on FPGA.

## Qualification record

The source was exercised by the SNN-MSRP paper at paper commit `e7c96c6`.
The accepted physical run is `official-20260728-b09b3d7`, generated from
measurement source commit `b09b3d7`.

- Result YAML SHA-256:
  `c69d761795be942cb8a0629e91b9cc18be2f8991b0d87205fbb90bb41fded98f`
- Result NPZ SHA-256:
  `4022000f5900e98eb1263fcfcb3ff2f905cf51e38683d9b922fac0d6592b0f96`
- Kernel SHA-256:
  `927e30945d07aae941f46655c3d8bfa0a7715ff5d704f1f926664505095d6ea1`
- Fixed-point header SHA-256:
  `d000c064e08f947621a093a506812d14064b9a3b2ded4e81263b0d538e43936a`
- Canonical C-core SHA-256:
  `5c1931a848dd096b7d11e331b1b1e43dbea8869b2bd8ee86a3d4eceb1389d0cf`

All nine paper cells passed physical recurrence, recovered-state, and
portfolio-feasibility gates. The routed design closed timing at 200 MHz on
the recorded K26 platform. The paper result, not this source directory, owns
the workload-specific measurements and scientific claims.

## Preserved contract

The reference keeps the paper compatibility ABI unchanged:

- top function `snn_qp_v05`;
- binary64 host-to-kernel input transport;
- `ap_fixed<32,8,AP_RND_CONV,AP_SAT>` state;
- `ap_fixed<48,16,AP_RND_CONV,AP_SAT>` accumulation;
- one normalized-distance candidate family ordered as explicit rows, lower
  facets, then upper facets, with strict first-maximal ties;
- fresh joint-feasibility recheck at the projection cap;
- status 2 when the cap is exhausted with unresolved infeasibility;
- no terminal clipping;
- fixed-horizon execution;
- scalar lower and upper bounds; and
- `1 <= n <= 64` and `1 <= m <= 64`.

The `msrp_v05` namespace, bundle magic, telemetry magic, and file names remain
unchanged so the imported source can be compared directly with the qualified
paper path.

## Qualification boundary

Physical qualification is narrower than the structural source capability.
It covers `n` in `{10, 25, 50}`, `m` in `{13, 28, 53}`, an active scalar
lower bound, no active upper bound, successful fixed-horizon runs, and inputs
inside the selected fixed-point range.

This release does not claim physical qualification for:

- box-only problems with `m == 0`;
- vector-valued bounds;
- early stopping or convergence behavior;
- diagonal or eigenbasis modes;
- dimensions above 64;
- active upper-bound facets;
- the projection-budget-abort path;
- range-violation telemetry; or
- arbitrary QPs outside the measured fixed-point envelope.

The historical `archive/fpga-implementation` branch predates v0.5 semantics
and is retained only as legacy engineering history. Do not merge its kernel
into this subtree.

## Build

The recipes target Vitis 2022.1 and the platform recorded in `env.yaml`.

```bash
fpga/kv260_v05/build/build_native.sh
vitis_hls -f fpga/kv260_v05/build/run_hls.tcl
fpga/kv260_v05/build/build_xclbin.sh
```

Generated HLS, Vitis, bitstream, and deployment artifacts are ignored. A
rebuild is validated by source and toolchain identity, timing and utilization
reports, and fresh board parity. Byte equality with the paper xclbin is not
required because implementation outputs can depend on path and tool state.
