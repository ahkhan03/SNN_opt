# Lean CPU baseline (double precision)

The CPU side of the FPGA-versus-CPU comparison: the same fixed-horizon SNN-QP
solve as the accelerator, written as a lean best-effort C++ program. It
replaces the generic-core harness `fpga/kv260_v05/src/cpu_baseline_v05.cpp`
(left unchanged) as the reference for the board's Cortex-A53 numbers.

## Build and run

    fpga/cpu_baseline/build_cpu_baseline.sh [output-dir]   # prints sha256
    OMP_NUM_THREADS=N cpu_baseline <bundle.bin> <output.bin> \
        [--mode canonical|dykstra] [--warmups N] [--reps N] [--energy-reps N]

Input is the binary64 problem bundle read by `msrp_v05::load_problem`
(`fpga/kv260_v05/src/msrp_bundle.hpp`, included unchanged). The output file and
the stdout timing and energy-mode protocol (`READY`/`GO`/`DONE ...`) are those
of `cpu_baseline_v05`. Work counters are additional keys on the same stdout
JSON object (`mode`, `rows`, `lower`, `upper`, `total`, `cap_rechecks` for
canonical; `mode`, `rounds`, `row_steps`, `guard_hits`, `round_caps` for
dykstra) and extra trailing fields on the energy-mode `DONE` line. The 16
telemetry words in the output file have no spare slot and are unchanged
(word 14 still carries the v0.5 constant 64).

## Modes

- `canonical` (default): the canonical SNN-QP event sweep with the library
  core's semantics for scalar bounds plus explicit rows. Gradient step, then
  the winner-take-all projection sweep over rows (normalised residual), lower
  facets and upper facets, strict first maximum, leave when the best violation
  is at most `constraint_tol`, projection cap with a fresh recheck. Row, lower,
  upper and total event counts, the event digest and the first/last candidate
  are bit-identical to the library observer.
- `dykstra`: Dykstra over the box member and the rows. Each round clips
  `x + p`, then removes row violations greedily with exact steps, with the
  residual updated incrementally and the stepped row reset to exactly zero;
  stops at `max|z - x| < 1e-12` (row-step guard 1000, round cap 100000).
  Reports rounds and row steps.

## What makes it fast without changing results

Everything is a layout or loop-shape change; every value is computed by the
same operations in the same order as the library core.

- The Hessian product uses a copy of `A` stored in column blocks of 16 outputs
  (one contiguous stream per block), so each output element is accumulated
  over `k = 0..n-1` in order (identical to the row dot product) with 16
  accumulators in registers.
- Row dot products (residuals) interleave 2-4 rows, each row summed in order.
  The facet update reads a transposed copy of `C` (one column contiguous).
- The winner-take-all search is a full linear scan per event: all m row
  scores and all 2n facet scores are evaluated, and the strict first maximum
  in the order rows, lower facets, upper facets is selected (a tie goes to
  the earlier family, then to the lower index; +0 and -0 compare equal, as in
  a sequential scan). Pass 1 is a value-only maximum (exact, so order free)
  with independent accumulators: on AArch64 two-wide NEON `fmaxnm` over
  chunks of 8 elements, fusing the lower and upper families over one load of
  `x`, with one partial maximum per chunk stored; elsewhere a scalar version
  with the same chunking (`-DLEAN_PORTABLE_SCAN` forces it). Pass 2 runs for
  the winning family only: the first chunk whose partial equals the maximum,
  then the first element inside it that compares equal. The reported score is
  recomputed at the chosen index. About 2.5 cycles per candidate on the
  Cortex-A53 versus 9-11 for a sequential compare-and-select scan.
- Numeric kernels are `noinline`, so the single-thread and team paths run the
  same compiled code and cannot differ in multiply-add contraction.
- No allocation in the iteration loop. With `OMP_NUM_THREADS > 1` one
  persistent OpenMP team spans the whole horizon: threads split the Hessian
  product by output column range, the serial sweep runs on thread 0, two
  barriers per iteration. The result is bitwise identical to one thread.
  Suggested for measurement: `OMP_PROC_BIND=close OMP_WAIT_POLICY=active`.

The block and transposed copies are built once after loading, outside the timed region
(the same status as the bundle load).

## Capability boundary

- Problem class: dense `A`, scalar lower and/or upper bound, explicit rows
  (`m >= 1`), fixed horizon, no early stopping. No cones, no per-coordinate
  bounds, no diagonal-Hessian path.
- `n, m <= 1024`.
- binary64 only (a float32 bundle is rejected).
- Canonical mode abort: projection cap with violation above tolerance gives
  exit status 3, as in the library.

## Equal-footing facts

- Precision of record is binary64: the library, every prior baseline and the
  definition of the solver's tolerances and event sequence are binary64. The
  FPGA kernel computes in `ap_fixed<32,8>`; there is no float32 CPU build.
- The argmax is a full linear scan per event, as in the FPGA kernel: every
  row and facet score is evaluated at every event.
- Compiler flags are `-O3 -march=native -mtune=native -funroll-loops -fopenmp`.
  No `-ffast-math` and no reassociation or reciprocal flags, so no reduction is
  reordered. The default floating-point contraction (FMA on aarch64) is left
  on; the tests pin `-ffp-contract=off` on both sides to show bitwise equality
  with the generic core, and check the default build to rounding level.
- The reference for equivalence is the generic library core compiled without
  OpenMP SIMD reductions (sequential summation).

## Qualification record

KV260 (4x Cortex-A53 at 1.333 GHz, Ubuntu 22.04, gcc 11.4.0), built with
`build_cpu_baseline.sh` from this tree (binary sha256
`0f511cece6213b4d2a7dcfc3820aa4640e730f79f6552c1876856ccef8c4c04a`), run with
`OMP_PROC_BIND=true`, one warm-up and five timed repetitions of 2000 outer
iterations on the economic-dispatch band fleet (m = 2, box [0, 1]).

- Canonical outputs agree with the generic-core baseline
  (`cpu_baseline_v05` with the 1024 dimension cap) at n = 10, 40, 160, 320:
  all 16 telemetry words identical (event counts, digest, first and last
  candidate) at one and four threads; x within 2.9e-15 at one thread (FMA
  placement) and bitwise equal at four. One and four threads are bitwise
  equal in x.
- Median time per outer iteration, microseconds:

| n | generic core, 1 thread | canonical, 1 thread | canonical, 4 threads | dykstra, 1 thread | dykstra, 4 threads |
|---|---|---|---|---|---|
| 10 | 19.1 | 12.8 | 14.4 | 25.5 | 27.3 |
| 40 | 89.2 | 35.9 | 35.8 | 80.4 | 80.5 |
| 160 | 1201.9 | 351.5 | 324.3 | 347.3 | 320.5 |
| 320 | 4706.2 | 1247.5 | 1160.5 | 849.9 | 744.0 |

Build with `-DLEAN_PROFILE` to print a per-section wall-clock profile to
stderr (timer calls add overhead; use it for the split, not for headline
numbers).

Tests: `tests/test_fpga_cpu_baseline.py`.
