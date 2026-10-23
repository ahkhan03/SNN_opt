# Regime diagnosis, 2026-10-23: v08 against the legacy row-interleave image

Why this directory exists: the alt-investment revision measured the qualified
v08 image at 2.1x the per-outer-iteration latency of the legacy
row-interleave WINNER image on its horizons (n = m = T = 16..80, K = 30000)
at equal, board-verified 200 MHz kernel clocks, and a full cycle accounting
was built to find the cause before choosing any remedy. Verdict: a regime
crossover, not a defect. v08 trades matvec width (8 products/cycle, RG 2 x
UF 4, against the legacy image's 32, RG 4 x UF 8) for URAM residency, one
fused 8-lane candidate scan per event, and current box-in-sweep semantics.
Event-heavy shapes (small m, tens to hundreds of events per iteration) are
where it wins big; matvec-heavy square shapes are where the width gap
dominates. The adopted routing rule lives in `fpga/CURRENT` and the package
README's regime-boundary section.

Files:

- `cycle_diagnosis.md`: per-iteration cycle accounting for both kernels,
  term by term with citations (residuals within +-6% on the four paper
  horizons and +-3% on the economic-dispatch fleet with ONE constant set),
  the event-count grounding (two box-semantics families), the
  confirmed/killed hypotheses, and the missing-evidence list.
- `remedy_options.md`: the quantified remedy options (A regime boundary,
  B bounded tweaks, C continuous event pipeline) and the recommendation,
  with a promotion note naming what was adopted.
- `csynth/`: the load-bearing Vitis HLS 2022.1 reports from a fresh csynth
  of the qualified v08 source (solution `solution_200mhz`, reports dated
  2026-10-23T20:17, `build/run_hls.tcl` reused unchanged): top level,
  `solve_current`, the resident matvec/scan/update pipelines, and the
  verbose schedule report behind the operation-free-state counts. One
  normalization was applied to two of these files: the build host's absolute
  source prefix was rewritten to the package-relative `src/` form (and the
  build-directory prefix to a neutral path); all numbers are unmodified.
  These are synthesis estimates for diagnosis; the routed timing and
  utilization evidence of the qualified image stays in
  `../board_2026-10-11_c2321d6f0833/`.

Measurement anchors (not mirrored here): the alt-investment paper's result
blocks `22_fpga_v08.yaml` (v08 arm, `clock_record.kernel_hz_measured`
199999998) and `20_fpga_protocol.yaml` / `19_fpga_resources.yaml` (legacy
image arm, same kernel clock), plus the economic-dispatch fleet logs in
`../board_2026-10-11_c2321d6f0833/` (`telemetry_by_call` event words:
60.912 events/iter at n = 10 down to 856.504 at n = 320, m = 2).

Reproduce: a csynth rerun from `fpga/kv260_v08/build/run_hls.tcl` on the
Vitis 2022.1 build host takes about 12 minutes (csynth only; no link, no
implementation).

- 2026-10-24 addendum: `lever_c_board_result_2026-10-24.md` — the Option C
  continuous event pipeline was built, qualified bit-identical, routed at
  55.4% LUT, and measured 1.7-2.0x SLOWER per iteration on this fleet: a
  dead end in that form; the fused-event root cause is open. The BRAM
  wide-matvec lineage (fpga/kv260_v09_bram) qualified the same day and is
  the resolver's arm for square matvec-heavy shapes inside n,m <= 128.
