# Square alt-investment fixtures

MSRPDL1 binary64 bundles at n = m = T for T in {16, 32, 64, 80}. Two iteration
counts each: K = 2000 (x86 parity) and K = 30000 (later board latency). The
K = 30000 files are generated here and are not in `build/run_parity.sh`.
After the matvec was split into two four-wide partials, an x86 run of these
four against v06 matched every raw state word and all 16 telemetry words at
every T, including T = 80 at K = 30000.

Provenance is `22a_fpga_v08.py` in
`the alt-investment paper tree (analysis/1_scripts/)`.
`make_square_T.py` calls `common.alt_invest_qp.build_qp` with that script's
REF (r = 0.04, gamma_call = 0.23, gamma_dist = 0.15, c_max = 4, u_max = 10,
n_des = 15, lmbda = 5). k0 and the row scale come from the paper venv's
snn_opt 0.8.1 `SNNSolver` (`k0_scale = 0.05`, box [0, 4]). G is C C^T and x0
is 0. projmax stays 10000. The two K variants of one T differ only in the
iteration field of the header; k0 is identical because it is
`0.05 / lambda_max(A)` and does not depend on K.

| file | k0 | sha256 |
| --- | --- | --- |
| square_T16_K2000.bin | 0.018927813 | 254091d8a3bfffa0879ab36c2c5db706452a59da0900bb7f33963c0816926e90 |
| square_T16_K30000.bin | 0.018927813 | a7376371627e22cb7d89f3308bba6771db933b2e3bd5e7efe417ad2042834a48 |
| square_T32_K2000.bin | 0.017053972 | c2b1eb3f9014959d47a5b939eccbfbd5424c706de336ce4647db1dcc8152ae3e |
| square_T32_K30000.bin | 0.017053972 | ed40f946b8b71edbc085f17e0b6992fc4853c8d424d38aa2fde14f678512759e |
| square_T64_K2000.bin | 0.023330401 | 16271c7a88c86a937ae9aa2503dead796817241099e46da19897ee0a7e371ab1 |
| square_T64_K30000.bin | 0.023330401 | 8c14fed570bbfed7cf4d24167a4ca709522d4f1f41d3961562ff14a9611a3721 |
| square_T80_K2000.bin | 0.027484454 | a1b8c07b8e180c5c66ad8d226eeb4ae66ee967f1ed4a864f0ab0826d9f3ff712 |
| square_T80_K30000.bin | 0.027484454 | 5edec5d00ccdaa183e4b01d8e5c01ed306777c212d27f0cfe8606f23e6835a4f |

`provenance.json` beside this file is the machine-readable copy, including
per-array absolute maxima. Every shipped value is inside ap_fixed<32,8>.
