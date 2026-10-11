"""Gate G3 of the v08 plan: the v3 cycle model with the v08 synthesis depths.

The cycle model and its calibration live in the programmable-datapath design
study (fpga/design_studies/programmable_datapath, refit_v3.py and
cycle_model.py). Run this file from that directory. Loop depths are those of
the qualified build (HLS clock uncertainty 2.5 ns); the residual update for
m <= 8 is a straight-line block counted at its 8-to-11-cycle body latency.
"""
import sys

import yaml

sys.path.insert(0, ".")
import cycle_model  # noqa: E402
import refit_v3  # noqa: E402

cl, call = cycle_model.cl, refit_v3.call
fitted = yaml.safe_load(open("results_v3.yaml"))["params"]
P = dict(c_call=fitted["c_call"], c_fn=fitted["c_fn"])
A53_BEST_US = {160: 324.3, 320: 1160.5}  # lean linear-scan harness, best of 1 and 4 cores


def cost_v08(c):
    n, m, er, eb = c["n"], c["m"], c["events_row"], c["events_bnd"]
    e = er + eb
    straight = 11
    small_m = m <= 8
    update = straight if small_m else call(cl(m, 8), (1, 10), P)
    parts = dict(
        hess=cl(n, 2) * (call(cl(n, 4), (1, 15), P) + 2),
        grad=call(n, (1, 9), P),
        resid=cl(m, 2) * (call(cl(n, 4), (1, 16), P) + 2),
        scans=(e + 1) * call(cl(max(n, m), 8), (1, 10), P),
        events=e * refit_v3.C_EVENT,
        bound=eb * update,
        row=er * (refit_v3.D_DIV + call(cl(n, 8), (1, 10), P) + update),
    )
    return sum(parts.values()), parts


def main():
    seen = set()
    for c in sorted(cycle_model.load_ed()[0], key=lambda c: c["n"]):
        if c["n"] in seen or c["n"] not in A53_BEST_US:
            continue
        seen.add(c["n"])
        cycles, parts = cost_v08(c)
        us = cycles / c["clock_mhz"]
        split = ", ".join(f"{k}:{v / c['clock_mhz']:.1f}" for k, v in parts.items())
        print(f"n={c['n']}: {us:6.1f} us  A53/v08={A53_BEST_US[c['n']] / us:4.2f}  ({split})")


if __name__ == "__main__":
    main()
