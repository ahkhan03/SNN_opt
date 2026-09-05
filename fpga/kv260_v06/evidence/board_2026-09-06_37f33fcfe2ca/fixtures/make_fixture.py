#!/usr/bin/env python3
"""Write the three binary64 problem bundles used for the v06 board parity smoke."""

import struct

import numpy as np


def write(path, n, m, seed, iters=50, projmax=512, lower=-0.75, upper=0.75, k0=None):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-0.3, 0.3, (n, n))
    a = q @ q.T / n + 0.2 * np.eye(n)
    b = rng.uniform(-0.5, 0.5, n)
    c = rng.uniform(-0.4, 0.4, (m, n))
    d = rng.uniform(-0.5, -0.2, m)
    cns = (c * c).sum(1)
    row_scale = 1.0 / np.sqrt(cns)
    g = c @ c.T
    x0 = rng.uniform(-0.2, 0.2, n)
    if k0 is None:
        k0 = 0.9 / np.linalg.eigvalsh(a).max()
    with open(path, "wb") as f:
        f.write(b"MSRPDL1\0")
        f.write(struct.pack("<7I", 1, n, m, iters, projmax, 1, 1))
        f.write(struct.pack("<4d", k0, 1e-6, lower, upper))
        for arr in (a, b, c, d, cns, row_scale, g, x0):
            f.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())


if __name__ == "__main__":
    for n, m, seed in ((6, 18, 11), (20, 60, 23), (64, 64, 37)):
        write(f"prob_{n}x{m}.bin", n, m, seed)
        print("wrote", f"prob_{n}x{m}.bin")
