"""MSRPDL1 problem bundles shared by the host tools and the CPU baseline."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"MSRPDL1\0"


def write_bundle(path, A, b, C, d, x0, k0, tol, lower, upper, iters, cap,
                 has_lower=True, has_upper=True):
    n, m = A.shape[0], C.shape[0]
    norms = (C * C).sum(axis=1)
    scale = np.where(norms > 0, 1.0 / np.sqrt(np.where(norms > 0, norms, 1)), 0.0)
    G = C @ C.T
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<7I", 1, n, m, iters, cap,
                            int(has_lower), int(has_upper)))
        f.write(struct.pack("<4d", k0, tol, lower, upper))
        for a in (A, b, C, d, norms, scale, G, x0):
            f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())


def read_bundle(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == MAGIC
    _, n, m, iters, cap, hl, hu = struct.unpack_from("<7I", raw, 8)
    k0, tol, lower, upper = struct.unpack_from("<4d", raw, 36)
    arr = np.frombuffer(raw, dtype="<f8", offset=68)
    pos = 0

    def take(count, shape):
        nonlocal pos
        v = arr[pos:pos + count].reshape(shape).copy()
        pos += count
        return v

    A = take(n * n, (n, n))
    b = take(n, (n,))
    C = take(m * n, (m, n))
    d = take(m, (m,))
    pos += 2 * m + m * m
    x0 = take(n, (n,))
    return dict(n=n, m=m, iters=iters, cap=cap, has_lower=bool(hl),
                has_upper=bool(hu), k0=k0, tol=tol, lower=lower, upper=upper,
                A=A, b=b, C=C, d=d, x0=x0)
