#!/usr/bin/env python3
"""Bit-accurate software transliteration of the Family-13 HLS solve.

This module is deliberately a *kernel* model, rather than an adapter around
``SNNSolver``.  The loop below has the same phases as ``snn_qp_v13``:

``row-interleave matvec -> gradient register -> state commit -> sequential
candidate scan -> exact native reset -> next candidate``.

The model is useful on a machine without Vitis.  NumPy supplies host-side
arrays and scalar conversion, while every value that crosses a datapath
register is rounded and saturated by the declared formats.  Accuracy checks
compare this path with an independent binary64 solver; no old-width callback
is treated as the deployment oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Family-13 K26 contract.  The state remains F24; the internal widths mirror
# ``dt.h`` and are intentionally independent of the historical Family-12
# software-only no-intermediate-rounding bound.
STATE_WIDTH = 32
STATE_INTEGER_BITS = 8
STATE_FRACTIONAL_BITS = STATE_WIDTH - STATE_INTEGER_BITS
NORM_WIDTH = 68
# Four extra integer bits retain the official eight-coordinate F24 ball
# envelope (8 * 128^2) while still rejecting a block whose worst-case sum
# cannot fit the norm register.  The binary point remains F48.
NORM_INTEGER_BITS = 20
NORM_FRACTIONAL_BITS = NORM_WIDTH - NORM_INTEGER_BITS
NORM_PRODUCT_WIDTH = 64
NORM_PRODUCT_INTEGER_BITS = 16
NORM_PRODUCT_FRACTIONAL_BITS = NORM_PRODUCT_WIDTH - NORM_PRODUCT_INTEGER_BITS
RSQRT_WIDTH = 49
RSQRT_INTEGER_BITS = 25
RSQRT_FRACTIONAL_BITS = RSQRT_WIDTH - RSQRT_INTEGER_BITS
PRODUCT_WIDTH = 48
PRODUCT_INTEGER_BITS = 16
PRODUCT_FRACTIONAL_BITS = PRODUCT_WIDTH - PRODUCT_INTEGER_BITS
GRADIENT_WIDTH = 48
GRADIENT_INTEGER_BITS = 16
GRADIENT_FRACTIONAL_BITS = GRADIENT_WIDTH - GRADIENT_INTEGER_BITS
CONSTANT_WIDTH = 32
CONSTANT_INTEGER_BITS = 8
CONSTANT_FRACTIONAL_BITS = CONSTANT_WIDTH - CONSTANT_INTEGER_BITS

# Fixed-point reciprocal-root implementation constants.  ``work`` holds the
# normalized mantissa/root, ``work_product`` holds the Newton products, and
# ``scale`` provides headroom while applying the power-of-two exponent.
RSQRT_WORK_WIDTH = 32
RSQRT_WORK_INTEGER_BITS = 2
RSQRT_WORK_FRACTIONAL_BITS = RSQRT_WORK_WIDTH - RSQRT_WORK_INTEGER_BITS
RSQRT_PRODUCT_WIDTH = 48
RSQRT_PRODUCT_INTEGER_BITS = 4
RSQRT_SIGNED_WIDTH = 48
RSQRT_SIGNED_INTEGER_BITS = 4
RSQRT_SCALE_WIDTH = 64
RSQRT_SCALE_INTEGER_BITS = 32
RSQRT_LUT_BITS = 5
RSQRT_NEWTON_STEPS = 2
RSQRT_LUT = tuple(
    1.0 / math.sqrt(1.0 + (i + 0.5) / (1 << RSQRT_LUT_BITS)) for i in range(1 << RSQRT_LUT_BITS)
)

MAXN = 64
MAXM = 64
RG = 2
UF = 2
DEFAULT_HORIZON = 32768
DEFAULT_PROJECTION_CAP = 64
DEFAULT_TOL = 1.0e-6
K0_SCALE = 0.03
UINT64_MASK = (1 << 64) - 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
NO_CANDIDATE = UINT64_MASK


@dataclass
class FormatStats:
    rounded: int = 0
    saturations: int = 0
    nonfinite: int = 0


@dataclass(frozen=True)
class FixedFormat:
    """A small ``ap_fixed``/``ap_ufixed`` equivalent.

    ``integer_bits`` follows Vitis semantics: a signed format includes its
    sign bit.  ``np.rint`` implements the required round-to-nearest-even tie
    rule.  A mutable stats object is supplied separately so frozen format
    descriptors remain safe to share between arrays.
    """

    name: str
    width: int
    integer_bits: int
    signed: bool
    stats: FormatStats = field(default_factory=FormatStats, compare=False, repr=False)

    def __post_init__(self) -> None:
        frac = int(self.width) - int(self.integer_bits)
        if self.width <= 0 or frac < 0:
            raise ValueError("invalid fixed-point format")
        if not self.signed and self.integer_bits == 0:
            raise ValueError("unsigned format needs an integer bit")
        object.__setattr__(self, "fractional_bits", frac)
        object.__setattr__(self, "quantum", float(2.0**-frac))
        if self.signed:
            lo = -float(2.0 ** (self.integer_bits - 1))
            hi = float(2.0 ** (self.integer_bits - 1) - self.quantum)
        else:
            lo = 0.0
            hi = float(2.0**self.integer_bits - self.quantum)
        object.__setattr__(self, "minimum", lo)
        object.__setattr__(self, "maximum", hi)

    def q(self, values):
        """Round and saturate a scalar or ndarray at this register boundary."""
        scalar = np.isscalar(values)
        arr = np.asarray(values, dtype=float)
        self.stats.rounded += int(arr.size)
        bad = ~np.isfinite(arr)
        if np.any(bad):
            self.stats.nonfinite += int(np.count_nonzero(bad))
            arr = np.where(bad, 0.0, arr)
        clipped = np.clip(arr, self.minimum, self.maximum)
        self.stats.saturations += int(np.count_nonzero(arr != clipped))
        out = np.rint(clipped / self.quantum) * self.quantum
        # Protect the endpoint from a one-bit floating scaling overshoot.
        out = np.clip(out, self.minimum, self.maximum)
        return float(out) if scalar else np.asarray(out, dtype=float)


@dataclass
class DatapathStats:
    guard_hits: int = 0
    rsqrt_calls: int = 0
    rsqrt_guarded_calls: int = 0
    rsqrt_relative_error_max: float = 0.0
    rsqrt_ulp_error_max: float = 0.0
    norm_sq_max: float = 0.0
    projector_evaluations: int = 0
    reset_events: int = 0
    branch_counts: dict[str, int] = field(
        default_factory=lambda: {
            "inside": 0,
            "radial": 0,
            "axial": 0,
            "apex": 0,
            "zero": 0,
            "boundary": 0,
        }
    )
    guard_histogram: dict[str, int] = field(
        default_factory=lambda: {
            "zero": 0,
            "(0,0.5]": 0,
            "(0.5,0.9]": 0,
            "(0.9,0.99]": 0,
            "(0.99,1]": 0,
            "(1,1.01]": 0,
            "(1.01,1.1]": 0,
            "(1.1,2]": 0,
            "(2,inf)": 0,
            "threshold_zero": 0,
        }
    )


class F24Datapath:
    """Named register formats and operations used by the direct loop.

    The name is retained for compatibility with the Family-12 harness.  It
    now models the Family-13 K26 path, including the normalized fixed-point
    LUT/Newton reciprocal-root core.
    """

    def __init__(self, rsqrt_model: str = "fixed_nr") -> None:
        if rsqrt_model not in {"fixed_nr", "exact", "bounded_ulp"}:
            raise ValueError("rsqrt_model must be fixed_nr, exact, or bounded_ulp")
        self.state = FixedFormat("ap_fixed<32,8>", 32, 8, True)
        self.norm_sq = FixedFormat(
            f"ap_ufixed<{NORM_WIDTH},{NORM_INTEGER_BITS}>", NORM_WIDTH, NORM_INTEGER_BITS, False
        )
        self.norm_product = FixedFormat(
            "ap_fixed<64,16>", NORM_PRODUCT_WIDTH, NORM_PRODUCT_INTEGER_BITS, True
        )
        self.rsqrt_out = FixedFormat(
            f"ap_ufixed<{RSQRT_WIDTH},{RSQRT_INTEGER_BITS}>", RSQRT_WIDTH, RSQRT_INTEGER_BITS, False
        )
        self.product = FixedFormat(
            f"ap_fixed<{PRODUCT_WIDTH},{PRODUCT_INTEGER_BITS}>",
            PRODUCT_WIDTH,
            PRODUCT_INTEGER_BITS,
            True,
        )
        self.gradient = FixedFormat(
            f"ap_fixed<{GRADIENT_WIDTH},{GRADIENT_INTEGER_BITS}>",
            GRADIENT_WIDTH,
            GRADIENT_INTEGER_BITS,
            True,
        )
        self.constants = FixedFormat(
            f"ap_fixed<{CONSTANT_WIDTH},{CONSTANT_INTEGER_BITS}>",
            CONSTANT_WIDTH,
            CONSTANT_INTEGER_BITS,
            True,
        )
        self.k0 = FixedFormat(
            f"ap_fixed<{PRODUCT_WIDTH},{PRODUCT_INTEGER_BITS}>",
            PRODUCT_WIDTH,
            PRODUCT_INTEGER_BITS,
            True,
        )
        self.rsqrt_work = FixedFormat(
            "ap_ufixed<32,2>", RSQRT_WORK_WIDTH, RSQRT_WORK_INTEGER_BITS, False
        )
        self.rsqrt_product = FixedFormat(
            "ap_ufixed<48,4>", RSQRT_PRODUCT_WIDTH, RSQRT_PRODUCT_INTEGER_BITS, False
        )
        self.rsqrt_signed = FixedFormat(
            "ap_fixed<48,4>", RSQRT_SIGNED_WIDTH, RSQRT_SIGNED_INTEGER_BITS, True
        )
        self.rsqrt_scale = FixedFormat(
            "ap_ufixed<64,32>", RSQRT_SCALE_WIDTH, RSQRT_SCALE_INTEGER_BITS, False
        )
        self.stats = DatapathStats()
        self.rsqrt_model = rsqrt_model
        self.threshold = float(self.norm_sq.q(self.state.quantum**2))
        self._recorded_rsqrts = 0

    @staticmethod
    def io_float32(values):
        """Apply the IEEE binary32 conversion at an m_axi boundary only.

        Internal state reads and writes must not call this helper.  Keeping the
        conversion in the explicit ``state_in``/``state_out`` boundary methods
        makes an accidental per-step re-quantization visible in the model.
        """
        arr = np.asarray(values, dtype=float)
        out = arr.astype(np.float32).astype(np.float64)
        return float(out) if np.isscalar(values) else out

    def state_in(self, values):
        """Initial host-to-kernel state ingress (the sole input wire cast)."""
        return self.state.q(self.io_float32(values))

    def state_out(self, values):
        """Final kernel-to-host state egress (kept separate from reset writes)."""
        return self.state.q(self.io_float32(values))

    def state_commit(self, values):
        """Write an internal value to the F24 state register, fixed only."""
        return self.state.q(values)

    def _histogram(self, n2: float) -> None:
        if self.threshold <= 0.0:
            self.stats.guard_histogram["threshold_zero"] += 1
            return
        ratio = float(n2) / self.threshold
        if ratio == 0.0:
            key = "zero"
        elif ratio <= 0.5:
            key = "(0,0.5]"
        elif ratio <= 0.9:
            key = "(0.5,0.9]"
        elif ratio <= 0.99:
            key = "(0.9,0.99]"
        elif ratio <= 1.0:
            key = "(0.99,1]"
        elif ratio <= 1.01:
            key = "(1,1.01]"
        elif ratio <= 1.1:
            key = "(1.01,1.1]"
        elif ratio <= 2.0:
            key = "(1.1,2]"
        else:
            key = "(2,inf)"
        self.stats.guard_histogram[key] += 1

    def guard(self, n2: float) -> bool:
        self._histogram(float(n2))
        hit = bool(float(n2) <= self.threshold)
        if hit:
            self.stats.guard_hits += 1
        return hit

    def norm_square(self, values: Sequence[float]) -> float:
        # This is the scheduled product/reduction path, not a NumPy dot.
        vv = self.state.q(np.asarray(values, dtype=float).reshape(-1))
        acc = 0.0
        for value in vv:
            value_p = float(self.product.q(float(value)))
            term = float(self.norm_product.q(value_p * value_p))
            acc = float(self.norm_sq.q(acc + term))
        self.stats.norm_sq_max = max(self.stats.norm_sq_max, float(acc))
        return acc

    def rsqrt(self, n2: float) -> float:
        self.stats.rsqrt_calls += 1
        n2q = float(self.norm_sq.q(n2))
        if n2q <= 0.0:
            self.stats.rsqrt_guarded_calls += 1
            return 0.0
        exact = 1.0 / math.sqrt(n2q)
        if self.rsqrt_model == "exact":
            raw = exact
        elif self.rsqrt_model == "bounded_ulp":
            # Retain the old deterministic stress model as an optional
            # diagnostic.  The deployment/default path is ``fixed_nr``.
            sign = -1.0 if (self.stats.rsqrt_calls & 1) else 1.0
            raw = exact + sign * self.rsqrt_out.quantum
        else:
            raw = self._fixed_nr_rsqrt(n2q)
        result = float(self.rsqrt_out.q(raw))
        self.stats.rsqrt_relative_error_max = max(
            self.stats.rsqrt_relative_error_max,
            abs(result - exact) / abs(exact),
        )
        self.stats.rsqrt_ulp_error_max = max(
            self.stats.rsqrt_ulp_error_max,
            abs(result - exact) / self.rsqrt_out.quantum,
        )
        return result

    @staticmethod
    def _floor_log2_raw(raw: int) -> int:
        """Return the exponent represented by a positive norm raw integer."""
        return int(raw.bit_length() - 1 - NORM_FRACTIONAL_BITS)

    @staticmethod
    def _shift_fixed(value: float, fmt: FixedFormat, amount: int) -> float:
        """Bit-shift a fixed register, matching ``ap_fixed`` shift semantics."""
        raw = int(np.rint(float(value) * (2**fmt.fractional_bits)))
        if amount >= 0:
            shifted = raw >> int(amount)
        else:
            shifted = raw << int(-amount)
        return float(fmt.q(shifted / (2**fmt.fractional_bits)))

    def _fixed_nr_rsqrt(self, n2q: float) -> float:
        """LUT seed + two fixed-point Newton updates, no floating datapath.

        Python uses integer raw bits only for normalization/indexing; every
        arithmetic boundary then goes through the same QFormat as ``dt.h``.
        This is the software golden for the HLS implementation.
        """
        raw_n2 = int(np.rint(float(n2q) * (2**NORM_FRACTIONAL_BITS)))
        if raw_n2 <= 0:
            return 0.0
        exponent = self._floor_log2_raw(raw_n2)
        # Normalize with a register-width bit shift, as in the C++ core.
        normalized = self._shift_fixed(n2q, self.norm_sq, exponent)
        # A saturated norm can quantize the shifted mantissa to exactly 2.
        # The LUT is indexed for [1,2), so renormalize the endpoint and carry
        # the extra power of two into the exponent.
        normalized_raw = int(np.rint(normalized * (2**NORM_FRACTIONAL_BITS)))
        if normalized_raw >= 2 * (2**NORM_FRACTIONAL_BITS):
            normalized_raw >>= 1
            exponent += 1
            normalized = float(self.norm_sq.q(normalized_raw / (2**NORM_FRACTIONAL_BITS)))
        mantissa = float(self.rsqrt_work.q(normalized))
        if mantissa >= 2.0:
            mantissa = float(self.rsqrt_work.q(mantissa / 2.0))
            exponent += 1
        mantissa_raw = int(np.rint(mantissa * (2**RSQRT_WORK_FRACTIONAL_BITS)))
        index = (mantissa_raw >> (RSQRT_WORK_FRACTIONAL_BITS - RSQRT_LUT_BITS)) & (
            (1 << RSQRT_LUT_BITS) - 1
        )
        index = max(0, min((1 << RSQRT_LUT_BITS) - 1, int(index)))
        y = float(self.rsqrt_work.q(RSQRT_LUT[index]))
        half = float(self.rsqrt_signed.q(0.5))
        one_point_five = float(self.rsqrt_signed.q(1.5))
        for _ in range(RSQRT_NEWTON_STEPS):
            y_sq = float(self.rsqrt_product.q(y * y))
            xy_sq = float(self.rsqrt_product.q(mantissa * y_sq))
            half_xy_sq = float(self.rsqrt_signed.q(xy_sq * half))
            correction = float(self.rsqrt_signed.q(one_point_five - half_xy_sq))
            # The HLS expression first stores the product in the signed
            # correction register, then casts that register back to the
            # unsigned work format.  Keep both boundaries explicit here.
            next_y = float(self.rsqrt_signed.q(y * correction))
            y = float(self.rsqrt_work.q(next_y))

        half_exponent = exponent // 2
        odd_exponent = bool(exponent - 2 * half_exponent)
        scaled = float(self.rsqrt_scale.q(y))
        if odd_exponent:
            inv_sqrt_two = float(self.rsqrt_scale.q(1.0 / math.sqrt(2.0)))
            scaled = float(self.rsqrt_scale.q(scaled * inv_sqrt_two))
        scaled = self._shift_fixed(scaled, self.rsqrt_scale, half_exponent)
        return float(self.rsqrt_out.q(scaled))

    def product_q(self, value: float) -> float:
        return float(self.product.q(float(value)))

    def product_div_trunc(self, numerator, denominator):
        """Match Vitis ``ap_fixed`` division at the product binary point.

        ``ap_fixed_base::operator/`` forms an integer quotient after shifting
        the dividend by the denominator's fractional-bit count.  For two
        ``product_t`` values (F32/F32), the result therefore already has F32
        fractional bits and integer division truncates toward zero.  It does
        not invoke ``AP_RND_CONV`` because no fractional bits are discarded
        by a subsequent cast to the same F32 product format.  Keep this
        vendor operation explicit instead of using Python's nearest-even
        ``product_q(numerator / denominator)``.
        """
        scale = 2**PRODUCT_FRACTIONAL_BITS
        num_arr = np.asarray(numerator, dtype=float)
        den_arr = np.asarray(denominator, dtype=float)
        scalar = num_arr.ndim == 0 and den_arr.ndim == 0

        def one(num_value: float, den_value: float) -> float:
            den_raw = int(np.rint(float(den_value) * scale))
            if den_raw == 0:
                return 0.0
            num_raw = int(np.rint(float(num_value) * scale))
            # The fixed-point quotient retains F32 fractional bits, so the
            # dividend raw integer is shifted by F32 before integer divide.
            quotient = (abs(num_raw) << PRODUCT_FRACTIONAL_BITS) // abs(den_raw)
            if (num_raw < 0) != (den_raw < 0):
                quotient = -quotient
            return float(self.product.q(quotient / scale))

        if scalar:
            return one(float(num_arr), float(den_arr))
        nums = np.broadcast_to(num_arr, np.broadcast(num_arr, den_arr).shape)
        dens = np.broadcast_to(den_arr, nums.shape)
        out = np.empty(nums.shape, dtype=float)
        for index in np.ndindex(nums.shape):
            out[index] = one(float(nums[index]), float(dens[index]))
        return out

    def scalar_q(self, value: float) -> float:
        return float(self.constants.q(float(value)))

    def gradient_q(self, value: float) -> float:
        return float(self.gradient.q(float(value)))

    def scaled_soc_reset(self, x: np.ndarray, t_index: int, z_indices: Sequence[int], mu: float):
        """Evaluate one exact scaled-SOC reset with all named roundings.

        Returns ``(projected_state, squared_displacement, branch)``.  The
        state is not committed here; the scheduler commits the selected
        candidate, which mirrors the HLS scan/commit split.
        """
        self.stats.projector_evaluations += 1
        # The candidate is evaluated from the state register already resident
        # in the kernel.  Re-entering through ``state_in`` would incorrectly
        # model an m_axi float32 crossing on every reset candidate.
        arr = np.asarray(x, dtype=float).copy()
        zidx = tuple(int(i) for i in z_indices)
        z = self.state.q(arr[list(zidx)])
        t = float(self.state.q(arr[int(t_index)]))
        mu_q = self.scalar_q(mu)
        one_q = self.scalar_q(1.0)
        n2 = self.norm_square(z)
        guarded = self.guard(n2)
        inv = 0.0 if guarded else self.rsqrt(n2)
        # Both operands cross the product register before the multiply in
        # HLS (`norm_sq` and `rsqrt_t` are not implicitly promoted in the
        # same way as a Python float expression).
        n2_product = self.product_q(n2)
        inv_product = self.product_q(inv)
        zn = 0.0 if guarded else self.product_q(n2_product * inv_product)
        lhs = self.product_q(zn)
        rhs = self.product_q(mu_q * t)
        if lhs == rhs:
            self.stats.branch_counts["boundary"] += 1
        if lhs <= rhs:
            self.stats.branch_counts["zero" if guarded else "inside"] += 1
            return np.asarray(arr, dtype=float).copy(), 0.0, "zero" if guarded else "inside"

        axial_test = self.product_q(t + self.product_q(mu_q * zn))
        if axial_test <= 0.0:
            branch = "apex" if (guarded or abs(zn) <= self.state.quantum) else "axial"
            self.stats.branch_counts[branch] += 1
            out = np.asarray(arr, dtype=float).copy()
            out[int(t_index)] = 0.0
            out[list(zidx)] = 0.0
            return self.state_commit(out), self._displacement_sq(arr, out, (t_index, *zidx)), branch

        self.stats.branch_counts["radial"] += 1
        denom = self.product_q(one_q + self.product_q(mu_q * mu_q))
        numerator = self.product_q(t + self.product_q(mu_q * zn))
        t_new = self.product_div_trunc(numerator, denom) if denom else 0.0
        zn_new = self.product_q(mu_q * t_new)
        ratio = self.product_q(zn_new * self.product_q(inv)) if inv and not guarded else 0.0
        out = np.asarray(arr, dtype=float).copy()
        out[int(t_index)] = t_new
        # The HLS lateral multiply is held in the wide product register before
        # the state-register cast.  Keep that boundary explicit so values near
        # a state ULP follow the same scheduled path.
        out[list(zidx)] = self.state.q(self.product.q(ratio * z))
        return self.state_commit(out), self._displacement_sq(arr, out, (t_index, *zidx)), "radial"

    def scaled_soc_reset_batch(self, x: np.ndarray, contacts: int, mu: float):
        """Evaluate disjoint friction candidates in parallel across contacts.

        The HLS scheduler still scans and commits one winner at a time.  This
        helper only vectorizes the independent candidate *evaluation* in the
        software model; every array operation is rounded at the same register
        boundary as :meth:`scaled_soc_reset`.  It keeps the q=16 accuracy
        sweep tractable without changing the realized event order.
        """
        qn = int(contacts)
        if qn <= 0:
            return [], np.zeros(0, dtype=float), []
        # ``x`` is the resident F24 state.  This helper must not perform an
        # interface conversion while scanning/resetting candidates.
        arr = np.asarray(x, dtype=float).copy()
        tids = np.arange(0, 3 * qn, 3, dtype=int)
        z0ids = tids + 1
        z1ids = tids + 2
        t = np.asarray(self.state.q(arr[tids]), dtype=float)
        z0 = np.asarray(self.state.q(arr[z0ids]), dtype=float)
        z1 = np.asarray(self.state.q(arr[z1ids]), dtype=float)
        z0p = np.asarray(self.product.q(z0), dtype=float)
        z1p = np.asarray(self.product.q(z1), dtype=float)
        p0 = np.asarray(self.norm_product.q(z0p * z0p), dtype=float)
        p1 = np.asarray(self.norm_product.q(z1p * z1p), dtype=float)
        n2 = np.asarray(self.norm_sq.q(p0), dtype=float)
        n2 = np.asarray(self.norm_sq.q(n2 + p1), dtype=float)
        if n2.size:
            self.stats.norm_sq_max = max(self.stats.norm_sq_max, float(np.max(n2, initial=0.0)))
        guarded = n2 <= float(self.threshold)
        self._histogram_batch(n2)
        self.stats.guard_hits += int(np.count_nonzero(guarded))
        inv = np.zeros(qn, dtype=float)
        active_indices = np.flatnonzero(~guarded)
        if active_indices.size:
            inv[active_indices] = self.rsqrt_batch(n2[active_indices])
        mu_q = self.scalar_q(mu)
        one_q = self.scalar_q(1.0)
        zn = np.zeros(qn, dtype=float)
        active = ~guarded
        n2_product = np.asarray(self.product.q(n2[active]), dtype=float)
        inv_product = np.asarray(self.product.q(inv[active]), dtype=float)
        zn[active] = np.asarray(self.product.q(n2_product * inv_product), dtype=float)
        lhs = np.asarray(self.product.q(zn), dtype=float)
        rhs = np.asarray(self.product.q(mu_q * t), dtype=float)
        violation = lhs > rhs
        axial_term = np.asarray(self.product.q(mu_q * zn), dtype=float)
        axial = np.asarray(self.product.q(t + axial_term), dtype=float)
        apex = violation & (axial <= 0.0)
        radial = violation & ~apex
        # Only the candidate's three coordinates can change.  Keeping this as
        # a q-by-3 tile avoids materializing q full n-vectors for every scan;
        # the scheduler still commits exactly one tile below.
        local_before = np.column_stack((t, z0, z1))
        outputs = np.asarray(local_before, dtype=float).copy()
        # Keep the raw product/state registers separate from the returned
        # candidate so the winner metric follows the HLS schedule exactly.
        raw_t = np.asarray(t, dtype=float).copy()
        raw_z0 = np.asarray(z0, dtype=float).copy()
        raw_z1 = np.asarray(z1, dtype=float).copy()
        if np.any(apex):
            outputs[apex, :] = 0.0
            raw_t[apex] = 0.0
            raw_z0[apex] = 0.0
            raw_z1[apex] = 0.0
        if np.any(radial):
            denom = self.product_q(one_q + self.product_q(mu_q * mu_q))
            numerator = np.asarray(self.product.q(t + axial_term), dtype=float)
            t_new = np.asarray(
                self.product_div_trunc(numerator[radial], denom)
                if denom
                else np.zeros(np.count_nonzero(radial)),
                dtype=float,
            )
            zn_new = np.asarray(self.product.q(mu_q * t_new), dtype=float)
            ratio = np.asarray(self.product.q(zn_new * self.product.q(inv[radial])), dtype=float)
            lateral0 = np.asarray(self.state.q(self.product.q(ratio * z0[radial])), dtype=float)
            lateral1 = np.asarray(self.state.q(self.product.q(ratio * z1[radial])), dtype=float)
            outputs[radial, 0] = t_new
            outputs[radial, 1] = lateral0
            outputs[radial, 2] = lateral1
            raw_t[radial] = t_new
            raw_z0[radial] = lateral0
            raw_z1[radial] = lateral1
        # Commit the candidate directly to fixed state registers.  The values
        # outside the candidate's three coordinates are untouched by design;
        # no float32 egress is present inside the projection loop.
        outputs = np.asarray(self.state_commit(outputs), dtype=float)
        raw_local = np.column_stack((raw_t, raw_z0, raw_z1))
        delta_wide = np.asarray(self.norm_product.q(raw_local - local_before), dtype=float)
        delta = np.asarray(self.product.q(delta_wide), dtype=float)
        sq = np.asarray(self.norm_product.q(delta * delta), dtype=float)
        scores = np.asarray(self.norm_sq.q(sq[:, 0]), dtype=float)
        scores = np.asarray(self.norm_sq.q(scores + sq[:, 1]), dtype=float)
        scores = np.asarray(self.norm_sq.q(scores + sq[:, 2]), dtype=float)
        self.stats.projector_evaluations += qn
        self.stats.branch_counts["zero"] += int(np.count_nonzero(guarded))
        self.stats.branch_counts["inside"] += int(np.count_nonzero(~violation & ~guarded))
        self.stats.branch_counts["apex"] += int(np.count_nonzero(apex & guarded))
        self.stats.branch_counts["axial"] += int(np.count_nonzero(apex & ~guarded))
        self.stats.branch_counts["radial"] += int(np.count_nonzero(radial))
        self.stats.branch_counts["boundary"] += int(np.count_nonzero(lhs == rhs))
        branches = [
            "zero"
            if guarded[i]
            else "inside"
            if not violation[i]
            else "apex"
            if apex[i] and guarded[i]
            else "axial"
            if apex[i]
            else "radial"
            for i in range(qn)
        ]
        return [outputs[i] for i in range(qn)], scores, branches

    def _histogram_batch(self, values: np.ndarray) -> None:
        for value in np.asarray(values, dtype=float).reshape(-1):
            self._histogram(float(value))

    def rsqrt_batch(self, values: np.ndarray) -> np.ndarray:
        """Vector form of the fixed reciprocal-root core for disjoint cones."""
        values = np.asarray(self.norm_sq.q(values), dtype=float).reshape(-1)
        count = int(values.size)
        self.stats.rsqrt_calls += count
        out = np.zeros(count, dtype=float)
        positive = values > 0.0
        self.stats.rsqrt_guarded_calls += int(count - np.count_nonzero(positive))
        if not np.any(positive):
            return out
        if self.rsqrt_model == "exact":
            raw = 1.0 / np.sqrt(values[positive])
            out[positive] = np.asarray(self.rsqrt_out.q(raw), dtype=float)
            exact = 1.0 / np.sqrt(values[positive])
            self.stats.rsqrt_relative_error_max = max(
                float(self.stats.rsqrt_relative_error_max),
                float(np.max(np.abs(out[positive] - exact) / np.abs(exact), initial=0.0)),
            )
            self.stats.rsqrt_ulp_error_max = max(
                float(self.stats.rsqrt_ulp_error_max),
                float(np.max(np.abs(out[positive] - exact) / self.rsqrt_out.quantum, initial=0.0)),
            )
            return out
        if self.rsqrt_model == "bounded_ulp":
            signs = np.where((np.arange(count)[positive] + self.stats.rsqrt_calls) & 1, -1.0, 1.0)
            raw = 1.0 / np.sqrt(values[positive]) + signs * self.rsqrt_out.quantum
            out[positive] = np.asarray(self.rsqrt_out.q(raw), dtype=float)
            exact = 1.0 / np.sqrt(values[positive])
            self.stats.rsqrt_relative_error_max = max(
                float(self.stats.rsqrt_relative_error_max),
                float(np.max(np.abs(out[positive] - exact) / np.abs(exact), initial=0.0)),
            )
            self.stats.rsqrt_ulp_error_max = max(
                float(self.stats.rsqrt_ulp_error_max),
                float(np.max(np.abs(out[positive] - exact) / self.rsqrt_out.quantum, initial=0.0)),
            )
            return out
        # The 64-bit unsigned norm register can reach raw values above the
        # signed-int64 midpoint when a legal state is near its ±128 endpoint.
        # Keep raw normalization in Python integers rather than silently
        # wrapping through NumPy's signed dtype.
        raw_n2 = np.rint(values[positive] * (2**NORM_FRACTIONAL_BITS)).astype(object)
        exponents = np.asarray(
            [int(v).bit_length() - 1 - NORM_FRACTIONAL_BITS for v in raw_n2], dtype=int
        )
        normalized_raw = [
            (int(v) >> int(e)) if int(e) >= 0 else (int(v) << int(-e))
            for v, e in zip(raw_n2, exponents)
        ]
        for index, value in enumerate(normalized_raw):
            if value >= 2 * (2**NORM_FRACTIONAL_BITS):
                normalized_raw[index] = value >> 1
                exponents[index] += 1
        normalized_raw = np.asarray(normalized_raw, dtype=object)
        normalized = normalized_raw.astype(float) / (2**NORM_FRACTIONAL_BITS)
        mantissa = np.asarray(self.rsqrt_work.q(normalized), dtype=float)
        endpoint = mantissa >= 2.0
        mantissa[endpoint] = self.rsqrt_work.q(mantissa[endpoint] / 2.0)
        exponents[endpoint] += 1
        mantissa_raw = np.rint(mantissa * (2**RSQRT_WORK_FRACTIONAL_BITS)).astype(np.int64)
        indices = (
            (mantissa_raw >> (RSQRT_WORK_FRACTIONAL_BITS - RSQRT_LUT_BITS))
            & ((1 << RSQRT_LUT_BITS) - 1)
        ).astype(int)
        y = np.asarray(self.rsqrt_work.q(np.asarray(RSQRT_LUT)[indices]), dtype=float)
        half = float(self.rsqrt_signed.q(0.5))
        one_point_five = float(self.rsqrt_signed.q(1.5))
        for _ in range(RSQRT_NEWTON_STEPS):
            y_sq = np.asarray(self.rsqrt_product.q(y * y), dtype=float)
            xy_sq = np.asarray(self.rsqrt_product.q(mantissa * y_sq), dtype=float)
            half_xy_sq = np.asarray(self.rsqrt_signed.q(xy_sq * half), dtype=float)
            correction = np.asarray(self.rsqrt_signed.q(one_point_five - half_xy_sq), dtype=float)
            next_y = np.asarray(self.rsqrt_signed.q(y * correction), dtype=float)
            y = np.asarray(self.rsqrt_work.q(next_y), dtype=float)
        half_exponents = np.asarray([int(e) // 2 for e in exponents], dtype=int)
        odd = (exponents - 2 * half_exponents) != 0
        scaled = np.asarray(self.rsqrt_scale.q(y), dtype=float)
        if np.any(odd):
            inv_sqrt_two = float(self.rsqrt_scale.q(1.0 / math.sqrt(2.0)))
            scaled[odd] = np.asarray(self.rsqrt_scale.q(scaled[odd] * inv_sqrt_two), dtype=float)
        scale_raw = np.rint(scaled * (2**self.rsqrt_scale.fractional_bits)).astype(np.int64)
        shifted_raw = np.asarray(
            [
                (int(v) >> int(e)) if int(e) >= 0 else (int(v) << int(-e))
                for v, e in zip(scale_raw, half_exponents)
            ],
            dtype=np.int64,
        )
        scaled = np.asarray(
            self.rsqrt_scale.q(shifted_raw.astype(float) / (2**self.rsqrt_scale.fractional_bits)),
            dtype=float,
        )
        result = np.asarray(self.rsqrt_out.q(scaled), dtype=float)
        out[positive] = result
        exact = 1.0 / np.sqrt(values[positive])
        self.stats.rsqrt_relative_error_max = max(
            float(self.stats.rsqrt_relative_error_max),
            float(np.max(np.abs(result - exact) / np.abs(exact), initial=0.0)),
        )
        self.stats.rsqrt_ulp_error_max = max(
            float(self.stats.rsqrt_ulp_error_max),
            float(np.max(np.abs(result - exact) / self.rsqrt_out.quantum, initial=0.0)),
        )
        return out

    def _displacement_sq(
        self, before: np.ndarray, after: np.ndarray, coords: Iterable[int]
    ) -> float:
        # The HLS scheduler compares squared fixed displacements.  Monotonicity
        # preserves the Family-12 Euclidean winner while avoiding a second
        # reciprocal root in the scan.
        acc = 0.0
        for idx in coords:
            # The kernel rounds the subtraction into norm_product_t before
            # squaring.  A direct Python subtraction would retain extra
            # precision and could change a winner at a tie boundary.
            delta_wide = float(
                self.norm_product.q(float(after[int(idx)]) - float(before[int(idx)]))
            )
            delta = float(self.product.q(delta_wide))
            acc = float(self.norm_sq.q(acc + float(self.norm_product.q(delta * delta))))
        return acc

    def snapshot(self) -> dict:
        def fmt(f: FixedFormat) -> dict:
            return {
                "name": f.name,
                "width": f.width,
                "integer_bits": f.integer_bits,
                "fractional_bits": f.fractional_bits,
                "signed": f.signed,
                "quantum": f.quantum,
                "minimum": f.minimum,
                "maximum": f.maximum,
                "rounded_values": f.stats.rounded,
                "saturation_count": f.stats.saturations,
                "nonfinite_count": f.stats.nonfinite,
            }

        return {
            "state": fmt(self.state),
            "norm_square": fmt(self.norm_sq),
            "norm_product": fmt(self.norm_product),
            "reciprocal_root": fmt(self.rsqrt_out),
            "product": fmt(self.product),
            "gradient_matvec": fmt(self.gradient),
            "constants": fmt(self.constants),
            "k0": fmt(self.k0),
            "rsqrt_work": fmt(self.rsqrt_work),
            "rsqrt_product": fmt(self.rsqrt_product),
            "rsqrt_signed": fmt(self.rsqrt_signed),
            "rsqrt_scale": fmt(self.rsqrt_scale),
            "rounding": "round-to-nearest-even",
            "overflow": "saturate",
            "guard_rule": "norm_sq <= threshold",
            "guard_threshold": self.threshold,
            "guard_threshold_source": "state LSB squared rounded in norm-square format",
            "guard_hits": self.stats.guard_hits,
            "norm_sq_max": self.stats.norm_sq_max,
            "rsqrt_count": self.stats.rsqrt_calls,
            "rsqrt_guarded_calls": self.stats.rsqrt_guarded_calls,
            "rsqrt_model": self.rsqrt_model,
            "rsqrt_implementation": "32-entry LUT seed + 2 fixed-point Newton updates",
            "rsqrt_error_bound_ulps": (0.0 if self.rsqrt_model == "exact" else 1.0),
            "rsqrt_error_bound_absolute": (
                0.0 if self.rsqrt_model == "exact" else self.rsqrt_out.quantum
            ),
            "rsqrt_relative_error_max": self.stats.rsqrt_relative_error_max,
            "rsqrt_ulp_error_max": self.stats.rsqrt_ulp_error_max,
            "projector_callback_calls": self.stats.projector_evaluations,
            "branch_counts": dict(self.stats.branch_counts),
            "norm_sq_threshold_histogram": dict(self.stats.guard_histogram),
            "saturation_count_all_paths": sum(
                f.stats.saturations
                for f in (
                    self.state,
                    self.norm_sq,
                    self.norm_product,
                    self.rsqrt_out,
                    self.product,
                    self.gradient,
                    self.constants,
                    self.rsqrt_work,
                    self.rsqrt_product,
                    self.rsqrt_signed,
                    self.rsqrt_scale,
                )
            ),
            "nonfinite_count_all_paths": sum(
                f.stats.nonfinite
                for f in (
                    self.state,
                    self.norm_sq,
                    self.norm_product,
                    self.rsqrt_out,
                    self.product,
                    self.gradient,
                    self.constants,
                    self.rsqrt_work,
                    self.rsqrt_product,
                    self.rsqrt_signed,
                    self.rsqrt_scale,
                )
            ),
        }


@dataclass
class SolveTrace:
    x: np.ndarray
    events: list[tuple[int, str, int]]
    status: int
    iterations_executed: int
    cap_rechecks: int
    fnv_digest: str
    event_stream_digest: str
    datapath: dict


def _digest_events(events: Sequence[tuple[int, str, int]]) -> str:
    h = hashlib.sha256()
    for event in events:
        h.update((json.dumps(tuple(event), separators=(",", ":")) + "\n").encode())
    return h.hexdigest()


def _fnv_word(h: int, word: int) -> int:
    value = (h ^ ((int(word) + 1) & UINT64_MASK)) & UINT64_MASK
    return (value * FNV_PRIME) & UINT64_MASK


def _fnv_digest(events: Sequence[tuple[int, str, int]], n: int, m: int = 0) -> str:
    h = FNV_OFFSET
    previous_outer = None
    ordinal = 0
    for outer, _kind, index in events:
        # The compact event tuple omits ordinal because it is recoverable from
        # the ordered stream.  Reconstruct the per-sweep ordinal used by the
        # HLS telemetry digest (reset to zero at every outer step).
        if previous_outer != int(outer):
            ordinal = 0
            previous_outer = int(outer)
        # Native candidates follow the reserved row/lower/upper slots.
        candidate = m + 2 * int(n) + int(index)
        for word in (outer, ordinal, candidate):
            h = _fnv_word(h, word)
        ordinal += 1
    return f"{h:016x}"


def friction_objective(seed: int, contacts: int):
    """Byte-compatible with the Family-09/12 friction fixture generator."""
    rng = np.random.default_rng(int(seed))
    blocks = []
    desired = []
    for _ in range(int(contacts)):
        R = rng.standard_normal((3, 3))
        blocks.append(R.T @ R + 0.5 * np.eye(3))
        d = rng.standard_normal(3)
        d[0] = abs(d[0]) + 1.0
        desired.append(d)
    A = np.zeros((3 * int(contacts), 3 * int(contacts)), dtype=float)
    for i, block in enumerate(blocks):
        A[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = block
    target = np.concatenate(desired)
    return A, -A @ target, target


def k0_for(A: np.ndarray) -> float:
    eig = np.linalg.eigvalsh(np.asarray(A, dtype=float))
    return float(K0_SCALE / max(float(np.max(eig)), 1e-12))


def solve_native_reset(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    mu: float,
    contacts: int,
    k0: float | None = None,
    n_iters: int = DEFAULT_HORIZON,
    projection_cap: int = DEFAULT_PROJECTION_CAP,
    constraint_tol: float = DEFAULT_TOL,
    rsqrt_model: str = "fixed_nr",
) -> SolveTrace:
    """Run the direct Family-13 schedule on a block friction cone."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(b.size)
    qn = int(contacts)
    if A.shape != (n, n) or n != 3 * qn:
        raise ValueError("friction dimensions must be A=(3q,3q), b=(3q,)")
    if not (1 <= n <= MAXN and 1 <= qn <= MAXN // 3):
        raise ValueError("dimensions exceed the HLS envelope")
    if projection_cap <= 0 or n_iters <= 0:
        raise ValueError("horizon and projection cap must be positive")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be finite and positive")
    k0_value = k0_for(A) if k0 is None else float(k0)
    dp = F24Datapath(rsqrt_model)

    # Constants are loaded directly into their fixed constant registers.  The
    # state input follows the explicit float32 m_axi ingress boundary.
    Aq = dp.constants.q(A)
    bq = dp.constants.q(b)
    x = np.asarray(dp.state_in(np.asarray(x0, dtype=float)), dtype=float)
    mu_q = dp.scalar_q(float(mu))
    # k0 is a fixed F32 scalar in the revised kernel.  The product path rounds
    # the scalar-times-gradient increment before the state-register write.
    k0_q = float(dp.k0.q(float(k0_value)))
    ctol_q = dp.state.q(float(constraint_tol))
    ctol_sq = dp.norm_square((ctol_q, 0.0))
    events: list[tuple[int, str, int]] = []
    status = 0
    cap_rechecks = 0
    executed = 0

    for outer in range(int(n_iters)):
        # Row-interleave schedule: RG rows, UF columns per pipelined group.
        # Each product and each gradient accumulator write is explicitly
        # rounded.  The scalar loop order is the serial order represented by
        # the II=1 HLS lanes.
        # Vectorize across rows while retaining the exact scalar schedule:
        # each column is one row-interleave pipeline tick, and ``gradient.q``
        # is still applied after every MAC addition.  This is algebraically
        # identical to the RG=2/UF=2 C++ loop, but keeps the full q=16 battery
        # practical on a workstation.
        gradient_rows = np.zeros(n, dtype=float)
        accumulator = np.zeros(n, dtype=float)
        for j in range(n):
            terms = dp.product.q(Aq[:, j] * float(x[j]))
            accumulator = dp.gradient.q(accumulator + terms)
        gradient_rows = np.asarray(dp.gradient.q(accumulator + bq), dtype=float)

        # The Euler register write is product-rounded, then committed directly
        # to F24 state.  The m_axi float32 conversion belongs only to the
        # initial ``state_in`` boundary (and the optional final ``state_out``),
        # never to this inner-step update.
        scaled = np.asarray(dp.product.q(k0_q * gradient_rows), dtype=float)
        updated_fixed = np.asarray(dp.product.q(x - scaled), dtype=float)
        x_gradient = np.asarray(dp.state_commit(updated_fixed), dtype=float)

        # Sequential native-reset scheduler.  Candidate order is cone 0..q-1
        # and the winner's exact reset is committed before the next scan.
        x = x_gradient
        left_sweep = False
        for _ordinal in range(int(projection_cap)):
            best_sq = 0.0
            winner = -1
            winner_state = None
            # The scheduler reads the resident fixed state at each scan.  This
            # is a register boundary in the callback-free HLS schedule, not a
            # fresh m_axi ingress.
            projected_batch, score_batch, _branches = dp.scaled_soc_reset_batch(x, qn, mu_q)
            for cone in range(qn):
                projected = projected_batch[cone]
                score_sq = float(score_batch[cone])
                # The score is a nonnegative fixed displacement.  Strict >
                # gives the required first-maximal tie rule.
                if score_sq > best_sq:
                    best_sq = float(score_sq)
                    winner = cone
                    winner_state = projected
            if winner < 0 or best_sq <= ctol_sq:
                left_sweep = True
                break
            assert winner_state is not None
            winner_start = 3 * int(winner)
            x[winner_start : winner_start + 3] = np.asarray(winner_state, dtype=float)
            event = (int(outer), "set", int(winner))
            events.append(event)
            dp.stats.reset_events += 1
        if not left_sweep:
            cap_rechecks += 1
            # Fresh joint cone check after cap exhaustion, using the same
            # candidate reset distances as the scheduler.
            maximum = 0.0
            _p, scores, _b = dp.scaled_soc_reset_batch(x, qn, mu_q)
            maximum = float(np.max(scores, initial=0.0))
            if maximum > ctol_sq:
                status = 2
                executed = outer + 1
                break
        executed = outer + 1

    return SolveTrace(
        x=np.asarray(x, dtype=float),
        events=events,
        status=status,
        iterations_executed=executed,
        cap_rechecks=cap_rechecks,
        fnv_digest=_fnv_digest(events, n),
        event_stream_digest=_digest_events(events),
        datapath=dp.snapshot(),
    )


def anchor_trace(*, rsqrt_model: str = "fixed_nr", n_iters: int = DEFAULT_HORIZON) -> SolveTrace:
    A, b, _target = friction_objective(4903, 1)
    return solve_native_reset(
        A,
        b,
        np.zeros(3),
        mu=0.4,
        contacts=1,
        k0=k0_for(A),
        n_iters=n_iters,
        rsqrt_model=rsqrt_model,
    )


def _json_trace(trace: SolveTrace, include_events: bool = False) -> dict:
    out = {
        "schema": "family13-kernel-model-v2",
        "state_fractional_bits": STATE_FRACTIONAL_BITS,
        "status": int(trace.status),
        "iterations_executed": int(trace.iterations_executed),
        "events": int(len(trace.events)),
        "event_stream_digest": trace.event_stream_digest,
        "fnv_digest": trace.fnv_digest,
        "final_raw_state": np.rint(trace.x * (2**STATE_FRACTIONAL_BITS)).astype(np.int64).tolist(),
        "final_state": trace.x.tolist(),
        "cap_rechecks": int(trace.cap_rechecks),
        "datapath": trace.datapath,
    }
    if include_events:
        out["event_stream"] = [list(e) for e in trace.events]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4903)
    parser.add_argument("--contacts", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.4)
    parser.add_argument("--iterations", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--projection-cap", type=int, default=DEFAULT_PROJECTION_CAP)
    parser.add_argument(
        "--rsqrt-model", choices=("fixed_nr", "exact", "bounded_ulp"), default="fixed_nr"
    )
    parser.add_argument("--events", action="store_true", help="include the full event stream")
    args = parser.parse_args(argv)
    A, b, _target = friction_objective(args.seed, args.contacts)
    trace = solve_native_reset(
        A,
        b,
        np.zeros(A.shape[0]),
        mu=args.mu,
        contacts=args.contacts,
        k0=k0_for(A),
        n_iters=args.iterations,
        projection_cap=args.projection_cap,
        rsqrt_model=args.rsqrt_model,
    )
    print(json.dumps(_json_trace(trace, args.events), sort_keys=True, separators=(",", ":")))
    return 0 if trace.status == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())


# v07 configure-time native reset model -------------------------------------------------
@dataclass(frozen=True)
class Cone:
    kind: str
    offset: int
    length: int
    radius: float = 0.0
    mu: float = 1.0
    center: float = 0.0


def validate_cone_table(cones: Sequence[Cone], n: int) -> None:
    """Reject descriptors that cannot be represented by the resident ABI."""
    if len(cones) > 64:
        raise ValueError("cone table overflow: at most 64 entries")
    spans: list[tuple[int, int]] = []
    for c in cones:
        if c.kind not in {"ball", "soc", "scaled_soc"}:
            raise ValueError(f"unsupported cone kind: {c.kind}")
        minimum = 3 if c.kind in {"soc", "scaled_soc"} else 1
        if c.length < minimum or c.offset < 0 or c.offset + c.length > n:
            raise ValueError("cone block must be contiguous and inside state vector")
        if not np.isfinite(c.radius) or c.radius < 0 or c.radius >= 128:
            raise ValueError("ball radius must be finite, non-negative, and fit ap_fixed<32,8>")
        if not np.isfinite(c.center) or c.center < -128 or c.center >= 128:
            raise ValueError("ball center must fit ap_fixed<32,8>")
        if c.kind in {"soc", "scaled_soc"} and (
            not np.isfinite(c.mu) or c.mu < 2**-24 or c.mu >= 128
        ):
            raise ValueError("SOC slope mu must be finite, at least 2^-24, and fit ap_fixed<32,8>")
        norm_length = c.length if c.kind == "ball" else c.length - 1
        if norm_length * 128**2 >= 2**NORM_INTEGER_BITS:
            raise ValueError("cone block exceeds norm-square capacity")
        span = (c.offset, c.offset + c.length)
        if any(span[0] < b and a < span[1] for a, b in spans):
            raise ValueError("cone blocks overlap")
        spans.append(span)


def validate_contiguous_indices(indices: Sequence[int], n: int) -> None:
    vals = [int(i) for i in indices]
    if any(i < 0 or i >= n for i in vals):
        raise ValueError("index list is outside the state vector")
    if any(b != a + 1 for a, b in zip(vals, vals[1:])):
        raise ValueError("cone indices must form one contiguous block")


def _v07_project(dp, x, c):
    """The v13 F48 norm / F32 product registers, generalized to a block."""
    if c.kind != "ball":
        y, score, _ = dp.scaled_soc_reset(
            x, c.offset, range(c.offset + 1, c.offset + c.length), c.mu
        )
        return y, score
    y = x.copy()
    sl = slice(c.offset, c.offset + c.length)
    z = dp.state.q(x[sl] - float(dp.constants.q(c.center)))
    n2 = 0.0
    for value in z:
        n2 = float(dp.norm_sq.q(n2 + float(dp.norm_product.q(value * value))))
    inv = dp.rsqrt(n2) if n2 > dp.threshold else 0.0
    norm = dp.product_q(dp.product_q(n2) * dp.product_q(inv))
    radius = float(dp.constants.q(c.radius))
    if norm > radius:
        ratio = dp.product_q(radius * dp.product_q(inv))
        y[sl] = dp.state.q(dp.product.q(float(dp.constants.q(c.center)) + dp.product.q(ratio * z)))
    score = 0.0
    for delta in y[sl] - x[sl]:
        score = float(dp.norm_sq.q(score + dp.norm_product.q(delta * delta)))
    return y, score


def project_ball(x: np.ndarray, c: Cone) -> np.ndarray:
    return _v07_project(F24Datapath(), np.asarray(x, float), c)[0]


def project_scaled_soc(x: np.ndarray, c: Cone) -> np.ndarray:
    return _v07_project(F24Datapath(), np.asarray(x, float), c)[0]


def _v07_div_state(num, den):
    # dt/dt retains F24; the vendor quotient truncates toward zero.
    a = int(np.rint(num * 2**24))
    b = int(np.rint(den * 2**24))
    if not b:
        return 0.0
    raw = (abs(a) << 24) // abs(b)
    return (-raw if (a < 0) != (b < 0) else raw) / 2**24


def solve_v07(
    A: np.ndarray,
    b: np.ndarray,
    C: np.ndarray | None = None,
    d: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    *,
    cones: Sequence[Cone] = (),
    row_scale: np.ndarray | None = None,
    cns: np.ndarray | None = None,
    G: np.ndarray | None = None,
    lower: float | None = None,
    upper: float | None = None,
    k0: float = 0.03,
    ctol: float = 1e-6,
    n_iters: int = 50,
    projection_cap: int = 64,
) -> dict:
    """Independent register-level model of the shared v07 resident solve.

    v06 MAC groups, F24 Ax/residual/step registers, incremental Gram updates,
    and cap rechecks are preserved; cone commits refresh row residuals.
    Cone-only inputs are represented by one zero row, as in the native ABI.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    n = len(b)
    C = np.zeros((0, n)) if C is None else np.asarray(C, float).reshape(-1, n)
    d = np.zeros(len(C)) if d is None else np.asarray(d, float)
    if A.shape != (n, n) or len(C) != len(d):
        raise ValueError("inconsistent dimensions")
    if len(C) == 0:
        C = np.zeros((1, n))
        d = np.zeros(1)
        cns = np.ones(1)
        row_scale = np.ones(1)
        G = np.zeros((1, 1))
    validate_cone_table(cones, n)
    m = len(C)
    dp = F24Datapath()
    S = dp.state.q
    P = dp.product.q
    cns = np.sum(C * C, axis=1) if cns is None else np.asarray(cns)
    row_scale = np.ones(m) if row_scale is None else np.asarray(row_scale)
    G = C @ C.T if G is None else np.asarray(G)
    A = S(A)
    C = S(C)
    G = S(G)
    cns = S(cns)
    row_scale = S(row_scale)
    b = S(b)
    d = S(d)
    x = S(np.zeros(n) if x0 is None else np.asarray(x0, float))
    k0 = float(dp.k0.q(k0) if cones else S(k0))
    ctol = float(S(ctol))
    lower = None if lower is None else float(S(lower))
    upper = None if upper is None else float(S(upper))

    def matvec(matrix, x, bias=None):
        acc = np.zeros(matrix.shape[0])
        for j0 in range(0, len(x), 4):
            partial = np.zeros_like(acc)
            for j in range(j0, min(j0 + 4, len(x))):
                partial = P(partial + matrix[:, j] * x[j])
            acc = P(acc + partial)
        return S(acc if bias is None else P(acc + bias))

    events = []
    status = 0
    rechecks = 0
    executed = 0
    digest = FNV_OFFSET
    for outer in range(n_iters):
        if cones:
            acc = np.zeros(n)
            for j in range(n):
                acc = dp.gradient.q(acc + P(A[:, j] * x[j]))
            gradient = dp.gradient.q(acc + b)
            x = S(P(x - P(k0 * gradient)))
        else:
            ax = matvec(A, x)
            x = S(P(x - k0 * (ax + b)))
        residual = matvec(C, x, d)
        settled = False
        for ordinal in range(projection_cap):
            scores = P(residual * row_scale)
            winner = int(np.argmax(scores))
            best = float(scores[winner])
            kind = "row"
            if lower is not None:
                for i, v in enumerate(lower - x):
                    if v > best:
                        best = float(v)
                        kind = "lower"
                        winner = i
            if upper is not None:
                for i, v in enumerate(x - upper):
                    if v > best:
                        best = float(v)
                        kind = "upper"
                        winner = i
            best_sq = float(dp.norm_sq.q(max(0.0, best) ** 2))
            tol_sq = float(dp.norm_sq.q(ctol * ctol))
            selected = None
            for q, c in enumerate(cones):
                y, score = _v07_project(dp, x, c)
                if score > best_sq and score > tol_sq:
                    best_sq = score
                    kind = "cone"
                    winner = q
                    selected = y
            if kind != "cone" and best <= ctol:
                settled = True
                break
            if kind == "row":
                step = float(S(_v07_div_state(residual[winner], cns[winner])))
                x = S(P(x - step * C[winner]))
                residual = S(P(residual - step * G[winner]))
                cid = winner
            elif kind in ("lower", "upper"):
                delta = float(S(best if kind == "lower" else -best))
                x[winner] = S(x[winner] + delta)
                residual = S(P(residual + delta * C[:, winner]))
                cid = m + (0 if kind == "lower" else n) + winner
            else:
                x = selected
                residual = matvec(C, x, d)
                cid = m + 2 * n + winner
            events.append((outer, kind, int(cid)))
            for word in (outer, ordinal, cid):
                digest = _fnv_word(digest, word)
        if not settled:
            rechecks += 1
            residual = matvec(C, x, d)
            maximum = max(0.0, float(np.max(P(residual * row_scale))))
            if lower is not None:
                maximum = max(maximum, float(np.max(lower - x)))
            if upper is not None:
                maximum = max(maximum, float(np.max(x - upper)))
            if maximum > ctol or any(
                _v07_project(dp, x, c)[1] > float(dp.norm_sq.q(ctol * ctol)) for c in cones
            ):
                status = 2
                executed = outer + 1
                break
        executed = outer + 1
    return {
        "x": x,
        "raw": np.rint(x * 2**24).astype(np.int64),
        "events": events,
        "event_stream_digest": _digest_events(events),
        "native_digest": f"{digest:016x}",
        "cone_count": len(cones),
        "status": status,
        "iterations_executed": executed,
        "cap_rechecks": rechecks,
    }
