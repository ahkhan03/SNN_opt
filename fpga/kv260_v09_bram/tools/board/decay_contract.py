"""Fixed-point contract for the v09b decay staircase.

The three integer keys live in the package environment record. Quantum and
the minimum floor are derived from them. The chain appends the current
quanta while they are strictly above 1.5 times the floor, then appends the
floor. That is the staircase the solver and the native oracle build.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV = Path(__file__).resolve().parents[2] / "env.yaml"
_REQUIRED = ("word_bits", "int_bits", "min_floor_quanta")
_KEY_RE = re.compile(r"^(\w+):\s*(\d+)\s*$", re.MULTILINE)
_POSITIVE_LIMIT = 0x7FFFFFFF


@dataclass(frozen=True)
class FixedPointContract:
    word_bits: int
    int_bits: int
    min_floor_quanta: int

    @property
    def fraction_bits(self) -> int:
        return self.word_bits - self.int_bits

    @property
    def quantum(self) -> float:
        return 1.0 / float(1 << self.fraction_bits)

    @property
    def min_floor(self) -> float:
        return self.dequantize(self.min_floor_quanta)

    def quantize(self, value: float) -> int:
        """Round half to even, then clamp to [0, 0x7fffffff]."""
        scaled = float(value) * float(1 << self.fraction_bits)
        if not _finite(scaled):
            return _POSITIVE_LIMIT if scaled > 0.0 else 0
        if scaled >= _POSITIVE_LIMIT:
            return _POSITIVE_LIMIT
        rounded = int(round(scaled))
        if rounded >= _POSITIVE_LIMIT:
            return _POSITIVE_LIMIT
        if rounded <= 0:
            return 0
        return rounded

    def dequantize(self, quanta: int) -> float:
        return float(quanta) / float(1 << self.fraction_bits)

    def chain(self, start_q: int, floor_q: int) -> tuple[int, ...]:
        """Append-current staircase. The final entry is ``floor_q`` exactly.

        Append the current quanta, then halve, while the current value is
        strictly above 1.5 times the floor. A start already inside that
        band contributes no level of its own here; the kernel still executes
        it, so that band is refused instead of being scheduled.
        """
        level = int(start_q)
        floor = int(floor_q)
        levels = []
        while 2 * level > 3 * floor:
            levels.append(level)
            level //= 2
        levels.append(floor)
        return tuple(levels)

    @classmethod
    def load(cls, path: Path = DEFAULT_ENV) -> FixedPointContract:
        path = Path(path)
        try:
            text = path.read_text()
        except OSError as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        found: dict[str, int] = {}
        for match in _KEY_RE.finditer(text):
            found[match.group(1)] = int(match.group(2))
        missing = [key for key in _REQUIRED if key not in found]
        if missing:
            raise ValueError(f"missing {', '.join(missing)} in {path}")
        return cls(
            word_bits=found["word_bits"],
            int_bits=found["int_bits"],
            min_floor_quanta=found["min_floor_quanta"],
        )


def load_contract(path: Path = DEFAULT_ENV) -> FixedPointContract:
    return FixedPointContract.load(path)


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def in_near_floor_band(start_q: int, floor_q: int) -> bool:
    """True when ``2*start_q <= 3*floor_q``.

    ``level_count`` then returns one level, while the kernel still executes
    the start and then the floor.
    """
    return 2 * int(start_q) <= 3 * int(floor_q)


def level_count(start_q: int, floor_q: int) -> int:
    """How many levels ``level_count`` in the decay header reports."""
    levels = 1
    level = int(start_q)
    floor = int(floor_q)
    while 2 * level > 3 * floor:
        level //= 2
        levels += 1
    return levels


def next_level(level: int, floor_q: int) -> int:
    """One decay step. A halving inside (floor, 1.5*floor] lands on the floor."""
    current = int(level)
    floor = int(floor_q)
    if current <= floor:
        return current
    halved = current // 2
    if 2 * halved <= 3 * floor:
        return floor
    return halved


def final_level(start_q: int, floor_q: int, segment: int, iterations: int) -> int:
    """Quanta of the last outer iteration, using the header's segment rule."""
    level = int(start_q)
    floor = int(floor_q)
    width = int(segment)
    for outer in range(max(int(iterations) - 1, 0)):
        if (outer + 1) % width == 0:
            level = next_level(level, floor)
    return level
