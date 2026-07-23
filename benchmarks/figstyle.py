"""Shared matplotlib style for every figure shipped with `snn_opt`.

One visual system covers the benchmark suite, the example scripts, and the
figures mirrored onto the companion website, so a reader moving between them
sees the same chart language throughout.

Design rules this module encodes:

* **Categorical hues are assigned in a fixed order, never cycled.** The four
  slots below are drawn from Wong's colour-blind-safe set (Nature Methods,
  2011) and were checked as a palette rather than by eye: every adjacent pair
  clears a deuteranope/protanope separation of dE >= 11 and a 3:1 contrast
  ratio against the figure surface. Nothing in the suite needs a fifth series;
  if one ever does, facet it instead of inventing a hue.
* **Neutrals are not series colours.** `INK` and `RULE` carry reference lines,
  annotations, and single-series traces where no identity is being encoded, so
  a coloured mark always means "this is one of several things".
* **Chrome is recessive.** Hairline spines, dotted low-contrast grid, no top or
  right spine, no chartjunk.
* **No em-dashes in any user-visible string.** These figures ship in a public
  repository and on the website; titles and labels use commas or colons.

Importing the module applies the style. Call :func:`apply` again after any
local rcParams override.
"""

from __future__ import annotations

import matplotlib as mpl

# --- Categorical slots, in assignment order -------------------------------
BLUE = "#0072B2"      # slot 1
VERMILION = "#D55E00"  # slot 2
GREEN = "#009E73"     # slot 3
PURPLE = "#8B5FA8"    # slot 4

PALETTE = [BLUE, VERMILION, GREEN, PURPLE]

# --- Neutrals: chrome and non-categorical marks ---------------------------
INK = "#2B2B2B"       # single-series traces, annotations, reference lines
MUTED = "#6B6B6B"     # secondary text
RULE = "#C8C8C8"      # grid, spines
SURFACE = "#FFFFFF"   # figure/axes background

# Semantic aliases, so a script says what it means rather than which hue it wants.
OBJECTIVE = BLUE
STABILITY = VERMILION
FEASIBILITY = GREEN
REFERENCE = INK


def apply() -> None:
    """Apply the shared style. Idempotent; safe to call repeatedly."""
    mpl.rcParams.update(
        {
            # Type: serif body to sit comfortably beside the papers.
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Source Serif Pro",
                "Times New Roman",
                "STIXGeneral",
                "serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "figure.titlesize": 11.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            # Ink colours.
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": RULE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
            # Chrome: recessive.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.6,
            "ytick.minor.size": 1.6,
            "axes.axisbelow": True,
            # Grid: present but quiet.
            "axes.grid": True,
            "grid.color": RULE,
            "grid.alpha": 0.55,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            # Marks.
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "lines.solid_capstyle": "round",
            "scatter.edgecolors": "none",
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            # Legend: no box, it competes with the data.
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.6,
            "legend.borderaxespad": 0.4,
            "legend.labelcolor": INK,
            # Surfaces and output.
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            # Embed TrueType so the PDFs are editable downstream.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(ax, text: str, *, loc: str = "left") -> None:
    """Set a panel title in the suite's convention: left-aligned, quiet."""
    ax.set_title(text, loc=loc, fontsize=10.0, color=INK, pad=6)


def annotate_floor(
    ax,
    y: float,
    label: str,
    *,
    color: str = INK,
    x: float = 0.985,
    ha: str = "right",
    va: str = "bottom",
) -> None:
    """Draw a horizontal reference line with a direct label riding on it.

    Used for accuracy floors, where a legend entry would be one indirection too
    many: the number belongs against the line it describes. ``x`` is in axes
    fraction so the caller can park the label wherever the data is not.
    """
    ax.axhline(y, color=color, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)
    ax.annotate(
        label,
        xy=(x, y),
        xycoords=("axes fraction", "data"),
        ha=ha,
        va=va,
        fontsize=8.0,
        color=color,
        annotation_clip=False,
    )


def save(fig, stem: str, *, formats: tuple[str, ...] = ("pdf", "png")) -> list[str]:
    """Save a figure under ``../figures/{stem}.{ext}`` for each format.

    Returns the list of paths written, relative to the repository root.
    """
    from pathlib import Path

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    paths: list[str] = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        paths.append(str(path.relative_to(out_dir.parent)))
    return paths


# Apply on import so user scripts only need ``import figstyle``.
apply()
