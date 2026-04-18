"""Shared matplotlib style for all benchmark figures.

Goals: a single, consistent academic look — serif body text, muted limited
palette, clean spines, no chartjunk. Importing this module sets the global
rcParams; call :func:`apply` again after any user-side overrides.
"""

from __future__ import annotations

import matplotlib as mpl

# A muted, color-blind-friendly palette — Wong (Nature, 2011) reordered.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]


def apply() -> None:
    """Apply the shared style. Idempotent; safe to call repeatedly."""
    mpl.rcParams.update(
        {
            # Fonts: prefer serif for an academic-paper feel.
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Source Serif Pro",
                "Times New Roman",
                "STIXGeneral",
                "serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            # Spines: drop the top and right; thin the rest.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Light gridlines.
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            # Lines.
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            # Color cycle.
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            # Figures.
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # PDF-friendly: embed Type 42 (TrueType) fonts.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem: str, *, formats: tuple[str, ...] = ("pdf", "png")) -> list[str]:
    """Save a figure under ``../figures/{stem}.{ext}`` for each format.

    Returns the list of paths written, relative to repo root.
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
