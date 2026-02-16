"""Shared styling and utility helpers for publication figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil


def apply_publication_style(font_size: int = 14, use_latex: bool = True) -> None:
    """Apply consistent matplotlib defaults for publication figures."""
    if use_latex and shutil.which("pdflatex") is None:
        use_latex = False

    params = {
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "font.size": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "figure.constrained_layout.use": True,
    }

    if use_latex:
        params.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "pgf.texsystem": "pdflatex",
                "pgf.preamble": "\\n".join(
                    [
                        r"\\usepackage[utf8]{inputenc}",
                        r"\\usepackage[T1]{fontenc}",
                    ]
                ),
            }
        )

    mpl.rcParams.update(params)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = 0.02,
    y: float = 1.02,
    fontsize: int = 24,
) -> None:
    """Add a bold panel label to an axis in axes coordinates."""
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=fontsize,
    )


def make_scalar_mappable(cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> mpl.cm.ScalarMappable:
    """Create a scalar mappable with an attached empty array."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm
