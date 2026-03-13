"""Plot predicted vs observed single-edge intervention effects.

This script is a lightweight visualization helper for Appendix-D style checks.
It reads cached edge-effect tables and compares:
    predicted_delta_sc_pct  vs  observed_delta_sc_pct
for each edge sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.figure_style import apply_publication_style

DEFAULT_INPUTS = (
    "cache/cologne-stadium-traffic-edge-effects.csv,"
    "cache/potsdam-traffic-edge-effects.csv,"
    "cache/intervention-validation-scgc-edge.csv"
)
DEFAULT_LABELS = "Cologne,Potsdam,Synthetic"
OUTPUT_FIGURE = "figs/intervention-predicted-vs-observed.pdf"
OUTPUT_SUMMARY = "cache/intervention-predicted-vs-observed-summary.csv"


def _parse_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _safe_corr(rx, ry)


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(
        f"Could not find {role} column. Tried {candidates}, available={list(frame.columns)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predicted-vs-observed scatter for intervention edge effects.",
    )
    parser.add_argument(
        "--input-files",
        default=DEFAULT_INPUTS,
        help="Comma-separated edge-effect CSVs.",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Comma-separated dataset labels matching --input-files.",
    )
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-summary", default=OUTPUT_SUMMARY)

    args, unknown = parser.parse_known_args()

    remaining = []
    skip_next = False
    for token in unknown:
        if skip_next:
            skip_next = False
            continue
        if token in ("-f", "--f"):
            skip_next = True
            continue
        if token.startswith("--f="):
            continue
        remaining.append(token)
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    files = _parse_list(args.input_files)
    labels = _parse_list(args.labels)
    if len(files) != len(labels):
        parser.error("--input-files and --labels must have equal length.")
    return args


def main() -> None:
    args = _parse_args()
    input_files = _parse_list(args.input_files)
    labels = _parse_list(args.labels)

    predicted_candidates = (
        "predicted_delta_sc_pct",
        "predicted_single_delta_sc_pct",
    )
    observed_candidates = (
        "observed_delta_sc_pct",
        "observed_single_delta_sc_pct",
    )
    colors = ["#1f78b4", "#e31a1c", "#33a02c", "#ff7f00", "#6a3d9a"]

    frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for idx, (path_str, label) in enumerate(zip(input_files, labels)):
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        frame = pd.read_csv(path)
        predicted_col = _resolve_column(frame, predicted_candidates, role="predicted")
        observed_col = _resolve_column(frame, observed_candidates, role="observed")
        frame = frame.copy()
        frame["predicted_delta_sc_pct"] = frame[predicted_col].astype(float)
        frame["observed_delta_sc_pct"] = frame[observed_col].astype(float)
        frame["dataset"] = label
        frame["color"] = colors[idx % len(colors)]
        frames.append(frame)

        x = frame["observed_delta_sc_pct"].to_numpy(dtype=float)
        y = frame["predicted_delta_sc_pct"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y - x)))
        summary_rows.append(
            {
                "dataset": label,
                "n_points": int(len(frame)),
                "pearson": _safe_corr(y, x),
                "spearman": _safe_spearman(y, x),
                "mae_pct": mae,
            }
        )

    all_df = pd.concat(frames, ignore_index=True)
    x_all = all_df["observed_delta_sc_pct"].to_numpy(dtype=float)
    y_all = all_df["predicted_delta_sc_pct"].to_numpy(dtype=float)
    summary_rows.append(
        {
            "dataset": "Pooled",
            "n_points": int(len(all_df)),
            "pearson": _safe_corr(y_all, x_all),
            "spearman": _safe_spearman(y_all, x_all),
            "mae_pct": float(np.mean(np.abs(y_all - x_all))),
        }
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_sum = Path(args.output_summary)
    out_sum.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=13)
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.4))

    for label in labels:
        sub = all_df[all_df["dataset"] == label]
        color = sub["color"].iloc[0]
        x = sub["observed_delta_sc_pct"].to_numpy(dtype=float)
        y = sub["predicted_delta_sc_pct"].to_numpy(dtype=float)
        rho = _safe_spearman(y, x)
        ax.scatter(
            x,
            y,
            s=28,
            alpha=0.75,
            c=color,
            edgecolor="white",
            linewidth=0.4,
            label=f"{label} ($\\rho$={rho:.3f}, n={len(sub)})",
        )

    lo = float(min(np.min(x_all), np.min(y_all)))
    hi = float(max(np.max(x_all), np.max(y_all)))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Observed $\Delta SC_e$ [\%]")
    ax.set_ylabel(r"Predicted $\widehat{\Delta SC}_e$ [\%]")
    ax.set_title("Single-edge intervention effect: prediction vs realization")
    ax.grid(alpha=0.28)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(out_sum, index=False)

    print(f"Saved figure: {out_fig}")
    print(f"Saved summary: {out_sum}")


if __name__ == "__main__":
    main()
