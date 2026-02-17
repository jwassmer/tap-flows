# %%
"""Scaling benchmark: SCGC pipeline vs brute-force intervention screening.

This study compares speed and complexity across two axes:
1) Candidate-budget combinatorics (vary n-candidates and budget k),
2) Network-size scaling (vary graph size with fixed n-candidates and k).

Outputs:
- detail CSV with per-instance timings/complexity/quality,
- summary CSV with grouped aggregates,
- 2-panel scaling figure.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Allow direct execution via `python analysis/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import Graphs as gr
from src import SocialCost as sc
from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc
from src.figure_style import apply_publication_style

OUTPUT_FIGURE = "figs/scgc-speed-complexity-scaling.pdf"
OUTPUT_DETAIL = "cache/scgc-speed-complexity-detail.csv"
OUTPUT_SUMMARY = "cache/scgc-speed-complexity-summary.csv"

GRAPH_FAMILY_LABELS = {
    "city_like": "City-like",
    "regular": "Regular",
    "random": "Random",
}

GRAPH_FAMILY_COLORS = {
    "city_like": "#1b9e77",
    "regular": "#4daf4a",
    "random": "#ff7f00",
}

GRAPH_FAMILY_ALIASES = {
    "city_planar": "city_like",
    "er_random": "random",
    "square_lattice": "regular",
}

REGIME_MARKERS = {
    "baseline": "o",
    "stress": "s",
}

REGIME_LABELS = {
    "baseline": "Baseline",
    "stress": "Stress demand",
}


@dataclass(frozen=True)
class Regime:
    gamma: float
    commodity_fraction: float
    min_commodities: int
    beta_cv: float


REGIMES = {
    "baseline": Regime(
        gamma=0.030,
        commodity_fraction=0.16,
        min_commodities=4,
        beta_cv=0.00,
    ),
    "stress": Regime(
        gamma=0.060,
        commodity_fraction=0.24,
        min_commodities=6,
        beta_cv=0.00,
    ),
}


def _parse_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _parse_int_list(raw: str) -> list[int]:
    out = []
    for token in _parse_list(raw):
        out.append(int(token))
    return out


def _parse_float_list(raw: str) -> list[float]:
    out = []
    for token in _parse_list(raw):
        out.append(float(token))
    return out


def _write_rows(path: str, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_graph(graph_family: str, num_nodes: int, seed: int) -> nx.DiGraph:
    graph_family = GRAPH_FAMILY_ALIASES.get(graph_family, graph_family)
    if graph_family == "city_like":
        return gr.random_planar_graph(n_nodes=num_nodes, seed=seed, alpha="random")
    if graph_family == "random":
        num_edges = max(num_nodes + 2, int(round(2.2 * num_nodes)))
        return gr.random_graph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            seed=seed,
            alpha="random",
            beta="random",
            directed=True,
        )
    if graph_family == "regular":
        side = max(4, int(round(np.sqrt(num_nodes))))
        return gr.squareLattice(
            n=side,
            alpha="random",
            beta="random",
            seed=seed,
            directed=True,
        )
    raise ValueError(f"Unknown graph family '{graph_family}'.")


def _ensure_population_attribute(graph: nx.DiGraph, seed: int) -> None:
    if all("population" in graph.nodes[node] for node in graph.nodes):
        return
    rng = np.random.default_rng(seed)
    pop = np.clip(
        250.0 * (1.0 + rng.pareto(1.3, size=graph.number_of_nodes())),
        100.0,
        6000.0,
    )
    nx.set_node_attributes(
        graph,
        {node: float(pop[idx]) for idx, node in enumerate(graph.nodes)},
        "population",
    )


def _apply_beta_heterogeneity(
    graph: nx.DiGraph, beta_cv: float, seed: int, min_beta: float
) -> None:
    if beta_cv <= 0:
        return
    edges = list(graph.edges)
    beta_base = np.array([float(graph.edges[e]["beta"]) for e in edges], dtype=float)
    rng = np.random.default_rng(seed)
    sigma = float(beta_cv)
    mu = -0.5 * sigma * sigma
    factor = rng.lognormal(mean=mu, sigma=sigma, size=beta_base.size)
    beta_new = np.maximum(min_beta, beta_base * factor)
    nx.set_edge_attributes(graph, dict(zip(edges, beta_new)), "beta")


def _build_demands(graph: nx.DiGraph, regime: Regime) -> np.ndarray:
    node_list = list(graph.nodes)
    num_nodes = len(node_list)
    num_commodities = min(
        num_nodes,
        max(regime.min_commodities, int(round(regime.commodity_fraction * num_nodes))),
    )
    populations = np.array(
        [float(graph.nodes[node]["population"]) for node in node_list],
        dtype=float,
    )
    source_positions = np.linspace(0, num_nodes - 1, num_commodities, dtype=int)
    demands = np.zeros((num_commodities, num_nodes), dtype=float)
    for commodity_idx, source_idx in enumerate(source_positions):
        load = max(regime.gamma * populations[source_idx], 1e-9)
        demand = np.full(num_nodes, -load / (num_nodes - 1), dtype=float)
        demand[source_idx] = load
        demands[commodity_idx] = demand
    return demands


def _top_flow_candidate_pool(flow: np.ndarray, max_candidates: int) -> np.ndarray:
    order = np.argsort(-np.asarray(flow, dtype=float), kind="mergesort")
    return order[: max(1, min(int(max_candidates), int(flow.size)))]


def _evaluate_subset_delta(
    graph: nx.DiGraph,
    demands: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    edge_ids: tuple[int, ...],
    rel_step: float,
    min_beta: float,
    base_sc: float,
    base_scale: float,
    solver,
) -> tuple[float, float]:
    beta_trial = beta.copy()
    if edge_ids:
        idx = np.asarray(edge_ids, dtype=int)
        beta_trial[idx] = np.maximum(min_beta, beta[idx] * (1.0 - rel_step))

    t0 = time.perf_counter()
    flow_trial = mc.solve_multicommodity_tap(
        graph,
        demands,
        alpha=alpha,
        beta=beta_trial,
        solver=solver,
        pos_flows=True,
    )
    dt = float(time.perf_counter() - t0)
    sc_trial = sc.total_social_cost(graph, flow_trial, alpha=alpha, beta=beta_trial)
    delta_pct = 100.0 * float((sc_trial - base_sc) / base_scale)
    return delta_pct, dt


def _median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    axis_a = detail_df[detail_df["axis_type"] == "axis_a"]
    axis_b = detail_df[detail_df["axis_type"] == "axis_b"]

    group_specs = [
        ("axis_a", axis_a, ["axis_type", "graph_family", "regime", "n_candidates", "budget_k"]),
        ("axis_b", axis_b, ["axis_type", "graph_family", "regime", "num_nodes"]),
    ]

    for _, frame, keys in group_specs:
        if frame.empty:
            continue
        grouped = frame.groupby(keys, as_index=False, dropna=False)
        for _, group in grouped:
            exact = group[~group["is_extrapolated"]]
            rows.append(
                {
                    **{k: group.iloc[0][k] for k in keys},
                    "n_rows": int(len(group)),
                    "fraction_extrapolated": float(group["is_extrapolated"].mean()),
                    "mean_speedup_incremental": float(group["speedup_incremental"].mean()),
                    "median_speedup_incremental": float(group["speedup_incremental"].median()),
                    "mean_speedup_end_to_end": float(group["speedup_end_to_end"].mean()),
                    "median_speedup_end_to_end": float(group["speedup_end_to_end"].median()),
                    "mean_t_scgc_total_s": float(group["t_scgc_total_s"].mean()),
                    "mean_t_brute_total_s": float(group["t_brute_total_s"].mean()),
                    "mean_t_scgc_incremental_s": float(group["t_scgc_incremental_s"].mean()),
                    "mean_t_brute_incremental_s": float(group["t_brute_incremental_s"].mean()),
                    "exact_match_share_pct": (
                        float(100.0 * exact["is_optimal_match"].mean())
                        if not exact.empty
                        else float("nan")
                    ),
                    "median_regret_sc_pct": (
                        float(exact["regret_sc_pct"].median())
                        if not exact.empty
                        else float("nan")
                    ),
                }
            )

    if not detail_df.empty:
        exact_all = detail_df[~detail_df["is_extrapolated"]]
        rows.append(
            {
                "axis_type": "ALL",
                "graph_family": "ALL",
                "regime": "ALL",
                "n_rows": int(len(detail_df)),
                "fraction_extrapolated": float(detail_df["is_extrapolated"].mean()),
                "mean_speedup_incremental": float(detail_df["speedup_incremental"].mean()),
                "median_speedup_incremental": float(detail_df["speedup_incremental"].median()),
                "mean_speedup_end_to_end": float(detail_df["speedup_end_to_end"].mean()),
                "median_speedup_end_to_end": float(detail_df["speedup_end_to_end"].median()),
                "mean_t_scgc_total_s": float(detail_df["t_scgc_total_s"].mean()),
                "mean_t_brute_total_s": float(detail_df["t_brute_total_s"].mean()),
                "mean_t_scgc_incremental_s": float(detail_df["t_scgc_incremental_s"].mean()),
                "mean_t_brute_incremental_s": float(detail_df["t_brute_incremental_s"].mean()),
                "exact_match_share_pct": (
                    float(100.0 * exact_all["is_optimal_match"].mean())
                    if not exact_all.empty
                    else float("nan")
                ),
                "median_regret_sc_pct": (
                    float(exact_all["regret_sc_pct"].median())
                    if not exact_all.empty
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_scaling(detail_df: pd.DataFrame, output_figure: str) -> None:
    if detail_df.empty:
        return

    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=12)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))

    ax_a, ax_b = axes

    # Panel A: incremental speedup vs combinatorics.
    data_a = detail_df[detail_df["axis_type"] == "axis_a"].copy()
    for family in sorted(data_a["graph_family"].dropna().unique()):
        color = GRAPH_FAMILY_COLORS.get(str(family), "#666666")
        for regime in sorted(data_a["regime"].dropna().unique()):
            marker = REGIME_MARKERS.get(str(regime), "o")
            subset = data_a[
                (data_a["graph_family"] == family) & (data_a["regime"] == regime)
            ]
            if subset.empty:
                continue

            exact = subset[~subset["is_extrapolated"]]
            extrap = subset[subset["is_extrapolated"]]

            if not exact.empty:
                ax_a.scatter(
                    exact["num_combinations"],
                    exact["speedup_incremental"],
                    s=34,
                    marker=marker,
                    c=color,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.45,
                )
            if not extrap.empty:
                extrap_sorted = extrap.sort_values("num_combinations")
                ax_a.scatter(
                    extrap_sorted["num_combinations"],
                    extrap_sorted["speedup_incremental"],
                    s=40,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidth=1.1,
                    alpha=0.95,
                )
                if len(extrap_sorted) >= 2:
                    ax_a.plot(
                        extrap_sorted["num_combinations"],
                        extrap_sorted["speedup_incremental"],
                        linestyle="--",
                        color=color,
                        linewidth=1.0,
                        alpha=0.85,
                    )

    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"Brute-force combinations $C(n,k)$")
    ax_a.set_ylabel("Incremental speedup (brute / SCGC)")
    ax_a.grid(alpha=0.28)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=GRAPH_FAMILY_COLORS[family],
            markeredgecolor=GRAPH_FAMILY_COLORS[family],
            markersize=7,
            label=GRAPH_FAMILY_LABELS[family],
        )
        for family in sorted(data_a["graph_family"].dropna().unique())
    ]
    regime_handles = [
        Line2D(
            [0],
            [0],
            marker=REGIME_MARKERS.get(regime, "o"),
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label=REGIME_LABELS.get(regime, regime),
        )
        for regime in sorted(data_a["regime"].dropna().unique())
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label="Exact brute-force",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="--",
            markerfacecolor="none",
            markeredgecolor="black",
            color="black",
            markersize=7,
            label="Extrapolated brute-force",
        ),
    ]
    legend_color = ax_a.legend(
        handles=color_handles,
        title="Family (color)",
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.95,
    )
    ax_a.add_artist(legend_color)
    legend_regime = ax_a.legend(
        handles=regime_handles,
        title="Regime (marker)",
        loc="center left",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.95,
    )
    ax_a.add_artist(legend_regime)
    ax_a.legend(
        handles=style_handles,
        title="Brute status",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.95,
    )

    # Panel B: end-to-end runtime vs network size for axis B.
    data_b = detail_df[detail_df["axis_type"] == "axis_b"].copy()
    if not data_b.empty:
        grouped = (
            data_b.groupby("num_edges", as_index=False)
            .agg(
                scgc_median=("t_scgc_total_s", "median"),
                brute_median=("t_brute_total_s", "median"),
            )
            .sort_values("num_edges")
        )
        ax_b.plot(
            grouped["num_edges"],
            grouped["scgc_median"],
            marker="o",
            linewidth=1.8,
            color="#1f78b4",
            label="SCGC pipeline",
        )
        ax_b.plot(
            grouped["num_edges"],
            grouped["brute_median"],
            marker="s",
            linewidth=1.8,
            color="#e31a1c",
            label="Brute force",
        )

        exact_b = data_b[~data_b["is_extrapolated"]]
        extrap_b = data_b[data_b["is_extrapolated"]]
        if not exact_b.empty:
            ax_b.scatter(
                exact_b["num_edges"],
                exact_b["t_brute_total_s"],
                s=14,
                c="#e31a1c",
                alpha=0.20,
                edgecolor="none",
            )
        if not extrap_b.empty:
            ax_b.scatter(
                extrap_b["num_edges"],
                extrap_b["t_brute_total_s"],
                s=28,
                facecolors="none",
                edgecolors="#e31a1c",
                linewidth=1.0,
                alpha=0.85,
                label="Brute (extrapolated points)",
            )

    ax_b.set_xlabel(r"Network size $|E|$")
    ax_b.set_ylabel("End-to-end runtime [s]")
    ax_b.grid(alpha=0.28)
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speed/complexity scaling benchmark for SCGC vs brute force.",
    )
    parser.add_argument("--graph-families", default="city_like,regular,random")
    parser.add_argument("--regimes", default="baseline,stress")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=11)

    parser.add_argument("--axis-a-num-nodes", type=int, default=22)
    parser.add_argument("--axis-a-candidates", default="5,7,9,11,13")
    parser.add_argument("--axis-a-budget-ratios", default="0.15,0.25,0.35")

    parser.add_argument("--axis-b-nodes", default="14,18,22,26,30,34")
    parser.add_argument("--axis-b-candidates", type=int, default=8)
    parser.add_argument("--axis-b-budget", type=int, default=2)

    parser.add_argument("--max-combinations-exact", type=int, default=2000)
    parser.add_argument("--rel-step", type=float, default=0.01)
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--optimal-tol-pct", type=float, default=1e-9)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-detail", default=OUTPUT_DETAIL)
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

    if args.smoke and args.medium:
        parser.error("--smoke and --medium are mutually exclusive.")

    if args.smoke:
        args.num_trials = 1
        args.axis_a_num_nodes = min(args.axis_a_num_nodes, 18)
        args.axis_a_candidates = "5,7"
        args.axis_a_budget_ratios = "0.2,0.35"
        args.axis_b_nodes = "14,18,22"
        args.axis_b_candidates = min(args.axis_b_candidates, 6)
        args.axis_b_budget = min(args.axis_b_budget, 2)
        args.max_combinations_exact = min(args.max_combinations_exact, 120)
    elif args.medium:
        args.num_trials = min(args.num_trials, 2)
        args.axis_a_num_nodes = min(args.axis_a_num_nodes, 22)
        args.axis_a_candidates = "5,7,9,11"
        args.axis_a_budget_ratios = "0.15,0.25,0.35"
        args.axis_b_nodes = "14,18,22,26,30"
        args.axis_b_candidates = min(args.axis_b_candidates, 8)
        args.axis_b_budget = min(args.axis_b_budget, 2)
        args.max_combinations_exact = min(args.max_combinations_exact, 700)

    if args.num_trials < 1:
        parser.error("--num-trials must be >= 1.")
    if args.seed_step < 1:
        parser.error("--seed-step must be >= 1.")
    if args.axis_a_num_nodes < 8:
        parser.error("--axis-a-num-nodes must be >= 8.")
    if args.axis_b_candidates < 2:
        parser.error("--axis-b-candidates must be >= 2.")
    if args.axis_b_budget < 1:
        parser.error("--axis-b-budget must be >= 1.")
    if args.max_combinations_exact < 1:
        parser.error("--max-combinations-exact must be >= 1.")
    if args.rel_step <= 0 or args.rel_step >= 1:
        parser.error("--rel-step must be in (0,1).")
    if args.min_beta <= 0:
        parser.error("--min-beta must be > 0.")
    if args.optimal_tol_pct < 0:
        parser.error("--optimal-tol-pct must be >= 0.")

    return args


def main() -> None:
    args = _parse_args()
    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK

    graph_families_raw = _parse_list(args.graph_families)
    graph_families = []
    for family in graph_families_raw:
        normalized = GRAPH_FAMILY_ALIASES.get(family, family)
        if normalized not in graph_families:
            graph_families.append(normalized)
    regime_names = _parse_list(args.regimes)

    for family in graph_families:
        if family not in GRAPH_FAMILY_LABELS:
            raise ValueError(f"Unknown graph family '{family}'.")
    for regime_name in regime_names:
        if regime_name not in REGIMES:
            raise ValueError(f"Unknown regime '{regime_name}'.")

    axis_a_candidates = sorted(set(_parse_int_list(args.axis_a_candidates)))
    axis_a_budget_ratios = sorted(set(_parse_float_list(args.axis_a_budget_ratios)))
    axis_b_nodes = sorted(set(_parse_int_list(args.axis_b_nodes)))
    if not axis_a_candidates:
        raise ValueError("axis A candidate list is empty.")
    if not axis_a_budget_ratios:
        raise ValueError("axis A budget ratio list is empty.")
    if not axis_b_nodes:
        raise ValueError("axis B node list is empty.")
    if any(v < 2 for v in axis_a_candidates):
        raise ValueError("All axis A candidate sizes must be >= 2.")
    if any(v <= 0 for v in axis_a_budget_ratios):
        raise ValueError("All axis A budget ratios must be > 0.")
    if any(v > 1 for v in axis_a_budget_ratios):
        raise ValueError("All axis A budget ratios must be <= 1.")
    if any(v < 8 for v in axis_b_nodes):
        raise ValueError("All axis B node values must be >= 8.")

    detail_rows: list[dict] = []
    bucket_time_per_subset: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    global_time_per_subset: list[float] = []

    total_points = 0

    def extrapolated_time_per_subset(axis_type: str, graph_family: str, regime: str) -> float:
        bucket = bucket_time_per_subset.get((axis_type, graph_family, regime), [])
        local = _median_or_nan(bucket)
        if np.isfinite(local):
            return local
        global_med = _median_or_nan(global_time_per_subset)
        return global_med

    def run_one_point(
        *,
        axis_type: str,
        graph_family: str,
        regime_name: str,
        trial_idx: int,
        seed: int,
        num_nodes: int,
        n_candidates_target: int,
        budget_k_target: int,
    ) -> None:
        nonlocal total_points

        regime = REGIMES[regime_name]
        graph = _build_graph(graph_family, num_nodes=num_nodes, seed=seed)
        _ensure_population_attribute(graph, seed=seed + 101)
        _apply_beta_heterogeneity(
            graph,
            beta_cv=regime.beta_cv,
            seed=seed + 303,
            min_beta=args.min_beta,
        )
        demands = _build_demands(graph, regime=regime)

        edge_list = list(graph.edges)
        alpha = np.array([float(graph.edges[e]["alpha"]) for e in edge_list], dtype=float)
        beta = np.array([float(graph.edges[e]["beta"]) for e in edge_list], dtype=float)

        t_base_start = time.perf_counter()
        flow_matrix, _ = mc.solve_multicommodity_tap(
            graph,
            demands,
            solver=solver,
            pos_flows=True,
            return_fw=True,
        )
        t_scgc_baseline = float(time.perf_counter() - t_base_start)
        flow = np.sum(flow_matrix, axis=0)

        t_deriv_start = time.perf_counter()
        dsc_dict = mcsc.derivative_social_cost(
            graph,
            flow_matrix,
            demands,
            eps=args.eps_active,
            demands_to_sinks=False,
        )
        t_scgc_derivative = float(time.perf_counter() - t_deriv_start)
        gradient = np.array([float(dsc_dict[e]) for e in edge_list], dtype=float)

        base_sc = sc.total_social_cost(graph, flow, alpha=alpha, beta=beta)
        base_scale = max(abs(base_sc), 1e-12)

        candidate_pool = _top_flow_candidate_pool(flow, max_candidates=n_candidates_target)
        n_candidates = int(candidate_pool.size)
        if n_candidates == 0:
            return
        budget_k = int(min(max(1, budget_k_target), n_candidates))
        num_combinations = int(math.comb(n_candidates, budget_k))

        predicted_delta = 100.0 * gradient * (-args.rel_step * beta) / base_scale
        ranked = candidate_pool[
            np.argsort(predicted_delta[candidate_pool], kind="mergesort")
        ]
        scgc_subset = tuple(sorted(int(v) for v in ranked[:budget_k]))

        scgc_delta, t_scgc_incremental = _evaluate_subset_delta(
            graph=graph,
            demands=demands,
            alpha=alpha,
            beta=beta,
            edge_ids=scgc_subset,
            rel_step=args.rel_step,
            min_beta=args.min_beta,
            base_sc=base_sc,
            base_scale=base_scale,
            solver=solver,
        )
        t_scgc_total = t_scgc_baseline + t_scgc_derivative + t_scgc_incremental

        is_extrapolated = bool(num_combinations > args.max_combinations_exact)
        calls_brute_incremental = int(num_combinations)
        brute_delta = float("nan")
        regret_sc_pct = float("nan")
        gain_ratio = float("nan")
        is_optimal = float("nan")

        if not is_extrapolated:
            t_brute_start = time.perf_counter()
            best_delta = float("inf")
            subset_times = []
            for combo in itertools.combinations(candidate_pool.tolist(), budget_k):
                subset = tuple(sorted(int(v) for v in combo))
                delta, subset_dt = _evaluate_subset_delta(
                    graph=graph,
                    demands=demands,
                    alpha=alpha,
                    beta=beta,
                    edge_ids=subset,
                    rel_step=args.rel_step,
                    min_beta=args.min_beta,
                    base_sc=base_sc,
                    base_scale=base_scale,
                    solver=solver,
                )
                subset_times.append(subset_dt)
                if delta < best_delta:
                    best_delta = float(delta)
            t_brute_incremental = float(time.perf_counter() - t_brute_start)

            time_per_subset = t_brute_incremental / num_combinations
            bucket_key = (axis_type, graph_family, regime_name)
            bucket_time_per_subset[bucket_key].append(float(time_per_subset))
            global_time_per_subset.append(float(time_per_subset))

            brute_delta = float(best_delta)
            regret_sc_pct = float(scgc_delta - brute_delta)
            brute_gain = float(-brute_delta)
            scgc_gain = float(-scgc_delta)
            gain_ratio = float(scgc_gain / brute_gain) if brute_gain > 1e-12 else float("nan")
            is_optimal = bool(abs(regret_sc_pct) <= args.optimal_tol_pct)
        else:
            tps = extrapolated_time_per_subset(axis_type, graph_family, regime_name)
            t_brute_incremental = float(tps * num_combinations) if np.isfinite(tps) else float("nan")

        t_brute_baseline = t_scgc_baseline
        t_brute_total = (
            float(t_brute_baseline + t_brute_incremental)
            if np.isfinite(t_brute_incremental)
            else float("nan")
        )

        speedup_incremental = (
            float(t_brute_incremental / t_scgc_incremental)
            if t_scgc_incremental > 0 and np.isfinite(t_brute_incremental)
            else float("nan")
        )
        speedup_end_to_end = (
            float(t_brute_total / t_scgc_total)
            if t_scgc_total > 0 and np.isfinite(t_brute_total)
            else float("nan")
        )

        detail_rows.append(
            {
                "axis_type": axis_type,
                "graph_family": graph_family,
                "regime": regime_name,
                "trial": int(trial_idx),
                "seed": int(seed),
                "num_nodes": int(graph.number_of_nodes()),
                "num_edges": int(graph.number_of_edges()),
                "num_commodities": int(demands.shape[0]),
                "n_candidates": int(n_candidates),
                "budget_k": int(budget_k),
                "num_combinations": int(num_combinations),
                "is_extrapolated": bool(is_extrapolated),
                "t_scgc_baseline_s": float(t_scgc_baseline),
                "t_scgc_derivative_s": float(t_scgc_derivative),
                "t_scgc_incremental_s": float(t_scgc_incremental),
                "t_scgc_total_s": float(t_scgc_total),
                "t_brute_baseline_s": float(t_brute_baseline),
                "t_brute_incremental_s": float(t_brute_incremental),
                "t_brute_total_s": float(t_brute_total),
                "calls_scgc_incremental": 1,
                "calls_brute_incremental": int(calls_brute_incremental),
                "speedup_incremental": float(speedup_incremental),
                "speedup_end_to_end": float(speedup_end_to_end),
                "regret_sc_pct": float(regret_sc_pct),
                "gain_ratio_scgc_over_best": float(gain_ratio),
                "is_optimal_match": is_optimal,
            }
        )
        total_points += 1

    # Axis A: combinatorial scaling
    for gf_idx, graph_family in enumerate(graph_families):
        for rg_idx, regime_name in enumerate(regime_names):
            for trial_idx in range(args.num_trials):
                seed_base = (
                    args.seed
                    + trial_idx * args.seed_step
                    + gf_idx * 1000
                    + rg_idx * 100
                )
                for n_candidates in axis_a_candidates:
                    budgets = sorted(
                        {
                            int(min(n_candidates, max(1, round(r * n_candidates))))
                            for r in axis_a_budget_ratios
                        }
                    )
                    for budget_k in budgets:
                        run_one_point(
                            axis_type="axis_a",
                            graph_family=graph_family,
                            regime_name=regime_name,
                            trial_idx=trial_idx,
                            seed=seed_base + n_candidates * 17 + budget_k * 31,
                            num_nodes=args.axis_a_num_nodes,
                            n_candidates_target=n_candidates,
                            budget_k_target=budget_k,
                        )

    # Axis B: network-size scaling
    for gf_idx, graph_family in enumerate(graph_families):
        for rg_idx, regime_name in enumerate(regime_names):
            for trial_idx in range(args.num_trials):
                seed_base = (
                    args.seed
                    + trial_idx * args.seed_step
                    + gf_idx * 1000
                    + rg_idx * 100
                    + 50_000
                )
                for num_nodes in axis_b_nodes:
                    run_one_point(
                        axis_type="axis_b",
                        graph_family=graph_family,
                        regime_name=regime_name,
                        trial_idx=trial_idx,
                        seed=seed_base + num_nodes * 13,
                        num_nodes=num_nodes,
                        n_candidates_target=args.axis_b_candidates,
                        budget_k_target=args.axis_b_budget,
                    )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        _write_rows(args.output_detail, [])
        _write_rows(args.output_summary, [])
        print("No rows generated.")
        return

    summary_df = _summarize(detail_df)
    detail_df.to_csv(args.output_detail, index=False)
    summary_df.to_csv(args.output_summary, index=False)
    _plot_scaling(detail_df, args.output_figure)

    exact = detail_df[~detail_df["is_extrapolated"]]
    print(
        f"Generated {len(detail_df)} rows ({len(exact)} exact, "
        f"{len(detail_df) - len(exact)} extrapolated) across {total_points} points."
    )
    if not exact.empty:
        print(
            "Exact-only median speedups: "
            f"incremental={exact['speedup_incremental'].median():.2f}x, "
            f"end-to-end={exact['speedup_end_to_end'].median():.2f}x"
        )
    print(
        f"Saved detail: {args.output_detail}\n"
        f"Saved summary: {args.output_summary}\n"
        f"Saved figure: {args.output_figure}"
    )


if __name__ == "__main__":
    main()
