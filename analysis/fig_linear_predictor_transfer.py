# %%
"""Benchmark raw linear breakpoint prediction across graph families and regimes.

This study intentionally uses the raw first-order predictor only (no v3 refinement)
to test how prediction power changes with:
1) initial conditions (demand/beta regimes), and
2) network structure (graph family).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

# Allow direct execution via `python analysis/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import Graphs as gr
from src.figure_style import apply_publication_style

_BREAKPOINT_CORE_PATH = (
    REPO_ROOT / "analysis" / "fig_active_set_breakpoint_prediction.py"
)
_CORE_SPEC = importlib.util.spec_from_file_location(
    "breakpoint_core",
    _BREAKPOINT_CORE_PATH,
)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise RuntimeError(
        f"Could not load breakpoint core module at {_BREAKPOINT_CORE_PATH}"
    )
bp = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_SPEC.name] = bp
_CORE_SPEC.loader.exec_module(bp)

OUTPUT_FIGURE = "figs/linear-predictor-transfer.pdf"
OUTPUT_TABLE = "cache/linear-predictor-transfer.csv"
OUTPUT_SUMMARY = "cache/linear-predictor-transfer-summary.csv"


class BaselineLike(Protocol):
    beta: np.ndarray


@dataclass(frozen=True)
class Regime:
    gamma: float
    commodity_fraction: float
    min_commodities: int
    beta_cv: float
    label: str


REGIMES = {
    "light": Regime(
        gamma=0.015,
        commodity_fraction=0.10,
        min_commodities=2,
        beta_cv=0.00,
        label="Light demand",
    ),
    "baseline": Regime(
        gamma=0.030,
        commodity_fraction=0.16,
        min_commodities=4,
        beta_cv=0.00,
        label="Baseline",
    ),
    "stress": Regime(
        gamma=0.060,
        commodity_fraction=0.24,
        min_commodities=6,
        beta_cv=0.00,
        label="Stress demand",
    ),
    "hetero": Regime(
        gamma=0.030,
        commodity_fraction=0.16,
        min_commodities=4,
        beta_cv=0.35,
        label="Heterogeneous beta",
    ),
}

GRAPH_FAMILY_LABELS = {
    "city_like": "City-like (planar)",
    "planar": "Planar",
    "regular": "Regular",
    "random": "Random",
    "small_world": "Small-world",
}

GRAPH_FAMILY_ALIASES = {
    "city_planar": "city_like",
    "er_random": "random",
    "square_lattice": "regular",
    "triangular_lattice": "planar",
}

GRAPH_FAMILY_COLORS = {
    "city_like": "#1b9e77",
    "planar": "#377eb8",
    "regular": "#4daf4a",
    "random": "#ff7f00",
    "small_world": "#984ea3",
}

REGIME_MARKERS = {
    "light": "o",
    "baseline": "s",
    "stress": "^",
    "hetero": "D",
}


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ensure_population_attribute(graph: nx.DiGraph, seed: int) -> None:
    if all("population" in graph.nodes[node] for node in graph.nodes):
        return
    rng = np.random.default_rng(seed)
    # Heavy-tailed but bounded to avoid extreme source loads.
    pop = np.clip(
        250.0 * (1.0 + rng.pareto(1.3, size=graph.number_of_nodes())), 100.0, 6000.0
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
    # Mean-one lognormal multiplicative perturbation.
    sigma = float(beta_cv)
    mu = -0.5 * sigma * sigma
    factor = rng.lognormal(mean=mu, sigma=sigma, size=beta_base.size)
    beta_new = np.maximum(min_beta, beta_base * factor)
    nx.set_edge_attributes(graph, dict(zip(edges, beta_new)), "beta")


def _build_graph(graph_family: str, num_nodes: int, seed: int) -> nx.DiGraph:
    graph_family = GRAPH_FAMILY_ALIASES.get(graph_family, graph_family)

    if graph_family == "city_like":
        return gr.random_planar_graph(n_nodes=num_nodes, seed=seed, alpha="random")

    if graph_family == "planar":
        # Generic planar-ish triangulation with randomized edge travel parameters.
        graph = gr.random_planar_graph(n_nodes=num_nodes, seed=seed, alpha="random")
        edges = list(graph.edges)
        rng = np.random.default_rng(seed + 17)
        beta = np.maximum(1e-6, rng.uniform(20.0, 120.0, size=len(edges)))
        nx.set_edge_attributes(graph, dict(zip(edges, beta)), "beta")
        return graph

    if graph_family == "random":
        # Sparse but connected directed ER-like graph.
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
        # Approximately regular ring-lattice style baseline.
        side = max(4, int(round(np.sqrt(num_nodes))))
        graph = gr.squareLattice(
            n=side,
            alpha="random",
            beta="random",
            seed=seed,
            directed=True,
        )
        return graph

    if graph_family == "small_world":
        n = max(10, num_nodes)
        k = min(max(4, int(round(np.sqrt(n)))), n - 1)
        if k % 2 == 1:
            k += 1
        p = 0.12
        base = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
        graph = base.to_directed()
        edges = list(graph.edges)
        rng = np.random.default_rng(seed + 29)
        alpha = rng.uniform(0.1, 1.0, size=len(edges))
        beta = np.maximum(1e-6, rng.uniform(15.0, 140.0, size=len(edges)))
        nx.set_edge_attributes(graph, dict(zip(edges, alpha)), "alpha")
        nx.set_edge_attributes(graph, dict(zip(edges, beta)), "beta")
        pos = nx.spring_layout(base, seed=seed)
        nx.set_node_attributes(graph, pos, "pos")
        return graph

    raise ValueError(f"Unknown graph family '{graph_family}'.")


def _build_demands(graph: nx.DiGraph, regime: Regime) -> np.ndarray:
    node_list = list(graph.nodes)
    num_nodes = len(node_list)
    num_commodities = min(
        num_nodes,
        max(regime.min_commodities, int(round(regime.commodity_fraction * num_nodes))),
    )
    populations = np.array(
        [float(graph.nodes[node]["population"]) for node in node_list], dtype=float
    )

    source_positions = np.linspace(0, num_nodes - 1, num_commodities, dtype=int)
    demands = np.zeros((num_commodities, num_nodes), dtype=float)
    for commodity_idx, source_idx in enumerate(source_positions):
        load = max(regime.gamma * populations[source_idx], 1e-9)
        demand = np.full(num_nodes, -load / (num_nodes - 1), dtype=float)
        demand[source_idx] = load
        demands[commodity_idx] = demand
    return demands


def _parse_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _observed_direction_breakpoint(
    baseline: BaselineLike,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    direction: int,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
    bisect_iters: int,
    initial_step_rel: float,
    expand_factor: float,
    max_expand_iters: int,
    max_rel_plus: float,
    seed_rel: float | None = None,
) -> float | None:
    """Return observed directional breakpoint radius (relative), or None if not found."""
    beta_k = float(baseline.beta[edge_idx])
    if beta_k <= min_beta:
        return None

    if direction < 0:
        max_rel = max((beta_k - min_beta) / beta_k, 0.0)
    else:
        max_rel = max_rel_plus
    if max_rel <= 1e-12:
        return None

    probe_rel = float(seed_rel) if seed_rel is not None and np.isfinite(seed_rel) and seed_rel > 0 else float(initial_step_rel)
    probe_rel = min(max(probe_rel, 1e-6), max_rel)

    def is_stable(rel: float) -> bool:
        rel = float(min(max(rel, 0.0), max_rel))
        return bp._is_active_set_stable(
            baseline=baseline,
            baseline_indicator=baseline_indicator,
            edge_idx=int(edge_idx),
            delta_abs=rel * beta_k,
            direction=int(direction),
            solver=solver,
            min_beta=min_beta,
            active_set_level=active_set_level,
            disagg_reg=disagg_reg,
        )

    if not is_stable(probe_rel):
        lo = 0.0
        hi = probe_rel
    else:
        lo = probe_rel
        hi = min(max_rel, max(lo * expand_factor, lo + initial_step_rel))
        found = False
        for _ in range(max_expand_iters):
            if not is_stable(hi):
                found = True
                break
            lo = hi
            if hi >= max_rel - 1e-12:
                break
            hi = min(max_rel, max(hi * expand_factor, hi + initial_step_rel))
        if not found:
            if is_stable(max_rel):
                return None
            hi = max_rel

    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


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


def _plot_directional_scatter(
    rows: list[dict],
    graph_families: list[str],
    regime_names: list[str],
    output_figure: str,
) -> None:
    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Render at manuscript single-column scale with sufficiently large typography.
    apply_publication_style(font_size=12)
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 3.55))

    x_all = np.array([float(r["obs_signed_pct"]) for r in rows], dtype=float)
    y_all = np.array([float(r["pred_signed_pct"]) for r in rows], dtype=float)
    low = float(min(np.min(x_all), np.min(y_all)))
    high = float(max(np.max(x_all), np.max(y_all)))
    if np.isclose(high, low):
        low -= 1.0
        high += 1.0
    pad = 0.05 * (high - low)
    lo = low - pad
    hi = high + pad

    for graph_family in graph_families:
        color = GRAPH_FAMILY_COLORS.get(graph_family, "#444444")
        for regime_name in regime_names:
            marker = REGIME_MARKERS.get(regime_name, "o")
            subset = [
                row
                for row in rows
                if row["graph_family"] == graph_family and row["regime"] == regime_name
            ]
            if not subset:
                continue
            x = np.array([float(r["obs_signed_pct"]) for r in subset], dtype=float)
            y = np.array([float(r["pred_signed_pct"]) for r in subset], dtype=float)
            ax.scatter(
                x,
                y,
                s=26,
                marker=marker,
                c=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.45,
            )

    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="black",
        linewidth=1.2,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Observed breakpoint [\%]")
    ax.set_ylabel(r"Predicted breakpoint [\%]")
    ax.set_title("")
    ax.grid(alpha=0.28)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=GRAPH_FAMILY_COLORS.get(gf, "#444444"),
            markeredgecolor=GRAPH_FAMILY_COLORS.get(gf, "#444444"),
            markersize=7.5,
            label=GRAPH_FAMILY_LABELS[gf],
        )
        for gf in graph_families
    ]
    regime_handles = [
        Line2D(
            [0],
            [0],
            marker=REGIME_MARKERS.get(regime, "o"),
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7.0,
            label=REGIMES[regime].label,
        )
        for regime in regime_names
    ]

    legend_family = ax.legend(
        handles=family_handles,
        title="Graph (color)",
        loc="upper left",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.95,
    )
    ax.add_artist(legend_family)
    ax.legend(
        handles=regime_handles,
        title="Regime (marker)",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.95,
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _print_hypotheses() -> None:
    print("Pre-simulation hypotheses (math/logical):")
    print(
        "1) Raw breakpoint error should grow with curvature and near-tie event spacing "
        "(winner-flip risk)."
    )
    print(
        "2) Higher demand/stress should reduce local linear validity, so mismatch should increase."
    )
    print(
        "3) City-like / planar families may show better transfer behavior than random "
        "or small-world families if alternatives are less degenerate."
    )
    print(
        "4) Strong beta heterogeneity can either help (clear winner) or hurt (localized "
        "instability); this must be resolved empirically."
    )
    print("")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer benchmark for raw linear active-set breakpoint predictor.",
    )
    parser.add_argument(
        "--graph-families",
        default="city_like,planar,regular,random,small_world",
        help="Comma-separated graph families.",
    )
    parser.add_argument(
        "--regimes",
        default="light,baseline,stress,hetero",
        help="Comma-separated regime names.",
    )
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=11)
    parser.add_argument(
        "--edge-selection", choices=["top-flow", "random", "all"], default="top-flow"
    )
    parser.add_argument("--max-edges", type=int, default=10)
    parser.add_argument("--bisect-iters", type=int, default=8)
    parser.add_argument("--initial-step-rel", type=float, default=0.01)
    parser.add_argument("--expand-factor", type=float, default=1.8)
    parser.add_argument("--max-expand-iters", type=int, default=14)
    parser.add_argument("--max-rel-plus", type=float, default=5.0)
    parser.add_argument(
        "--active-set-level", choices=["commodity", "edge"], default="commodity"
    )
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--disagg-reg", type=float, default=1e-6)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--mismatch-tol-pct", type=float, default=0.5)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument(
        "--smoke", action="store_true", help="Run a tiny quick benchmark."
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run an intermediate benchmark between smoke and full.",
    )
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
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
        args.num_nodes = min(args.num_nodes, 24)
        args.max_edges = min(args.max_edges, 4)
        args.bisect_iters = min(args.bisect_iters, 5)
        args.graph_families = "city_like,regular"
        args.regimes = "light,baseline"
        args.max_expand_iters = min(args.max_expand_iters, 8)
        args.max_rel_plus = min(args.max_rel_plus, 2.0)
    elif args.medium:
        args.num_trials = min(args.num_trials, 2)
        args.num_nodes = min(args.num_nodes, 28)
        args.max_edges = min(args.max_edges, 6)
        args.bisect_iters = min(args.bisect_iters, 6)
        args.graph_families = "city_like,planar,random,small_world"
        args.regimes = "light,baseline,stress"
        args.max_expand_iters = min(args.max_expand_iters, 10)
        args.max_rel_plus = min(args.max_rel_plus, 3.0)

    if args.num_nodes < 8:
        parser.error("--num-nodes must be at least 8.")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1.")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1.")
    if args.max_edges < 1:
        parser.error("--max-edges must be at least 1.")
    if args.bisect_iters < 1:
        parser.error("--bisect-iters must be at least 1.")
    if args.initial_step_rel <= 0:
        parser.error("--initial-step-rel must be > 0.")
    if args.expand_factor <= 1.0:
        parser.error("--expand-factor must be > 1.")
    if args.max_expand_iters < 1:
        parser.error("--max-expand-iters must be at least 1.")
    if args.max_rel_plus <= 0:
        parser.error("--max-rel-plus must be > 0.")
    if args.disagg_reg < 0:
        parser.error("--disagg-reg must be >= 0.")
    if args.mismatch_tol_pct < 0:
        parser.error("--mismatch-tol-pct must be >= 0.")
    return args


def main() -> None:
    args = _parse_args()
    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK
    graph_families_raw = _parse_list(args.graph_families)
    graph_families = []
    for gf in graph_families_raw:
        normalized = GRAPH_FAMILY_ALIASES.get(gf, gf)
        if normalized not in graph_families:
            graph_families.append(normalized)
    regime_names = _parse_list(args.regimes)

    for family in graph_families:
        if family not in GRAPH_FAMILY_LABELS:
            raise ValueError(f"Unknown graph family '{family}'.")
    for regime_name in regime_names:
        if regime_name not in REGIMES:
            raise ValueError(f"Unknown regime '{regime_name}'.")

    _print_hypotheses()

    rows: list[dict] = []
    summary_rows: list[dict] = []
    total_tests = 0
    total_directional_attempted = 0
    total_directional_kept = 0

    for graph_family in graph_families:
        for regime_name in regime_names:
            regime = REGIMES[regime_name]
            cfg_rows: list[dict] = []
            cfg_attempted_dirs = 0
            cfg_dropped_dirs = 0

            for trial_idx in range(args.num_trials):
                seed = args.seed + trial_idx * args.seed_step
                graph = _build_graph(
                    graph_family=graph_family, num_nodes=args.num_nodes, seed=seed
                )
                _ensure_population_attribute(graph, seed=seed + 101)
                _apply_beta_heterogeneity(
                    graph,
                    beta_cv=regime.beta_cv,
                    seed=seed + 303,
                    min_beta=args.min_beta,
                )
                demands = _build_demands(graph, regime=regime)

                baseline = bp._build_baseline_system(
                    graph=graph,
                    demands=demands,
                    solver=solver,
                    eps_active=args.eps_active,
                    regularization=args.regularization,
                    disagg_reg=args.disagg_reg,
                )
                baseline_indicator = bp._active_set_indicator(
                    baseline.flow_matrix,
                    eps_active=args.eps_active,
                    level=args.active_set_level,
                )

                baseline_edge_flow = np.sum(baseline.flow_matrix, axis=0)
                selected_indices = bp._select_edge_indices(
                    baseline_edge_flow=baseline_edge_flow,
                    strategy=args.edge_selection,
                    max_edges=args.max_edges,
                    seed=seed,
                )
                total_tests += int(selected_indices.size)

                for edge_idx in selected_indices:
                    edge_idx = int(edge_idx)
                    beta_k = float(baseline.beta[edge_idx])
                    if beta_k <= args.min_beta:
                        continue
                    cfg_attempted_dirs += 2

                    eta_minus, eta_plus, evt_minus, evt_plus, meta = (
                        bp._first_breakpoints(
                            baseline=baseline,
                            edge_idx=edge_idx,
                            tol=args.tol,
                            active_set_level=args.active_set_level,
                            return_meta=True,
                        )
                    )
                    pred_minus_rel = float(eta_minus / beta_k)
                    pred_plus_rel = float(eta_plus / beta_k)

                    obs_minus_rel = _observed_direction_breakpoint(
                        baseline=baseline,
                        baseline_indicator=baseline_indicator,
                        edge_idx=edge_idx,
                        direction=-1,
                        solver=solver,
                        min_beta=args.min_beta,
                        active_set_level=args.active_set_level,
                        disagg_reg=args.disagg_reg,
                        bisect_iters=args.bisect_iters,
                        initial_step_rel=args.initial_step_rel,
                        expand_factor=args.expand_factor,
                        max_expand_iters=args.max_expand_iters,
                        max_rel_plus=args.max_rel_plus,
                        seed_rel=pred_minus_rel if np.isfinite(pred_minus_rel) else None,
                    )
                    obs_plus_rel = _observed_direction_breakpoint(
                        baseline=baseline,
                        baseline_indicator=baseline_indicator,
                        edge_idx=edge_idx,
                        direction=+1,
                        solver=solver,
                        min_beta=args.min_beta,
                        active_set_level=args.active_set_level,
                        disagg_reg=args.disagg_reg,
                        bisect_iters=args.bisect_iters,
                        initial_step_rel=args.initial_step_rel,
                        expand_factor=args.expand_factor,
                        max_expand_iters=args.max_expand_iters,
                        max_rel_plus=args.max_rel_plus,
                        seed_rel=pred_plus_rel if np.isfinite(pred_plus_rel) else None,
                    )

                    directional = [
                        (-1, pred_minus_rel, obs_minus_rel, evt_minus),
                        (+1, pred_plus_rel, obs_plus_rel, evt_plus),
                    ]
                    for direction, pred_rel, obs_rel, event in directional:
                        if (not np.isfinite(pred_rel)) or obs_rel is None or (not np.isfinite(obs_rel)):
                            cfg_dropped_dirs += 1
                            continue

                        pred_radius_pct = 100.0 * float(pred_rel)
                        obs_radius_pct = 100.0 * float(obs_rel)
                        pred_signed_pct = float(direction) * pred_radius_pct
                        obs_signed_pct = float(direction) * obs_radius_pct
                        abs_error = abs(pred_signed_pct - obs_signed_pct)
                        row = {
                            "graph_family": graph_family,
                            "regime": regime_name,
                            "trial": trial_idx,
                            "seed": seed,
                            "num_nodes": int(graph.number_of_nodes()),
                            "num_edges": int(graph.number_of_edges()),
                            "num_commodities": int(demands.shape[0]),
                            "edge_index": edge_idx,
                            "edge": bp._format_edge(baseline.edges[edge_idx]),
                            "direction": int(direction),
                            "direction_label": "-" if direction < 0 else "+",
                            "event": event,
                            "pred_gap_sym": float(meta["gap_sym"]),
                            "pred_radius_pct": pred_radius_pct,
                            "obs_radius_pct": obs_radius_pct,
                            "pred_signed_pct": pred_signed_pct,
                            "obs_signed_pct": obs_signed_pct,
                            "abs_error_pct": abs_error,
                            "is_mismatch": bool(abs_error > args.mismatch_tol_pct),
                        }
                        cfg_rows.append(row)
                        rows.append(row)

            if not cfg_rows:
                continue

            obs = np.array([float(r["obs_signed_pct"]) for r in cfg_rows], dtype=float)
            pred = np.array([float(r["pred_signed_pct"]) for r in cfg_rows], dtype=float)
            mismatch = np.array([bool(r["is_mismatch"]) for r in cfg_rows], dtype=bool)
            enter = np.array([r["event"] == "enter" for r in cfg_rows], dtype=bool)

            mae = float(np.mean(np.abs(pred - obs)))
            corr = _safe_corr(pred, obs)

            summary = {
                "graph_family": graph_family,
                "graph_label": GRAPH_FAMILY_LABELS[graph_family],
                "regime": regime_name,
                "regime_label": regime.label,
                "num_edges_selected": int(cfg_attempted_dirs // 2),
                "num_directional_attempted": int(cfg_attempted_dirs),
                "num_directional_kept": int(len(cfg_rows)),
                "num_directional_dropped": int(cfg_dropped_dirs),
                "dropped_share_pct": (
                    100.0 * float(cfg_dropped_dirs / cfg_attempted_dirs)
                    if cfg_attempted_dirs > 0
                    else float("nan")
                ),
                "mae_pp": mae,
                "corr": corr,
                "mismatch_rate": 100.0 * float(np.mean(mismatch)),
                "winner_enter_share": 100.0 * float(np.mean(enter)),
                "mean_abs_obs_radius_pct": float(np.mean(np.abs(obs))),
                "obs_signed_min_pct": float(np.min(obs)),
                "obs_signed_max_pct": float(np.max(obs)),
            }
            summary_rows.append(summary)
            total_directional_attempted += cfg_attempted_dirs
            total_directional_kept += len(cfg_rows)
            print(
                f"{GRAPH_FAMILY_LABELS[graph_family]:>20s} | {regime.label:>18s} | "
                f"n_dir={summary['num_directional_kept']:4d} | "
                f"drop={summary['dropped_share_pct']:5.1f}% | "
                f"corr={summary['corr']:.3f} | "
                f"mismatch={summary['mismatch_rate']:5.1f}%"
            )

    _write_rows(args.output_table, rows)
    _write_rows(args.output_summary, summary_rows)
    _plot_directional_scatter(
        rows=rows,
        graph_families=graph_families,
        regime_names=regime_names,
        output_figure=args.output_figure,
    )

    print("")
    print(f"Total tested edges: {total_tests}")
    if total_directional_attempted > 0:
        kept_share = 100.0 * total_directional_kept / total_directional_attempted
        print(
            f"Directional samples kept: {total_directional_kept}/{total_directional_attempted} "
            f"({kept_share:.1f}%)"
        )
    print(f"Saved detail table: {args.output_table}")
    print(f"Saved summary table: {args.output_summary}")
    print(f"Saved scatter figure: {args.output_figure}")


if __name__ == "__main__":
    main()

# %%
