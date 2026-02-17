# %%
"""Compare SCGC intervention pipeline against brute-force intervention screening.

For each synthetic graph/regime instance:
1) solve baseline multicommodity TAP and SCGC values,
2) select a candidate edge pool,
3) build an SCGC top-k intervention plan (beta reductions),
4) brute-force all size-k subsets in the same candidate pool,
5) compare realized social-cost deltas and runtime/solver calls.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
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

OUTPUT_FIGURE = "figs/scgc-pipeline-vs-bruteforce.pdf"
OUTPUT_DETAIL = "cache/scgc-pipeline-vs-bruteforce-detail.csv"
OUTPUT_SUMMARY = "cache/scgc-pipeline-vs-bruteforce-summary.csv"

GRAPH_FAMILY_LABELS = {
    "city_like": "City-like",
    "planar": "Planar",
    "regular": "Regular",
    "random": "Random",
    "small_world": "Small-world",
}

GRAPH_FAMILY_COLORS = {
    "city_like": "#1b9e77",
    "planar": "#377eb8",
    "regular": "#4daf4a",
    "random": "#ff7f00",
    "small_world": "#984ea3",
}

GRAPH_FAMILY_ALIASES = {
    "city_planar": "city_like",
    "er_random": "random",
    "square_lattice": "regular",
    "triangular_lattice": "planar",
}

BUDGET_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "P",
    6: "X",
}


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


def _parse_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


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

    if graph_family == "planar":
        graph = gr.random_planar_graph(n_nodes=num_nodes, seed=seed, alpha="random")
        edges = list(graph.edges)
        rng = np.random.default_rng(seed + 17)
        beta = np.maximum(1e-6, rng.uniform(20.0, 120.0, size=len(edges)))
        nx.set_edge_attributes(graph, dict(zip(edges, beta)), "beta")
        return graph

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


def _select_candidate_indices(
    flow: np.ndarray,
    gradient: np.ndarray,
    mode: str,
    max_candidates: int,
    random_seed: int,
) -> np.ndarray:
    n = int(flow.size)
    all_indices = np.arange(n, dtype=int)
    if mode == "all" or max_candidates >= n:
        return all_indices
    if mode == "top-flow":
        order = np.argsort(-np.asarray(flow, dtype=float))
        return np.sort(order[:max_candidates])
    if mode == "top-abs-scgc":
        order = np.argsort(-np.abs(np.asarray(gradient, dtype=float)))
        return np.sort(order[:max_candidates])
    if mode == "random":
        rng = np.random.default_rng(random_seed)
        return np.sort(rng.choice(all_indices, size=max_candidates, replace=False))
    raise ValueError(f"Unknown candidate mode '{mode}'.")


def _plot_comparison(rows_df: pd.DataFrame, output_figure: str) -> None:
    if rows_df.empty:
        return

    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=12)
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.4))

    for _, row in rows_df.iterrows():
        graph_family = str(row["graph_family"])
        budget = int(row["budget"])
        color = GRAPH_FAMILY_COLORS.get(graph_family, "#666666")
        marker = BUDGET_MARKERS.get(budget, "o")
        ax.scatter(
            float(row["bruteforce_delta_sc_pct"]),
            float(row["scgc_delta_sc_pct"]),
            s=45,
            marker=marker,
            c=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
        )

    x_all = rows_df["bruteforce_delta_sc_pct"].to_numpy(dtype=float)
    y_all = rows_df["scgc_delta_sc_pct"].to_numpy(dtype=float)
    low = float(min(np.min(x_all), np.min(y_all)))
    high = float(max(np.max(x_all), np.max(y_all)))
    if np.isclose(low, high):
        low -= 1.0
        high += 1.0
    pad = 0.05 * (high - low)
    lo = low - pad
    hi = high + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Brute-force best $\Delta SC$ [\%]")
    ax.set_ylabel(r"SCGC pipeline $\Delta SC$ [\%]")
    ax.grid(alpha=0.28)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=GRAPH_FAMILY_COLORS[family],
            markeredgecolor=GRAPH_FAMILY_COLORS[family],
            markersize=7.5,
            label=GRAPH_FAMILY_LABELS[family],
        )
        for family in sorted(rows_df["graph_family"].unique())
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=BUDGET_MARKERS.get(int(budget), "o"),
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7.0,
            label=f"Budget {int(budget)}",
        )
        for budget in sorted(rows_df["budget"].unique())
    ]

    legend_graph = ax.legend(
        handles=color_handles,
        title="Graph family",
        loc="upper left",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.95,
    )
    ax.add_artist(legend_graph)
    ax.legend(
        handles=marker_handles,
        title="Budget",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.95,
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SCGC intervention ranking with brute-force screening.",
    )
    parser.add_argument(
        "--graph-families",
        default="city_like,regular,random",
        help="Comma-separated graph families.",
    )
    parser.add_argument(
        "--regimes",
        default="light,baseline,stress",
        help="Comma-separated regime names.",
    )
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=11)
    parser.add_argument(
        "--candidate-mode",
        choices=["top-flow", "top-abs-scgc", "random", "all"],
        default="top-flow",
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=3)
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=1000,
        help="Skip budgets with more combinations than this limit.",
    )
    parser.add_argument(
        "--rel-step",
        type=float,
        default=0.01,
        help="Relative beta reduction per intervened edge (e.g. 0.01 = 1%%).",
    )
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--optimal-tol-pct", type=float, default=1e-9)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny configuration for quick checks.",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run an intermediate benchmark between smoke and full.",
    )
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
        args.num_nodes = min(args.num_nodes, 18)
        args.max_candidates = min(args.max_candidates, 6)
        args.max_budget = min(args.max_budget, 2)
        args.max_combinations = min(args.max_combinations, 200)
        args.graph_families = "city_like,regular"
        args.regimes = "light,baseline"
    elif args.medium:
        args.num_trials = min(args.num_trials, 2)
        args.num_nodes = min(args.num_nodes, 22)
        args.max_candidates = min(args.max_candidates, 7)
        args.max_budget = min(args.max_budget, 3)
        args.max_combinations = min(args.max_combinations, 700)
        args.graph_families = "city_like,planar,regular,random,small_world"
        args.regimes = "light,baseline,stress"

    if args.num_nodes < 8:
        parser.error("--num-nodes must be at least 8.")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1.")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1.")
    if args.max_candidates < 2:
        parser.error("--max-candidates must be at least 2.")
    if args.max_budget < 1:
        parser.error("--max-budget must be at least 1.")
    if args.max_combinations < 1:
        parser.error("--max-combinations must be at least 1.")
    if args.rel_step <= 0 or args.rel_step >= 1:
        parser.error("--rel-step must be in (0, 1).")
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

    detail_rows: list[dict] = []
    total_solver_calls = 0
    total_experiments = 0

    for graph_family in graph_families:
        for regime_name in regime_names:
            regime = REGIMES[regime_name]
            for trial_idx in range(args.num_trials):
                seed = args.seed + trial_idx * args.seed_step
                graph = _build_graph(graph_family, num_nodes=args.num_nodes, seed=seed)
                _ensure_population_attribute(graph, seed=seed + 101)
                _apply_beta_heterogeneity(
                    graph,
                    beta_cv=regime.beta_cv,
                    seed=seed + 303,
                    min_beta=args.min_beta,
                )
                demands = _build_demands(graph, regime=regime)

                edge_list = list(graph.edges)
                alpha = np.array(
                    [float(graph.edges[e]["alpha"]) for e in edge_list], dtype=float
                )
                beta = np.array(
                    [float(graph.edges[e]["beta"]) for e in edge_list], dtype=float
                )

                flow_matrix, _ = mc.solve_multicommodity_tap(
                    graph,
                    demands,
                    solver=solver,
                    pos_flows=True,
                    return_fw=True,
                )
                flow = np.sum(flow_matrix, axis=0)
                dsc_dict = mcsc.derivative_social_cost(
                    graph,
                    flow_matrix,
                    demands,
                    eps=args.eps_active,
                    demands_to_sinks=False,
                )
                gradient = np.array([float(dsc_dict[e]) for e in edge_list], dtype=float)

                base_sc = sc.total_social_cost(graph, flow, alpha=alpha, beta=beta)
                base_scale = max(abs(base_sc), 1e-12)

                candidate_indices = _select_candidate_indices(
                    flow=flow,
                    gradient=gradient,
                    mode=args.candidate_mode,
                    max_candidates=args.max_candidates,
                    random_seed=seed + 701,
                )
                n_candidates = int(candidate_indices.size)
                if n_candidates == 0:
                    continue

                predicted_delta_sc_pct = (
                    100.0 * gradient * (-args.rel_step * beta) / base_scale
                )
                order = candidate_indices[
                    np.argsort(predicted_delta_sc_pct[candidate_indices], kind="mergesort")
                ]

                cache: dict[tuple[int, ...], float] = {}
                solver_counter = [0]

                def simulate(edge_ids: tuple[int, ...] | list[int] | np.ndarray) -> float:
                    key = tuple(sorted(set(int(i) for i in edge_ids)))
                    if key in cache:
                        return cache[key]
                    beta_trial = beta.copy()
                    if key:
                        idx = np.asarray(key, dtype=int)
                        beta_trial[idx] = np.maximum(
                            args.min_beta,
                            beta[idx] * (1.0 - args.rel_step),
                        )
                    flow_trial = mc.solve_multicommodity_tap(
                        graph,
                        demands,
                        alpha=alpha,
                        beta=beta_trial,
                        solver=solver,
                        pos_flows=True,
                    )
                    solver_counter[0] += 1
                    sc_trial = sc.total_social_cost(
                        graph,
                        flow_trial,
                        alpha=alpha,
                        beta=beta_trial,
                    )
                    delta_pct = 100.0 * float((sc_trial - base_sc) / base_scale)
                    cache[key] = delta_pct
                    return delta_pct

                budgets = []
                combos_by_budget = {}
                max_budget_eff = min(args.max_budget, n_candidates)
                for budget in range(1, max_budget_eff + 1):
                    n_combo = math.comb(n_candidates, budget)
                    if n_combo <= args.max_combinations:
                        budgets.append(budget)
                        combos_by_budget[budget] = n_combo
                    else:
                        print(
                            f"Skipping budget {budget} for {graph_family}/{regime_name}/trial={trial_idx}: "
                            f"C({n_candidates},{budget})={n_combo} exceeds --max-combinations={args.max_combinations}."
                        )
                if not budgets:
                    print(
                        f"No feasible budgets for {graph_family}/{regime_name}/trial={trial_idx}; "
                        "increase --max-combinations or reduce --max-candidates."
                    )
                    continue

                total_experiments += 1

                for budget in budgets:
                    scgc_selected = tuple(int(v) for v in order[:budget])
                    scgc_selected_sorted = tuple(sorted(scgc_selected))

                    calls_before_scgc = solver_counter[0]
                    t_scgc_start = time.perf_counter()
                    scgc_delta = simulate(scgc_selected_sorted)
                    t_scgc = time.perf_counter() - t_scgc_start
                    scgc_solver_calls = solver_counter[0] - calls_before_scgc

                    calls_before_bruteforce = solver_counter[0]
                    t_brute_start = time.perf_counter()
                    brute_best = float("inf")
                    brute_best_subset: tuple[int, ...] | None = None
                    for combo in itertools.combinations(candidate_indices.tolist(), budget):
                        combo_sorted = tuple(sorted(int(v) for v in combo))
                        delta = simulate(combo_sorted)
                        if delta < brute_best:
                            brute_best = float(delta)
                            brute_best_subset = combo_sorted
                    t_brute = time.perf_counter() - t_brute_start
                    brute_solver_calls = solver_counter[0] - calls_before_bruteforce

                    if brute_best_subset is None:
                        continue

                    regret_pct = float(scgc_delta - brute_best)
                    brute_gain = float(-brute_best)
                    scgc_gain = float(-scgc_delta)
                    gain_ratio = (
                        float(scgc_gain / brute_gain)
                        if brute_gain > 1e-12
                        else float("nan")
                    )
                    is_optimal = abs(regret_pct) <= args.optimal_tol_pct

                    detail_rows.append(
                        {
                            "graph_family": graph_family,
                            "regime": regime_name,
                            "trial": int(trial_idx),
                            "seed": int(seed),
                            "num_nodes": int(graph.number_of_nodes()),
                            "num_edges": int(graph.number_of_edges()),
                            "num_commodities": int(demands.shape[0]),
                            "candidate_mode": args.candidate_mode,
                            "n_candidates": int(n_candidates),
                            "budget": int(budget),
                            "num_combinations": int(combos_by_budget[budget]),
                            "rel_step_pct": float(100.0 * args.rel_step),
                            "scgc_delta_sc_pct": float(scgc_delta),
                            "bruteforce_delta_sc_pct": float(brute_best),
                            "regret_sc_pct": regret_pct,
                            "scgc_gain_over_best_gain": gain_ratio,
                            "is_optimal_match": bool(is_optimal),
                            "scgc_selected": ";".join(
                                str(v) for v in scgc_selected_sorted
                            ),
                            "bruteforce_selected": ";".join(
                                str(v) for v in brute_best_subset
                            ),
                            "overlap_with_best": int(
                                len(set(scgc_selected_sorted) & set(brute_best_subset))
                            ),
                            "runtime_scgc_s": float(t_scgc),
                            "runtime_bruteforce_s": float(t_brute),
                            "solver_calls_scgc": int(scgc_solver_calls),
                            "solver_calls_bruteforce": int(brute_solver_calls),
                        }
                    )

                total_solver_calls += solver_counter[0]

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        _write_rows(args.output_detail, [])
        _write_rows(args.output_summary, [])
        print("No results generated (all configurations were skipped).")
        return

    summary_rows = []
    group_cols = ["graph_family", "regime", "budget"]
    grouped = detail_df.groupby(group_cols, as_index=False)
    for _, frame in grouped:
        first = frame.iloc[0]
        finite_ratio = frame["scgc_gain_over_best_gain"].to_numpy(dtype=float)
        finite_ratio = finite_ratio[np.isfinite(finite_ratio)]
        summary_rows.append(
            {
                "graph_family": str(first["graph_family"]),
                "regime": str(first["regime"]),
                "budget": int(first["budget"]),
                "n_cases": int(len(frame)),
                "mean_scgc_delta_sc_pct": float(frame["scgc_delta_sc_pct"].mean()),
                "mean_bruteforce_delta_sc_pct": float(
                    frame["bruteforce_delta_sc_pct"].mean()
                ),
                "mean_regret_sc_pct": float(frame["regret_sc_pct"].mean()),
                "median_regret_sc_pct": float(frame["regret_sc_pct"].median()),
                "optimal_match_share_pct": float(
                    100.0 * frame["is_optimal_match"].mean()
                ),
                "mean_gain_ratio": (
                    float(np.mean(finite_ratio)) if len(finite_ratio) else float("nan")
                ),
                "median_gain_ratio": (
                    float(np.median(finite_ratio)) if len(finite_ratio) else float("nan")
                ),
                "mean_runtime_scgc_s": float(frame["runtime_scgc_s"].mean()),
                "mean_runtime_bruteforce_s": float(
                    frame["runtime_bruteforce_s"].mean()
                ),
                "mean_solver_calls_scgc": float(frame["solver_calls_scgc"].mean()),
                "mean_solver_calls_bruteforce": float(
                    frame["solver_calls_bruteforce"].mean()
                ),
            }
        )

    overall_finite_ratio = detail_df["scgc_gain_over_best_gain"].to_numpy(dtype=float)
    overall_finite_ratio = overall_finite_ratio[np.isfinite(overall_finite_ratio)]
    summary_rows.append(
        {
            "graph_family": "ALL",
            "regime": "ALL",
            "budget": -1,
            "n_cases": int(len(detail_df)),
            "mean_scgc_delta_sc_pct": float(detail_df["scgc_delta_sc_pct"].mean()),
            "mean_bruteforce_delta_sc_pct": float(
                detail_df["bruteforce_delta_sc_pct"].mean()
            ),
            "mean_regret_sc_pct": float(detail_df["regret_sc_pct"].mean()),
            "median_regret_sc_pct": float(detail_df["regret_sc_pct"].median()),
            "optimal_match_share_pct": float(
                100.0 * detail_df["is_optimal_match"].mean()
            ),
            "mean_gain_ratio": (
                float(np.mean(overall_finite_ratio))
                if len(overall_finite_ratio)
                else float("nan")
            ),
            "median_gain_ratio": (
                float(np.median(overall_finite_ratio))
                if len(overall_finite_ratio)
                else float("nan")
            ),
            "mean_runtime_scgc_s": float(detail_df["runtime_scgc_s"].mean()),
            "mean_runtime_bruteforce_s": float(
                detail_df["runtime_bruteforce_s"].mean()
            ),
            "mean_solver_calls_scgc": float(detail_df["solver_calls_scgc"].mean()),
            "mean_solver_calls_bruteforce": float(
                detail_df["solver_calls_bruteforce"].mean()
            ),
        }
    )
    summary_df = pd.DataFrame(summary_rows)

    detail_df.to_csv(args.output_detail, index=False)
    summary_df.to_csv(args.output_summary, index=False)
    _plot_comparison(detail_df, args.output_figure)

    overall = summary_df[
        (summary_df["graph_family"] == "ALL") & (summary_df["regime"] == "ALL")
    ].iloc[0]
    print(
        "Overall: "
        f"mean regret={overall['mean_regret_sc_pct']:.4f}% | "
        f"optimal-match share={overall['optimal_match_share_pct']:.2f}% | "
        f"mean gain ratio={overall['mean_gain_ratio']:.4f}"
    )
    print(
        f"Saved detail: {args.output_detail}\n"
        f"Saved summary: {args.output_summary}\n"
        f"Saved figure: {args.output_figure}"
    )
    print(
        f"Evaluated {len(detail_df)} (instance,budget) points across {total_experiments} instances; "
        f"total cached solver calls={total_solver_calls}."
    )


if __name__ == "__main__":
    main()
