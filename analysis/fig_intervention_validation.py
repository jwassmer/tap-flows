# %%
"""Intervention validation for SCGC-based edge ranking.

This script tests whether first-order SCGC ranking predicts realized social-cost
improvements from small top-k edge interventions better than simple baselines.

Intervention model:
- pick a method-specific ranking over candidate edges,
- choose top-k edges,
- reduce their free-flow times by a fixed relative factor rho,
- re-solve multicommodity TAP,
- evaluate realized change in total social cost.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import rankdata

# Allow direct execution via `python analysis/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import SocialCost as sc
from src import multiCommoditySocialCost as mcsc
from src.figure_style import apply_publication_style

_LINEAR_TRANSFER_PATH = REPO_ROOT / "analysis" / "fig_linear_predictor_transfer.py"
_LINEAR_SPEC = importlib.util.spec_from_file_location(
    "linear_transfer_core", _LINEAR_TRANSFER_PATH
)
if _LINEAR_SPEC is None or _LINEAR_SPEC.loader is None:
    raise RuntimeError(f"Could not load module at {_LINEAR_TRANSFER_PATH}")

lpt = importlib.util.module_from_spec(_LINEAR_SPEC)
sys.modules[_LINEAR_SPEC.name] = lpt
_LINEAR_SPEC.loader.exec_module(lpt)

bp = lpt.bp

OUTPUT_FIGURE = "figs/intervention-validation-scgc.pdf"
OUTPUT_TABLE = "cache/intervention-validation-scgc-detail.csv"
OUTPUT_EDGE_TABLE = "cache/intervention-validation-scgc-edge.csv"
OUTPUT_SUMMARY = "cache/intervention-validation-scgc-summary.csv"
OUTPUT_CORRELATION = "cache/intervention-validation-scgc-correlation.csv"
OUTPUT_CORRELATION_SUMMARY = (
    "cache/intervention-validation-scgc-correlation-summary.csv"
)

METHOD_ORDER = ["scgc_linear", "flow", "betweenness", "random"]
METHOD_LABELS = {
    "scgc_linear": "SCGC linear",
    "flow": "High flow",
    "betweenness": "Betweenness",
    "random": "Random",
}
METHOD_COLORS = {
    "scgc_linear": "#1f78b4",
    "flow": "#ff7f00",
    "betweenness": "#33a02c",
    "random": "#6a3d9a",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SCGC ranking against realized intervention effects.",
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
        "--edge-selection",
        choices=["top-flow", "random", "all"],
        default="top-flow",
    )
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument(
        "--rel-step",
        type=float,
        default=0.01,
        help="Relative beta reduction per intervened edge (e.g. 0.01 = 1%).",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=6,
        help="Number of random rankings to average for the random baseline.",
    )
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--disagg-reg", type=float, default=1e-6)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny configuration for a quick end-to-end test.",
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Run an intermediate benchmark between smoke and full.",
    )
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
    parser.add_argument("--output-edge-table", default=OUTPUT_EDGE_TABLE)
    parser.add_argument("--output-summary", default=OUTPUT_SUMMARY)
    parser.add_argument("--output-correlation", default=OUTPUT_CORRELATION)
    parser.add_argument(
        "--output-correlation-summary", default=OUTPUT_CORRELATION_SUMMARY
    )

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
        args.max_candidates = min(args.max_candidates, 5)
        args.max_budget = min(args.max_budget, 3)
        args.random_repeats = min(args.random_repeats, 3)
        args.graph_families = "city_like,regular"
        args.regimes = "light,baseline"
    elif args.medium:
        args.num_trials = min(args.num_trials, 2)
        args.num_nodes = min(args.num_nodes, 24)
        args.max_candidates = min(args.max_candidates, 8)
        args.max_budget = min(args.max_budget, 4)
        args.random_repeats = min(args.random_repeats, 5)
        args.graph_families = "city_like,planar,regular,random,small_world"
        args.regimes = "light,baseline,stress"

    if args.num_nodes < 8:
        parser.error("--num-nodes must be at least 8.")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1.")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1.")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1.")
    if args.max_budget < 1:
        parser.error("--max-budget must be at least 1.")
    if args.rel_step <= 0 or args.rel_step >= 1:
        parser.error("--rel-step must be in (0, 1).")
    if args.random_repeats < 1:
        parser.error("--random-repeats must be at least 1.")
    if args.disagg_reg < 0:
        parser.error("--disagg-reg must be >= 0.")

    return args


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")

    rx = rankdata(x_arr, method="average")
    ry = rankdata(y_arr, method="average")
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def _subset_key(
    edge_indices: np.ndarray | list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    unique = np.unique(np.asarray(edge_indices, dtype=int))
    return tuple(int(v) for v in unique)


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


def _compute_method_scores(
    baseline_beta: np.ndarray,
    baseline_flow: np.ndarray,
    dsc: np.ndarray,
    betweenness: np.ndarray,
    rel_step: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    predicted_delta_sc = dsc * (-rel_step * baseline_beta)
    rng = np.random.default_rng(random_seed)

    return {
        "scgc_linear": np.asarray(predicted_delta_sc, dtype=float),
        "flow": -np.asarray(baseline_flow, dtype=float),
        "betweenness": -np.asarray(betweenness, dtype=float),
        "random": rng.random(len(baseline_beta)),
    }


def _rank_edges(score: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate_indices, dtype=int)
    return cand[np.argsort(score[cand], kind="mergesort")]


def _annotate_best_flags(rows: list[dict], tol: float = 1e-12) -> None:
    grouped: dict[tuple[int, int], list[int]] = {}
    for idx, row in enumerate(rows):
        key = (int(row["instance_id"]), int(row["budget"]))
        grouped.setdefault(key, []).append(idx)

    for _, indices in grouped.items():
        values = np.array(
            [float(rows[i]["delta_sc_pct"]) for i in indices], dtype=float
        )
        winner = float(np.min(values))
        for i in indices:
            rows[i]["is_best"] = bool(float(rows[i]["delta_sc_pct"]) <= winner + tol)


def _build_summary_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    result: list[dict] = []
    budgets = sorted({int(row["budget"]) for row in rows})

    lookup_scgc = {
        (int(row["instance_id"]), int(row["budget"])): float(row["delta_sc_pct"])
        for row in rows
        if row["method"] == "scgc_linear"
    }

    for budget in budgets:
        for method in METHOD_ORDER:
            subset = [
                row
                for row in rows
                if int(row["budget"]) == budget and row["method"] == method
            ]
            if not subset:
                continue

            delta = np.array(
                [float(row["delta_sc_pct"]) for row in subset], dtype=float
            )
            best = np.array([bool(row["is_best"]) for row in subset], dtype=bool)

            row_out = {
                "budget": int(budget),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_instances": int(len(subset)),
                "mean_delta_sc_pct": float(np.mean(delta)),
                "median_delta_sc_pct": float(np.median(delta)),
                "std_delta_sc_pct": float(np.std(delta)),
                "sem_delta_sc_pct": float(np.std(delta) / np.sqrt(len(delta))),
                "min_delta_sc_pct": float(np.min(delta)),
                "max_delta_sc_pct": float(np.max(delta)),
                "best_share_pct": 100.0 * float(np.mean(best)),
                "mean_regret_to_oracle1_pct": float("nan"),
                "scgc_win_rate_pct": float("nan"),
                "method_win_rate_vs_scgc_pct": float("nan"),
                "tie_rate_vs_scgc_pct": float("nan"),
                "mean_method_minus_scgc_pct": float("nan"),
            }

            if budget == 1:
                regrets = np.array(
                    [float(r["regret_to_oracle1_pct"]) for r in subset], dtype=float
                )
                finite_mask = np.isfinite(regrets)
                if np.any(finite_mask):
                    row_out["mean_regret_to_oracle1_pct"] = float(
                        np.mean(regrets[finite_mask])
                    )

            if method != "scgc_linear":
                paired = []
                for r in subset:
                    key = (int(r["instance_id"]), int(r["budget"]))
                    if key not in lookup_scgc:
                        continue
                    scgc_delta = lookup_scgc[key]
                    method_delta = float(r["delta_sc_pct"])
                    paired.append((scgc_delta, method_delta))

                if paired:
                    scgc_arr = np.array([p[0] for p in paired], dtype=float)
                    method_arr = np.array([p[1] for p in paired], dtype=float)
                    scgc_wins = np.mean(scgc_arr < method_arr - 1e-12)
                    method_wins = np.mean(method_arr < scgc_arr - 1e-12)
                    ties = 1.0 - scgc_wins - method_wins

                    row_out["scgc_win_rate_pct"] = 100.0 * float(scgc_wins)
                    row_out["method_win_rate_vs_scgc_pct"] = 100.0 * float(method_wins)
                    row_out["tie_rate_vs_scgc_pct"] = 100.0 * float(ties)
                    row_out["mean_method_minus_scgc_pct"] = float(
                        np.mean(method_arr - scgc_arr)
                    )

            result.append(row_out)

    return result


def _build_correlation_summary(correlation_rows: list[dict]) -> list[dict]:
    if not correlation_rows:
        return []

    summary: list[dict] = []
    for method in METHOD_ORDER:
        subset = [r for r in correlation_rows if r["method"] == method]
        if not subset:
            continue
        values = np.array([float(r["spearman"]) for r in subset], dtype=float)
        finite = values[np.isfinite(values)]

        summary.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_instances": int(len(subset)),
                "n_finite": int(len(finite)),
                "mean_spearman": (
                    float(np.mean(finite)) if len(finite) else float("nan")
                ),
                "median_spearman": (
                    float(np.median(finite)) if len(finite) else float("nan")
                ),
                "std_spearman": float(np.std(finite)) if len(finite) else float("nan"),
            }
        )

    return summary


def _plot_budget_curve(rows: list[dict], output_figure: str) -> None:
    if not rows:
        return

    output = Path(output_figure)
    output.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=13)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))

    budgets = sorted({int(r["budget"]) for r in rows})
    n_instances = len({int(r["instance_id"]) for r in rows})

    for method in METHOD_ORDER:
        medians = []
        q25 = []
        q75 = []
        for k in budgets:
            vals = np.array(
                [
                    float(r["delta_sc_pct"])
                    for r in rows
                    if int(r["budget"]) == k and r["method"] == method
                ],
                dtype=float,
            )
            if vals.size == 0:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
                continue
            medians.append(float(np.median(vals)))
            q25.append(float(np.percentile(vals, 25)))
            q75.append(float(np.percentile(vals, 75)))

        x = np.array(budgets, dtype=float)
        y = np.array(medians, dtype=float)
        lo = np.array(q25, dtype=float)
        hi = np.array(q75, dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue

        color = METHOD_COLORS[method]
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            color=color,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(x[mask], lo[mask], hi[mask], color=color, alpha=0.15)

    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=1.0)
    ax.set_xticks(budgets)
    ax.set_xlabel("Intervention budget $k$ [edges]")
    ax.set_ylabel(r"Realized social-cost change $\Delta SC$ [\%]")
    ax.set_title(
        "Top-$k$ intervention validation (edgewise $\\beta$ reductions, lower is better)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax.text(
        0.99,
        0.02,
        f"Instances: {n_instances}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color="0.25",
    )

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK

    graph_families_raw = lpt._parse_list(args.graph_families)
    graph_families = []
    for gf in graph_families_raw:
        normalized = lpt.GRAPH_FAMILY_ALIASES.get(gf, gf)
        if normalized not in graph_families:
            graph_families.append(normalized)

    regime_names = lpt._parse_list(args.regimes)

    for family in graph_families:
        if family not in lpt.GRAPH_FAMILY_LABELS:
            raise ValueError(f"Unknown graph family '{family}'.")
    for regime_name in regime_names:
        if regime_name not in lpt.REGIMES:
            raise ValueError(f"Unknown regime '{regime_name}'.")

    detail_rows: list[dict] = []
    edge_rows: list[dict] = []
    correlation_rows: list[dict] = []

    instance_id = 0

    for graph_family in graph_families:
        for regime_name in regime_names:
            regime = lpt.REGIMES[regime_name]

            for trial_idx in range(args.num_trials):
                seed = args.seed + trial_idx * args.seed_step
                graph = lpt._build_graph(
                    graph_family=graph_family,
                    num_nodes=args.num_nodes,
                    seed=seed,
                )
                lpt._ensure_population_attribute(graph, seed=seed + 101)
                lpt._apply_beta_heterogeneity(
                    graph,
                    beta_cv=regime.beta_cv,
                    seed=seed + 303,
                    min_beta=args.min_beta,
                )

                demands = lpt._build_demands(graph, regime=regime)
                baseline = bp._build_baseline_system(
                    graph=graph,
                    demands=demands,
                    solver=solver,
                    eps_active=args.eps_active,
                    regularization=args.regularization,
                    disagg_reg=args.disagg_reg,
                )

                base_flow = np.asarray(baseline.total_flow, dtype=float)
                selected = bp._select_edge_indices(
                    baseline_edge_flow=base_flow,
                    strategy=args.edge_selection,
                    max_edges=args.max_candidates,
                    seed=seed,
                )
                selected = np.asarray(selected, dtype=int)
                if selected.size == 0:
                    continue

                sc_baseline = sc.total_social_cost(
                    baseline.graph,
                    baseline.total_flow,
                    alpha=baseline.alpha,
                    beta=baseline.beta,
                )
                sc_scale = max(abs(sc_baseline), 1e-12)

                dsc_dict = mcsc.derivative_social_cost(
                    graph,
                    baseline.flow_matrix,
                    demands,
                    eps=args.eps_active,
                    reg=args.regularization,
                )
                dsc = np.array(
                    [dsc_dict.get(edge, 0.0) for edge in baseline.edges], dtype=float
                )

                centrality_dict = nx.edge_betweenness_centrality(graph, normalized=True)
                betweenness = np.array(
                    [centrality_dict.get(edge, 0.0) for edge in baseline.edges],
                    dtype=float,
                )

                delta_cache: dict[tuple[int, ...], float] = {}

                def simulate(
                    edge_indices: np.ndarray | list[int] | tuple[int, ...],
                ) -> float:
                    key = _subset_key(edge_indices)
                    if key in delta_cache:
                        return float(delta_cache[key])

                    beta_trial = baseline.beta.copy()
                    if key:
                        idx = np.asarray(key, dtype=int)
                        beta_trial[idx] = np.maximum(
                            args.min_beta,
                            beta_trial[idx] * (1.0 - args.rel_step),
                        )

                    flow_trial = bp._solve_flows(
                        graph=baseline.graph,
                        demands=baseline.demands,
                        beta=beta_trial,
                        solver=solver,
                        disagg_reg=args.disagg_reg,
                    )
                    total_flow_trial = np.sum(
                        np.asarray(flow_trial, dtype=float), axis=0
                    )
                    sc_trial = sc.total_social_cost(
                        baseline.graph,
                        total_flow_trial,
                        alpha=baseline.alpha,
                        beta=beta_trial,
                    )
                    delta_pct = 100.0 * float((sc_trial - sc_baseline) / sc_scale)
                    delta_cache[key] = delta_pct
                    return delta_pct

                single_edge_delta = {}
                for edge_idx in selected:
                    edge_idx = int(edge_idx)
                    delta_pct = simulate([edge_idx])
                    single_edge_delta[edge_idx] = delta_pct

                    predicted_abs = dsc[edge_idx] * (
                        -args.rel_step * baseline.beta[edge_idx]
                    )
                    predicted_pct = 100.0 * float(predicted_abs / sc_scale)
                    edge_rows.append(
                        {
                            "instance_id": int(instance_id),
                            "graph_family": graph_family,
                            "graph_label": lpt.GRAPH_FAMILY_LABELS[graph_family],
                            "regime": regime_name,
                            "regime_label": regime.label,
                            "trial": int(trial_idx),
                            "seed": int(seed),
                            "edge_index": int(edge_idx),
                            "edge": bp._format_edge(baseline.edges[edge_idx]),
                            "beta": float(baseline.beta[edge_idx]),
                            "baseline_flow": float(base_flow[edge_idx]),
                            "betweenness": float(betweenness[edge_idx]),
                            "scgc": float(dsc[edge_idx]),
                            "predicted_single_delta_sc_pct": float(predicted_pct),
                            "observed_single_delta_sc_pct": float(delta_pct),
                        }
                    )

                oracle1_delta = float(np.min(list(single_edge_delta.values())))

                scores = _compute_method_scores(
                    baseline_beta=baseline.beta,
                    baseline_flow=base_flow,
                    dsc=dsc,
                    betweenness=betweenness,
                    rel_step=args.rel_step,
                    random_seed=seed + 707,
                )

                realized_vec = np.array(
                    [single_edge_delta[int(i)] for i in selected], dtype=float
                )
                for method in METHOD_ORDER:
                    corr = _safe_spearman(scores[method][selected], realized_vec)
                    correlation_rows.append(
                        {
                            "instance_id": int(instance_id),
                            "graph_family": graph_family,
                            "graph_label": lpt.GRAPH_FAMILY_LABELS[graph_family],
                            "regime": regime_name,
                            "regime_label": regime.label,
                            "trial": int(trial_idx),
                            "seed": int(seed),
                            "n_candidates": int(selected.size),
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "spearman": float(corr),
                        }
                    )

                max_k = min(int(args.max_budget), int(selected.size))
                budgets = list(range(1, max_k + 1))

                deterministic_orders = {
                    method: _rank_edges(scores[method], selected)
                    for method in METHOD_ORDER
                    if method != "random"
                }

                for method in METHOD_ORDER:
                    for budget in budgets:
                        if method == "random":
                            rng = np.random.default_rng(seed + 10_000 + budget)
                            trials = []
                            for _ in range(args.random_repeats):
                                choice = rng.permutation(selected)[:budget]
                                trials.append(simulate(choice))
                            delta_sc_pct = float(np.mean(trials))
                            delta_std_pct = float(
                                np.std(np.asarray(trials, dtype=float))
                            )
                        else:
                            chosen = deterministic_orders[method][:budget]
                            delta_sc_pct = float(simulate(chosen))
                            delta_std_pct = float("nan")

                        detail_rows.append(
                            {
                                "instance_id": int(instance_id),
                                "graph_family": graph_family,
                                "graph_label": lpt.GRAPH_FAMILY_LABELS[graph_family],
                                "regime": regime_name,
                                "regime_label": regime.label,
                                "trial": int(trial_idx),
                                "seed": int(seed),
                                "num_nodes": int(graph.number_of_nodes()),
                                "num_edges": int(graph.number_of_edges()),
                                "num_commodities": int(demands.shape[0]),
                                "n_candidates": int(selected.size),
                                "budget": int(budget),
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "rel_step_pct": float(100.0 * args.rel_step),
                                "delta_sc_pct": float(delta_sc_pct),
                                "random_repeat_std_pct": float(delta_std_pct),
                                "oracle1_delta_sc_pct": float(oracle1_delta),
                                "regret_to_oracle1_pct": (
                                    float(delta_sc_pct - oracle1_delta)
                                    if budget == 1
                                    else float("nan")
                                ),
                            }
                        )

                print(
                    f"Instance {instance_id:03d} | {lpt.GRAPH_FAMILY_LABELS[graph_family]:>18s}"
                    f" | {regime.label:>18s} | n_edges={graph.number_of_edges():3d}"
                    f" | n_candidates={selected.size:2d}"
                )
                instance_id += 1

    _annotate_best_flags(detail_rows)
    summary_rows = _build_summary_rows(detail_rows)
    corr_summary_rows = _build_correlation_summary(correlation_rows)

    _write_rows(args.output_table, detail_rows)
    _write_rows(args.output_edge_table, edge_rows)
    _write_rows(args.output_summary, summary_rows)

    _write_rows(args.output_correlation, correlation_rows)
    _write_rows(args.output_correlation_summary, corr_summary_rows)

    _plot_budget_curve(detail_rows, args.output_figure)

    print("")
    print(f"Saved detail table: {args.output_table}")
    print(f"Saved edge table: {args.output_edge_table}")
    print(f"Saved summary table: {args.output_summary}")
    print(f"Saved correlation table: {args.output_correlation}")
    print(f"Saved correlation summary: {args.output_correlation_summary}")
    print(f"Saved figure: {args.output_figure}")


if __name__ == "__main__":
    main()

# %%
