# %%
"""Active-set stability scan under edge-wise free-flow time perturbations.

Defaults are set for a larger synthetic planar network and perturbations in
the range [-5%, +5%].
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from src import Graphs as gr
from src import multiCommodityTAP as mc
from src.figure_style import add_panel_label, apply_publication_style
from src.paper_examples import build_classic_braess_validation_graph

OUTPUT_FIGURE = "figs/active-set-stability-synthetic.pdf"
OUTPUT_TABLE = "cache/active-set-stability-synthetic.csv"


def _format_edge(edge: tuple) -> str:
    return f"{edge[0]}→{edge[1]}"


def _active_mask(flow_matrix: npt.ArrayLike, eps_active: float) -> np.ndarray:
    flows = np.asarray(flow_matrix, dtype=float)
    if flows.ndim == 1:
        flows = flows.reshape(1, -1)
    return flows > eps_active


def _jaccard_similarity(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(mask_a & mask_b)
    return float(intersection / union)


def _solve_flows(
    graph,
    demands: np.ndarray,
    beta: np.ndarray,
    solver,
) -> np.ndarray:
    flows, _ = mc.solve_multicommodity_tap(
        graph,
        demands,
        beta=beta,
        pos_flows=True,
        return_fw=True,
        solver=solver,
    )
    return np.asarray(flows, dtype=float)


def _local_stability_radius(delta_values: np.ndarray, stable_flags: np.ndarray) -> float:
    """Return maximal symmetric radius (discrete grid) around delta=0 that stays stable."""
    deltas = np.asarray(delta_values, dtype=float)
    stable = np.asarray(stable_flags, dtype=bool)

    zero_idx = int(np.argmin(np.abs(deltas)))
    if not stable[zero_idx]:
        return 0.0

    left = zero_idx
    while left - 1 >= 0 and stable[left - 1]:
        left -= 1

    right = zero_idx
    while right + 1 < stable.size and stable[right + 1]:
        right += 1

    return float(min(abs(deltas[left]), abs(deltas[right])))


def _select_edge_indices(
    baseline_edge_flow: np.ndarray,
    strategy: str,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    num_edges = baseline_edge_flow.size
    all_indices = np.arange(num_edges, dtype=int)

    if strategy == "all" or max_edges >= num_edges:
        return all_indices

    if strategy == "top-flow":
        order = np.argsort(-baseline_edge_flow)
        return np.sort(order[:max_edges])

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(all_indices, size=max_edges, replace=False))

    raise ValueError(f"Unknown edge-selection strategy '{strategy}'.")


def _run_scan(
    graph,
    demands: np.ndarray,
    delta_values: np.ndarray,
    eps_active: float,
    min_beta: float,
    solver,
    edge_selection: str,
    max_edges: int,
    seed: int,
):
    edges = list(graph.edges)
    beta_base = np.array([graph.edges[edge]["beta"] for edge in edges], dtype=float)

    baseline_flows = _solve_flows(graph, demands, beta_base, solver)
    baseline_mask = _active_mask(baseline_flows, eps_active)
    baseline_edge_flow = np.sum(baseline_flows, axis=0)

    selected_indices = _select_edge_indices(
        baseline_edge_flow=baseline_edge_flow,
        strategy=edge_selection,
        max_edges=max_edges,
        seed=seed,
    )

    num_selected = selected_indices.size
    num_deltas = len(delta_values)
    stable_flags = np.zeros((num_selected, num_deltas), dtype=bool)
    changed_entries = np.zeros((num_selected, num_deltas), dtype=int)
    jaccard = np.zeros((num_selected, num_deltas), dtype=float)

    for local_idx, edge_idx in enumerate(selected_indices):
        edge = edges[edge_idx]
        print(
            f"Scanning edge {_format_edge(edge)} "
            f"({local_idx + 1}/{num_selected})"
        )

        for delta_idx, delta in enumerate(delta_values):
            beta_trial = beta_base.copy()
            beta_trial[edge_idx] = max(min_beta, beta_base[edge_idx] * (1.0 + delta))

            trial_flows = _solve_flows(graph, demands, beta_trial, solver)
            trial_mask = _active_mask(trial_flows, eps_active)

            stable = np.array_equal(trial_mask, baseline_mask)
            stable_flags[local_idx, delta_idx] = stable
            changed_entries[local_idx, delta_idx] = int(
                np.count_nonzero(trial_mask ^ baseline_mask)
            )
            jaccard[local_idx, delta_idx] = _jaccard_similarity(trial_mask, baseline_mask)

    return (
        edges,
        selected_indices,
        baseline_mask,
        baseline_edge_flow,
        stable_flags,
        changed_entries,
        jaccard,
    )


def _write_summary_table(
    output_path: str,
    edges: list[tuple],
    selected_indices: np.ndarray,
    baseline_edge_flow: np.ndarray,
    delta_values: np.ndarray,
    stable_flags: np.ndarray,
    changed_entries: np.ndarray,
    baseline_mask: np.ndarray,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline_active_entries = int(np.count_nonzero(baseline_mask))
    radii = np.array(
        [_local_stability_radius(delta_values, row) for row in stable_flags],
        dtype=float,
    )

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "edge_index",
                "edge",
                "baseline_edge_flow",
                "baseline_active_entries",
                "local_stability_radius_pct",
                "stable_fraction_over_scan",
                "max_changed_active_entries",
            ]
        )

        for local_idx, edge_idx in enumerate(selected_indices):
            edge = edges[edge_idx]
            writer.writerow(
                [
                    int(edge_idx),
                    _format_edge(edge),
                    float(baseline_edge_flow[edge_idx]),
                    baseline_active_entries,
                    100.0 * radii[local_idx],
                    float(np.mean(stable_flags[local_idx])),
                    int(np.max(changed_entries[local_idx])),
                ]
            )


def _plot_results(
    output_figure: str,
    edges: list[tuple],
    selected_indices: np.ndarray,
    delta_values: np.ndarray,
    stable_flags: np.ndarray,
    changed_entries: np.ndarray,
    jaccard: np.ndarray,
) -> None:
    path = Path(output_figure)
    path.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=16)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.8, 6.2),
        gridspec_kw={"height_ratios": [1.25, 1.9]},
    )
    ax_heatmap, ax_summary = axes

    add_panel_label(ax_heatmap, r"\textbf{a}", x=0.03, y=1.12, fontsize=20)
    add_panel_label(ax_summary, r"\textbf{b}", x=0.03, y=1.04, fontsize=20)

    vmax = max(int(np.max(changed_entries)), 1)
    heatmap = ax_heatmap.imshow(
        changed_entries,
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=0,
        vmax=vmax,
    )

    tick_idx = np.linspace(0, len(delta_values) - 1, 5, dtype=int)
    tick_labels = [f"{100.0 * delta_values[i]:.1f}" for i in tick_idx]
    ax_heatmap.set_xticks(tick_idx, tick_labels)

    selected_labels = [_format_edge(edges[i]) for i in selected_indices]
    if len(selected_labels) > 8:
        y_step = max(1, int(np.ceil(len(selected_labels) / 8)))
        y_ticks = np.arange(0, len(selected_labels), y_step, dtype=int)
        ax_heatmap.set_yticks(y_ticks, [selected_labels[i] for i in y_ticks])
    else:
        ax_heatmap.set_yticks(np.arange(len(selected_labels)), selected_labels)

    ax_heatmap.set_xlabel(r"Relative perturbation $\beta_k$ [\%]")
    ax_heatmap.set_ylabel("Perturbed edge")
    ax_heatmap.set_title("Deviation from baseline")

    colorbar = fig.colorbar(
        heatmap,
        ax=ax_heatmap,
        orientation="vertical",
        pad=0.02,
        shrink=0.96,
        fraction=0.075,
    )
    colorbar.set_label("Changed active-set entries")

    delta_percent = 100.0 * delta_values
    stable_fraction = np.mean(stable_flags, axis=0)
    mean_jaccard = np.mean(jaccard, axis=0)

    ax_summary.plot(
        delta_percent,
        stable_fraction,
        color="black",
        linewidth=2.5,
        label="Stable fraction",
    )
    ax_summary.plot(
        delta_percent,
        mean_jaccard,
        color="#0571b0",
        linewidth=2.5,
        linestyle="--",
        label="Mean Jaccard similarity",
    )
    ax_summary.axvline(0.0, color="grey", linestyle=":", linewidth=1.2)
    ax_summary.set_ylim(-0.02, 1.02)
    ax_summary.set_xlabel(r"Relative perturbation of edge $\beta_k$ [\%]")
    ax_summary.set_ylabel("Stability score")
    ax_summary.set_title("Aggregate active-set stability")
    ax_summary.grid(alpha=0.3)
    ax_summary.legend(loc="lower left")

    radii = np.array(
        [_local_stability_radius(delta_values, row) for row in stable_flags],
        dtype=float,
    )
    ax_summary.text(
        0.98,
        0.95,
        f"Median local radius: {100.0 * np.median(radii):.2f}%",
        transform=ax_summary.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_problem(args: argparse.Namespace):
    if args.network == "braess":
        graph = build_classic_braess_validation_graph()
        load = 9.5
        demand = -np.ones(graph.number_of_nodes()) * load / (graph.number_of_nodes() - 1)
        demand[0] = load
        demands = np.asarray([demand], dtype=float)
        return graph, demands

    graph = gr.random_planar_graph(
        n_nodes=args.num_nodes,
        seed=args.seed,
        alpha="random",
    )
    demands = _synthetic_population_demands(
        graph=graph,
        num_commodities=args.num_commodities,
        gamma=args.gamma,
    )
    return graph, demands


def _synthetic_population_demands(
    graph,
    num_commodities: int,
    gamma: float,
) -> np.ndarray:
    """Create source-to-all OD demands from node populations without GeoPandas."""
    node_ids = np.array(list(graph.nodes))
    num_nodes = node_ids.size
    if num_commodities > num_nodes:
        raise ValueError(
            f"num_commodities ({num_commodities}) must be <= number of nodes ({num_nodes})."
        )

    population_values = [graph.nodes[node].get("population", 1.0) for node in node_ids]
    populations = np.array(
        [float(pop if pop is not None and pop > 0 else 1.0) for pop in population_values],
        dtype=float,
    )

    source_positions = np.linspace(0, num_nodes - 1, num_commodities, dtype=int)
    demands = np.zeros((num_commodities, num_nodes), dtype=float)

    for commodity_idx, source_idx in enumerate(source_positions):
        load = max(float(populations[source_idx] * gamma), 1e-9)
        demand = np.full(num_nodes, -load / (num_nodes - 1), dtype=float)
        demand[source_idx] = load
        demands[commodity_idx] = demand

    return demands


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse active-set stability under beta perturbations."
    )
    parser.add_argument("--network", choices=["synthetic", "braess"], default="synthetic")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-commodities", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--edge-selection", choices=["top-flow", "random", "all"], default="top-flow")
    parser.add_argument("--max-edges", type=int, default=40)

    parser.add_argument("--delta-min", type=float, default=-0.05)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)

    args, unknown = parser.parse_known_args()

    # Jupyter/IPython can inject kernel args (for example: --f=... or -f ...).
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

    if args.num_points < 3:
        parser.error("--num-points must be at least 3.")
    if args.max_edges < 1:
        parser.error("--max-edges must be at least 1.")
    if args.delta_min > args.delta_max:
        parser.error("--delta-min must be <= --delta-max.")
    if args.num_nodes < 5:
        parser.error("--num-nodes must be at least 5.")
    if args.num_commodities < 1:
        parser.error("--num-commodities must be at least 1.")

    return args


def main() -> None:
    args = _parse_args()

    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK
    delta_values = np.linspace(args.delta_min, args.delta_max, args.num_points)
    graph, demands = _build_problem(args)

    (
        edges,
        selected_indices,
        baseline_mask,
        baseline_edge_flow,
        stable_flags,
        changed_entries,
        jaccard,
    ) = _run_scan(
        graph=graph,
        demands=demands,
        delta_values=delta_values,
        eps_active=args.eps_active,
        min_beta=args.min_beta,
        solver=solver,
        edge_selection=args.edge_selection,
        max_edges=args.max_edges,
        seed=args.seed,
    )

    _write_summary_table(
        output_path=args.output_table,
        edges=edges,
        selected_indices=selected_indices,
        baseline_edge_flow=baseline_edge_flow,
        delta_values=delta_values,
        stable_flags=stable_flags,
        changed_entries=changed_entries,
        baseline_mask=baseline_mask,
    )
    _plot_results(
        output_figure=args.output_figure,
        edges=edges,
        selected_indices=selected_indices,
        delta_values=delta_values,
        stable_flags=stable_flags,
        changed_entries=changed_entries,
        jaccard=jaccard,
    )

    print(f"Network: {args.network}")
    print(f"Selected edges: {len(selected_indices)} / {len(edges)}")
    print(f"Perturbation range: [{100.0 * args.delta_min:.2f}%, {100.0 * args.delta_max:.2f}%]")
    print(f"Saved figure to {args.output_figure}")
    print(f"Saved summary table to {args.output_table}")


if __name__ == "__main__":
    main()

# %%
