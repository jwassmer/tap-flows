# %%
"""Potsdam OSM case study: edge utilization, SCGC map, and intervention checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cvxpy as cp
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

from src import osmGraphs as og
from src.figure_style import add_panel_label, apply_publication_style, make_scalar_mappable
from src.paper_analysis import (
    PRIMARY_ROAD_FILTER,
    add_utilization_columns,
    braess_edge_subset,
    destination_population_demands,
    run_intervention_benchmark,
    solve_city_multicommodity_case,
    summarize_edge_metrics,
)

PLACE = "Potsdam, Germany"
OUTPUT_FIGURE = "figs/potsdam-germany-braess.pdf"
OUTPUT_EDGES = "figs/potsdam-germany-braess.geojson"
OUTPUT_SUMMARY = "cache/potsdam-traffic-summary.csv"
OUTPUT_INTERVENTIONS = "cache/potsdam-traffic-interventions.csv"
OUTPUT_INTERVENTION_SUMMARY = "cache/potsdam-traffic-intervention-summary.csv"
OUTPUT_EDGE_EFFECTS = "cache/potsdam-traffic-edge-effects.csv"
OUTPUT_RANK_CORRELATION = "cache/potsdam-traffic-rank-correlation.csv"
OUTPUT_INTERVENTION_FIGURE = "figs/potsdam-intervention-validation.pdf"
OUTPUT_NODES = "cache/potsdam-traffic-nodes.geojson"
OUTPUT_BOUNDARY = "cache/potsdam-traffic-boundary.geojson"
OUTPUT_DESTINATION_NODES = "cache/potsdam-traffic-destination-nodes.geojson"


def _parse_list(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Potsdam daily-traffic SCGC mapping and intervention validation.",
    )
    parser.add_argument("--place", default=PLACE)
    parser.add_argument("--gamma", type=float, default=0.03)
    parser.add_argument(
        "--demand-multipliers",
        default="0.9,1.0,1.1",
        help="Comma-separated OD scaling factors for robustness analysis.",
    )
    parser.add_argument(
        "--intervention-rel-step",
        type=float,
        default=0.01,
        help="Relative beta perturbation for intervention validation.",
    )
    parser.add_argument("--intervention-max-candidates", type=int, default=25)
    parser.add_argument("--intervention-max-budget", type=int, default=6)
    parser.add_argument(
        "--intervention-candidate-mode",
        choices=["top-flow", "top-abs-scgc", "random", "all"],
        default="top-flow",
    )
    parser.add_argument("--intervention-random-repeats", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=20_000)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-edges", default=OUTPUT_EDGES)
    parser.add_argument("--output-summary", default=OUTPUT_SUMMARY)
    parser.add_argument("--output-interventions", default=OUTPUT_INTERVENTIONS)
    parser.add_argument(
        "--output-intervention-summary", default=OUTPUT_INTERVENTION_SUMMARY
    )
    parser.add_argument("--output-edge-effects", default=OUTPUT_EDGE_EFFECTS)
    parser.add_argument("--output-rank-correlation", default=OUTPUT_RANK_CORRELATION)
    parser.add_argument(
        "--output-intervention-figure", default=OUTPUT_INTERVENTION_FIGURE
    )
    parser.add_argument("--output-nodes", default=OUTPUT_NODES)
    parser.add_argument("--output-boundary", default=OUTPUT_BOUNDARY)
    parser.add_argument("--output-destination-nodes", default=OUTPUT_DESTINATION_NODES)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from cached outputs without solving TAP.",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Solve and export only the baseline map/caches, skip sweeps.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast run for quick checks.",
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

    if args.smoke:
        args.demand_multipliers = "1.0"
        args.intervention_max_candidates = min(args.intervention_max_candidates, 15)
        args.intervention_max_budget = min(args.intervention_max_budget, 4)
        args.intervention_random_repeats = min(args.intervention_random_repeats, 6)
        args.max_iter = min(args.max_iter, 8_000)

    if args.plot_only and args.map_only:
        parser.error("--plot-only and --map-only are mutually exclusive.")
    if args.gamma <= 0:
        parser.error("--gamma must be positive.")
    if args.intervention_rel_step <= 0 or args.intervention_rel_step >= 1:
        parser.error("--intervention-rel-step must be in (0, 1).")
    if args.intervention_max_candidates < 1:
        parser.error("--intervention-max-candidates must be >= 1.")
    if args.intervention_max_budget < 1:
        parser.error("--intervention-max-budget must be >= 1.")
    if args.intervention_random_repeats < 1:
        parser.error("--intervention-random-repeats must be >= 1.")

    return args


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _plot_intervention_validation(interventions: pd.DataFrame, output_figure: str) -> None:
    if interventions.empty:
        return

    _ensure_parent(output_figure)
    apply_publication_style(font_size=20)
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.4))

    method_order = ["scgc_linear", "flow", "betweenness", "random"]
    labels = {
        "scgc_linear": "SCGC linear",
        "flow": "High flow",
        "betweenness": "Betweenness",
        "random": "Random",
    }
    colors = {
        "scgc_linear": "#1f78b4",
        "flow": "#ff7f00",
        "betweenness": "#33a02c",
        "random": "#6a3d9a",
    }

    for method in method_order:
        subset = interventions[interventions["method"] == method]
        if subset.empty:
            continue
        grouped = (
            subset.groupby("budget", as_index=False)["delta_sc_pct"]
            .agg(
                median="median",
                q25=lambda x: np.percentile(x, 25),
                q75=lambda x: np.percentile(x, 75),
            )
            .sort_values("budget")
        )
        x = grouped["budget"].to_numpy(dtype=float)
        y = grouped["median"].to_numpy(dtype=float)
        q25 = grouped["q25"].to_numpy(dtype=float)
        q75 = grouped["q75"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=colors[method],
            label=labels[method],
        )
        ax.fill_between(x, q25, q75, color=colors[method], alpha=0.15)

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Intervention budget $k$ [edges]")
    ax.set_ylabel(r"Realized $\Delta SC$ [\%]")
    ax.set_title(r"Potsdam: stress test (+$\Delta\beta_e$) on candidate edges")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _map_figure(result, boundary: gpd.GeoDataFrame, destination_nodes: gpd.GeoDataFrame, output_figure: str) -> None:
    edges = add_utilization_columns(result.edges, utilization_column="utilization")
    braess_edges = braess_edge_subset(edges)

    _ensure_parent(output_figure)
    # Keep old committed visual style for the main Potsdam figure.
    apply_publication_style(font_size=16)
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)

    for ax in axes:
        boundary.boundary.plot(linewidth=1, ax=ax, color="black", zorder=2)
        destination_nodes.plot(
            ax=ax,
            marker=".",
            zorder=4,
            color="black",
            markersize=25,
            label="Sources and destinations",
        )
        ax.grid()

    add_panel_label(axes[0], r"$\textbf{a}$")
    utilization_cmap = plt.get_cmap("cividis")
    utilization_cmap.set_under("lightgrey")
    utilization_norm = mpl.colors.Normalize(vmin=0.2, vmax=float(edges["utilization"].max()))

    edges.sort_values("utilization", ascending=True).plot(
        ax=axes[0],
        linewidth=2,
        column="utilization",
        cmap=utilization_cmap,
        norm=utilization_norm,
        legend=False,
    )

    utilization_bar = fig.colorbar(
        make_scalar_mappable(utilization_cmap, utilization_norm),
        ax=axes[0],
        orientation="horizontal",
        pad=0.01,
        shrink=2 / 3,
        extend="min",
    )
    utilization_bar.ax.set_xlabel(r"Utilization $1-\beta_e/(\alpha_e f_e + \beta_e)$")

    add_panel_label(axes[1], r"$\textbf{b}$")
    gradient_cmap = plt.get_cmap("Reds")
    gradient_cmap.set_under("#0571b0")
    gradient_norm = mpl.colors.Normalize(vmin=0, vmax=max(float(edges["derivative_social_cost"].max()), 1e-9))

    edges.sort_values("derivative_social_cost", ascending=True).plot(
        ax=axes[1],
        linewidth=2,
        column="derivative_social_cost",
        cmap=gradient_cmap,
        norm=gradient_norm,
    )

    if len(braess_edges) > 0:
        braess_edges.plot(
            ax=axes[1],
            linewidth=2,
            color="#0571b0",
            label="Braess edges",
        )
        axes[1].legend(loc="upper right")

    gradient_bar = fig.colorbar(
        make_scalar_mappable(gradient_cmap, gradient_norm),
        ax=axes[1],
        orientation="horizontal",
        pad=0.01,
        shrink=2 / 3,
        extend="min",
    )
    gradient_bar.ax.set_xlabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$")

    axes[0].set_xlabel("Lon [°]")
    axes[1].set_xlabel("Lon [°]")
    axes[0].set_ylabel("Lat [°]")

    fig.savefig(output_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    multipliers = _parse_list(args.demand_multipliers)

    if args.plot_only:
        edges_path = Path(args.output_edges)
        nodes_path = Path(args.output_nodes)
        boundary_path = Path(args.output_boundary)
        destination_path = Path(args.output_destination_nodes)

        required = [edges_path, nodes_path, boundary_path, destination_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing cached map files for --plot-only: "
                f"{', '.join(missing)}. Run once without --plot-only."
            )

        edges = gpd.read_file(edges_path)
        nodes = gpd.read_file(nodes_path)
        boundary = gpd.read_file(boundary_path)
        destination_nodes = gpd.read_file(destination_path)

        cached_result = SimpleNamespace(edges=edges, nodes=nodes)
        _map_figure(cached_result, boundary, destination_nodes, args.output_figure)

        interventions_df = _read_csv_if_exists(args.output_interventions)
        _plot_intervention_validation(interventions_df, args.output_intervention_figure)

        print(f"Regenerated map figure from cache: {args.output_figure}")
        print(f"Regenerated intervention figure from cache: {args.output_intervention_figure}")
        return

    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK

    graph, boundary = og.osmGraph(
        args.place,
        highway_filter=PRIMARY_ROAD_FILTER,
        return_boundary=True,
        tolerance_meters=100,
    )
    nodes, _ = ox.graph_to_gdfs(graph)

    destination_nodes = og.select_evenly_distributed_nodes(nodes, len(nodes))
    base_od = destination_population_demands(
        nodes,
        destination_nodes,
        gamma=args.gamma,
        demand_column="population",
    )

    baseline_result = solve_city_multicommodity_case(
        graph,
        base_od,
        eps=args.eps,
        demands_to_sinks=False,
        solver=solver,
        verbose=not args.smoke,
        max_iter=args.max_iter,
    )

    _map_figure(baseline_result, boundary, destination_nodes, args.output_figure)

    edges_with_util = add_utilization_columns(baseline_result.edges, utilization_column="utilization")

    for path in [
        args.output_edges,
        args.output_nodes,
        args.output_boundary,
        args.output_destination_nodes,
    ]:
        _ensure_parent(path)

    edges_with_util.to_file(args.output_edges, driver="GeoJSON")
    baseline_result.nodes.to_file(args.output_nodes, driver="GeoJSON")
    boundary.to_file(args.output_boundary, driver="GeoJSON")
    geom_col = destination_nodes.geometry.name
    destination_export = gpd.GeoDataFrame(
        destination_nodes[[geom_col]].copy(),
        geometry=geom_col,
        crs=destination_nodes.crs,
    )
    destination_export.to_file(args.output_destination_nodes, driver="GeoJSON")

    if args.map_only:
        print(f"Saved map figure: {args.output_figure}")
        print(f"Saved edge GeoJSON: {args.output_edges}")
        print(f"Saved nodes GeoJSON: {args.output_nodes}")
        print(f"Saved boundary GeoJSON: {args.output_boundary}")
        print(f"Saved destination nodes GeoJSON: {args.output_destination_nodes}")
        return

    summary_rows = []
    intervention_tables = []
    intervention_summary_tables = []
    edge_effect_tables = []
    corr_tables = []

    for idx, multiplier in enumerate(multipliers):
        od_scaled = base_od * float(multiplier)
        result = solve_city_multicommodity_case(
            graph,
            od_scaled,
            eps=args.eps,
            demands_to_sinks=False,
            solver=solver,
            verbose=False,
            max_iter=args.max_iter,
        )
        edges = add_utilization_columns(result.edges, utilization_column="utilization")

        metrics = summarize_edge_metrics(edges, utilization_column="utilization")
        metrics.update(
            {
                "city": "Potsdam",
                "scenario": "daily-traffic",
                "demand_multiplier": float(multiplier),
                "num_nodes": float(result.graph.number_of_nodes()),
                "num_commodities": float(od_scaled.shape[0]),
            }
        )
        summary_rows.append(metrics)

        bench = run_intervention_benchmark(
            result.graph,
            od_scaled,
            edges,
            rel_step=args.intervention_rel_step,
            direction=+1,
            target="increase",
            max_budget=args.intervention_max_budget,
            max_candidates=args.intervention_max_candidates,
            candidate_mode=args.intervention_candidate_mode,
            random_repeats=args.intervention_random_repeats,
            random_seed=args.seed + idx,
            min_beta=args.min_beta,
            solver=solver,
            max_iter=args.max_iter,
        )

        for table in [
            bench.interventions,
            bench.summary,
            bench.edge_effects,
            bench.rank_correlation,
        ]:
            if table.empty:
                continue
            table["city"] = "Potsdam"
            table["scenario"] = "daily-traffic"
            table["demand_multiplier"] = float(multiplier)

        if not bench.interventions.empty:
            intervention_tables.append(bench.interventions)
        if not bench.summary.empty:
            intervention_summary_tables.append(bench.summary)
        if not bench.edge_effects.empty:
            edge_effect_tables.append(bench.edge_effects)
        if not bench.rank_correlation.empty:
            corr_tables.append(bench.rank_correlation)

        print(
            f"demand x{multiplier:.2f} | edges={int(metrics['num_edges']):4d} | "
            f"braess={int(metrics['num_braess_edges']):3d} ({metrics['braess_share_pct']:.2f}%) | "
            f"SCGC[min,max]=({metrics['scgc_min']:.2f},{metrics['scgc_max']:.2f})"
        )

    summary_df = pd.DataFrame(summary_rows)
    interventions_df = pd.concat(intervention_tables, ignore_index=True) if intervention_tables else pd.DataFrame()
    intervention_summary_df = pd.concat(intervention_summary_tables, ignore_index=True) if intervention_summary_tables else pd.DataFrame()
    edge_effect_df = pd.concat(edge_effect_tables, ignore_index=True) if edge_effect_tables else pd.DataFrame()
    corr_df = pd.concat(corr_tables, ignore_index=True) if corr_tables else pd.DataFrame()

    for path in [
        args.output_summary,
        args.output_interventions,
        args.output_intervention_summary,
        args.output_edge_effects,
        args.output_rank_correlation,
    ]:
        _ensure_parent(path)

    summary_df.to_csv(args.output_summary, index=False)
    interventions_df.to_csv(args.output_interventions, index=False)
    intervention_summary_df.to_csv(args.output_intervention_summary, index=False)
    edge_effect_df.to_csv(args.output_edge_effects, index=False)
    corr_df.to_csv(args.output_rank_correlation, index=False)

    _plot_intervention_validation(interventions_df, args.output_intervention_figure)

    print(f"Saved map figure: {args.output_figure}")
    print(f"Saved edge GeoJSON: {args.output_edges}")
    print(f"Saved nodes GeoJSON: {args.output_nodes}")
    print(f"Saved boundary GeoJSON: {args.output_boundary}")
    print(f"Saved destination nodes GeoJSON: {args.output_destination_nodes}")
    print(f"Saved scenario summary: {args.output_summary}")
    print(f"Saved intervention detail: {args.output_interventions}")
    print(f"Saved intervention summary: {args.output_intervention_summary}")
    print(f"Saved edge effects: {args.output_edge_effects}")
    print(f"Saved rank correlation: {args.output_rank_correlation}")
    print(f"Saved intervention figure: {args.output_intervention_figure}")


if __name__ == "__main__":
    main()

# %%
