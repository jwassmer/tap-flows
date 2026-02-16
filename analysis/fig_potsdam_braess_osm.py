# %%
"""Potsdam OSM case study: edge utilization and social-cost gradient map."""

from __future__ import annotations

import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import osmnx as ox

from src import osmGraphs as og
from src.figure_style import (
    add_panel_label,
    apply_publication_style,
    make_scalar_mappable,
)
from src.paper_analysis import (
    PRIMARY_ROAD_FILTER,
    add_utilization_columns,
    braess_edge_subset,
    destination_population_demands,
    solve_city_multicommodity_case,
)

PLACE = "Potsdam, Germany"
OUTPUT_FIGURE = "figs/potsdam-germany-braess.pdf"
OUTPUT_EDGES = "figs/potsdam-germany-braess.geojson"


def main() -> None:
    apply_publication_style(font_size=16)

    graph, boundary = og.osmGraph(
        PLACE,
        highway_filter=PRIMARY_ROAD_FILTER,
        return_boundary=True,
        tolerance_meters=100,
    )
    nodes, _ = ox.graph_to_gdfs(graph)

    destination_nodes = og.select_evenly_distributed_nodes(nodes, len(nodes))
    od_matrix = destination_population_demands(
        nodes,
        destination_nodes,
        gamma=0.03,
        demand_column="population",
    )

    result = solve_city_multicommodity_case(
        graph,
        od_matrix,
        eps=1e-3,
        demands_to_sinks=False,
        solver=cp.OSQP,
        verbose=True,
        max_iter=20_000,
    )

    edges = add_utilization_columns(result.edges, utilization_column="utilization")
    braess_edges = braess_edge_subset(edges)

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
    utilization_norm = mpl.colors.Normalize(
        vmin=0.2, vmax=float(edges["utilization"].max())
    )

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
    gradient_norm = mpl.colors.Normalize(
        vmin=0,
        vmax=max(float(edges["derivative_social_cost"].max()), 1e-9),
    )

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

    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    edges.to_file(OUTPUT_EDGES, driver="GeoJSON")


if __name__ == "__main__":
    main()

# %%
