# %%
"""Cologne stadium-traffic case study on an OSM road network."""

from __future__ import annotations

import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

from src import osmGraphs as og
from src.figure_style import (
    add_panel_label,
    apply_publication_style,
    make_scalar_mappable,
)
from src.paper_analysis import (
    PRIMARY_ROAD_FILTER,
    braess_edge_subset,
    nearest_graph_nodes,
    solve_city_multicommodity_case,
    stadium_vehicle_demands,
)

PLACE = "Cologne,Germany"
PARKING_ADDRESSES = [
    "Salzburger Weg, 50858 Köln",
    "Wendelinstr., 50933 Köln",
    "Brauweiler Weg, 50933 Köln",
]

OUTPUT_FIGURE = "figs/cologne-stadium-traffic.pdf"
OUTPUT_EDGES = "figs/cologne-stadium-traffic-edges.geojson"


def main() -> None:
    apply_publication_style(font_size=16)

    graph, boundary = og.osmGraph(
        PLACE,
        return_boundary=True,
        tolerance_meters=100,
        highway_filter=PRIMARY_ROAD_FILTER,
    )

    nodes, _ = ox.graph_to_gdfs(graph)

    sink_node_ids = nearest_graph_nodes(graph, PARKING_ADDRESSES)

    parking_lots = 7_500
    total_vehicles = parking_lots * 2
    od_matrix = stadium_vehicle_demands(
        nodes, sink_node_ids, total_vehicles=total_vehicles
    )

    result = solve_city_multicommodity_case(
        graph,
        od_matrix,
        eps=1e-3,
        demands_to_sinks=False,
        solver=cp.OSQP,
        verbose=True,
    )
    edges = result.edges
    braess_edges = braess_edge_subset(edges)
    stadium_nodes = result.nodes.loc[sink_node_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)

    for ax in axes:
        boundary.boundary.plot(linewidth=1, ax=ax, color="black", zorder=2)
        result.nodes.plot(
            ax=ax,
            marker=".",
            color="black",
            markersize=5,
            zorder=4,
            label="Source nodes",
        )
        ax.grid()

    add_panel_label(axes[0], r"$\textbf{a}$")
    flow_cmap = plt.get_cmap("viridis")
    flow_cmap.set_under("lightgrey")

    flow_values = edges["flow"].to_numpy(dtype=float)
    top_flows = np.sort(flow_values)[-max(5, min(30, flow_values.size)) :]
    flow_norm = mpl.colors.LogNorm(vmin=100, vmax=max(float(np.mean(top_flows)), 100.0))

    edges.sort_values("flow", ascending=True).plot(
        ax=axes[0],
        linewidth=2,
        column="flow",
        cmap=flow_cmap,
        norm=flow_norm,
        legend=False,
    )

    stadium_nodes.plot(
        ax=axes[0],
        marker="s",
        color="black",
        markersize=100,
        zorder=5,
    )

    flow_bar = fig.colorbar(
        make_scalar_mappable(flow_cmap, flow_norm),
        ax=axes[0],
        orientation="horizontal",
        pad=0.01,
        shrink=2 / 3,
        extend="both",
    )
    flow_bar.ax.set_xlabel(r"Vehicle flow $f_e$")

    add_panel_label(axes[1], r"$\textbf{b}$")
    gradient_cmap = plt.get_cmap("Reds")
    gradient_cmap.set_under("#0571b0")

    gradient_max = max(float(edges["derivative_social_cost"].max()), 1e-9)
    gradient_norm = mpl.colors.Normalize(vmin=0, vmax=gradient_max)

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

    stadium_nodes.plot(
        ax=axes[1],
        marker="s",
        color="black",
        markersize=100,
        zorder=5,
        label="Parking lots near football stadium",
    )
    axes[1].legend(loc="upper right")

    gradient_bar = fig.colorbar(
        make_scalar_mappable(gradient_cmap, gradient_norm),
        ax=axes[1],
        orientation="horizontal",
        pad=0.01,
        shrink=2 / 3,
        extend="both",
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
