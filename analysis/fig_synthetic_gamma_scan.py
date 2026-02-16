# %%
"""Synthetic planar-graph sensitivity analysis across demand scaling gamma."""

from __future__ import annotations

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox

from src import Graphs as gr
from src.figure_style import (
    add_panel_label,
    apply_publication_style,
    make_scalar_mappable,
)

OUTPUT_FIGURE = "figs/synthetic-gamma-scan.pdf"


def _scan_gamma_values(
    graph: nx.MultiDiGraph, gamma_values: list[float]
) -> tuple[dict, dict]:
    flow_by_gamma = {}
    gradient_by_gamma = {}

    for gamma in gamma_values:
        print(f"Computing flows and SCGC for gamma={gamma:.2f}")
        graph.flows(num_sources="all", gamma=gamma, solver=cp.OSQP)
        graph.derivative_social_cost(
            num_sources="all", gamma=gamma, eps=1e-3, solver=cp.OSQP
        )

        flow = np.array(
            list(nx.get_edge_attributes(graph, "flow").values()), dtype=float
        )
        dsc = np.array(
            list(nx.get_edge_attributes(graph, "derivative_social_cost").values()),
            dtype=float,
        )
        flow_by_gamma[gamma] = flow
        gradient_by_gamma[gamma] = dsc

    return flow_by_gamma, gradient_by_gamma


def main() -> None:
    apply_publication_style(font_size=14)

    graph = gr.random_planar_graph(50, seed=42)
    _, edges = ox.graph_to_gdfs(graph)

    edges = edges.to_crs(epsg=3857)
    edges["length"] = edges.geometry.length
    edges = edges.to_crs(epsg=4326)

    gamma_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    flow_by_gamma, gradient_by_gamma = _scan_gamma_values(graph, gamma_values)

    selected_gamma = 0.07
    edges["flow"] = flow_by_gamma[selected_gamma]
    edges["derivative_social_cost"] = gradient_by_gamma[selected_gamma]

    braess_edges = edges[edges["derivative_social_cost"] < 0]
    print("Length of Braess edges:", float(np.sum(braess_edges["length"])))

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])

    ax_flow = fig.add_subplot(gs[0, 0])
    ax_gradient = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 2])

    for ax in (ax_flow, ax_gradient):
        ax.axis("off")

    add_panel_label(ax_flow, r"\textbf{a}", x=0.05, y=0.95, fontsize=16)
    add_panel_label(ax_gradient, r"\textbf{b}", x=0.05, y=0.95, fontsize=16)
    add_panel_label(ax_scatter, r"\textbf{c}", x=0.05, y=0.95, fontsize=16)

    flow_cmap = plt.get_cmap("viridis")
    flow_cmap.set_under("lightgrey")
    flow_norm = plt.Normalize(vmin=0, vmax=float(edges["flow"].max()))

    edges.sort_values("flow", ascending=True).plot(
        ax=ax_flow,
        column="flow",
        legend=False,
        cmap=flow_cmap,
        norm=flow_norm,
    )
    flow_bar = fig.colorbar(
        make_scalar_mappable(flow_cmap, flow_norm),
        ax=ax_flow,
        orientation="horizontal",
        pad=-0.12,
        aspect=20,
        shrink=0.5,
    )
    flow_bar.ax.set_xlabel(r"Flow $f_e$")

    gradient_cmap = plt.get_cmap("Reds")
    gradient_cmap.set_under("#0571b0")
    gradient_norm = plt.Normalize(
        vmin=0,
        vmax=max(float(edges["derivative_social_cost"].max()), 1e-9),
    )

    edges.sort_values("derivative_social_cost", ascending=False).plot(
        ax=ax_gradient,
        column="derivative_social_cost",
        legend=False,
        cmap=gradient_cmap,
        norm=gradient_norm,
    )
    if len(braess_edges) > 0:
        braess_edges.plot(
            ax=ax_gradient,
            color="#0571b0",
            linewidth=2,
            label="Braess edges",
        )
        ax_gradient.legend(loc="upper right")
    gradient_bar = fig.colorbar(
        make_scalar_mappable(gradient_cmap, gradient_norm),
        ax=ax_gradient,
        orientation="horizontal",
        pad=-0.12,
        aspect=20,
        shrink=0.5,
        extend="min",
    )
    gradient_bar.ax.set_xlabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$")

    ax_scatter.scatter(
        edges["flow"],
        edges["derivative_social_cost"],
        c="lightgrey",
        edgecolors="black",
        s=10,
        alpha=1,
        marker="o",
    )
    ax_scatter.grid()
    ax_scatter.set_xlabel(r"Flow $f_e$")
    ax_scatter.set_ylabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$", labelpad=5)
    ax_scatter.yaxis.tick_right()
    ax_scatter.yaxis.set_label_position("right")

    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")


if __name__ == "__main__":
    main()

# %%
