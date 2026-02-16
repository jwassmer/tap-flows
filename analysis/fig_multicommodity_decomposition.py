# %%
"""Visual decomposition of combined flow into commodity-specific subflows."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src import Graphs as gr
from src import Plotting as pl
from src import multiCommodityTAP as mc
from src.figure_style import apply_publication_style
from src.paper_examples import build_multicommodity_demo_graph

OUTPUT_FIGURE = "figs/multicommodity-decomposition.pdf"


def main() -> None:
    apply_publication_style(font_size=12)

    graph = build_multicommodity_demo_graph()
    od_matrix = np.array(
        [
            [3, 0, 0, -3],
            [-1, -1, 3, -1],
            [-1, 3, -1, -1],
            [-3, 0, 0, 3],
        ],
        dtype=float,
    )

    graph.edges[("B", "C")]["beta"] = 1
    graph.edges[("A", "C")]["beta"] = 1
    graph.edges[("B", "D")]["beta"] = 1

    flow_matrix, _ = mc.solve_multicommodity_tap(
        graph, od_matrix, pos_flows=True, return_fw=True
    )
    aggregate_flow = np.sum(flow_matrix, axis=0)
    node_to_index = {node: idx for idx, node in enumerate(graph.nodes())}

    panel_labels = [
        r"\textbf{a}",
        r"\textbf{b}",
        r"\textbf{c}",
        r"\textbf{d}",
        r"\textbf{e}",
    ]

    fig = plt.figure(figsize=(7, 3.2), constrained_layout=True)
    grid = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], figure=fig)

    commodity_axes = [
        fig.add_subplot(grid[i, j + 1]) for i in range(2) for j in range(2)
    ]

    flow_cmap = plt.get_cmap("cividis")
    flow_norm = mpl.colors.Normalize(vmin=0.5 - 1e-3, vmax=float(flow_matrix.max()))

    od_cmap = plt.get_cmap("coolwarm_r")
    od_norm = mpl.colors.Normalize(
        vmin=float(od_matrix.min()) - 1e-3,
        vmax=float(od_matrix.max()) + 1e-3,
    )

    for commodity_idx, commodity_flow in enumerate(flow_matrix):
        ax = commodity_axes[commodity_idx]
        ax.text(
            0.1,
            1.05,
            panel_labels[commodity_idx + 1],
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        subgraph = gr.potential_subgraph(graph, commodity_flow)
        subgraph_flow = nx.get_edge_attributes(subgraph, "flow")

        node_supply = od_matrix[commodity_idx]
        node_colors = {
            node: od_cmap(od_norm(node_supply[node_to_index[node]]))
            for node in subgraph.nodes()
        }

        pl.graphPlot(
            subgraph,
            ec=subgraph_flow,
            edge_labels=subgraph_flow,
            ax=ax,
            title="",
            cbar=False,
            cmap=flow_cmap,
            norm=flow_norm,
            node_size=150,
            edgewith=3,
            nc=node_colors,
        )

    commodity_flow_bar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=flow_norm, cmap=flow_cmap),
        ax=commodity_axes[1:3],
        shrink=1 / 2,
        extend="both",
        aspect=20,
        pad=0.01,
    )
    commodity_flow_bar.ax.set_ylabel(r"Flow $f_e^w$", labelpad=2.5, fontsize=12)

    od_bar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=od_norm, cmap=od_cmap),
        ax=commodity_axes[1:3],
        shrink=1 / 3,
        aspect=15,
        pad=-0.02,
        orientation="horizontal",
    )
    od_bar.ax.set_xlabel(r"OD matrix $p_n^w$", labelpad=2.5, fontsize=12)

    aggregate_ax = fig.add_subplot(grid[0:2, 0])
    aggregate_ax.text(
        0.15,
        0.95,
        panel_labels[0],
        ha="center",
        va="center",
        transform=aggregate_ax.transAxes,
    )

    aggregate_cmap = plt.get_cmap("viridis")
    aggregate_norm = mpl.colors.Normalize(
        vmin=float(aggregate_flow.min()),
        vmax=float(aggregate_flow.max()),
    )
    pl.graphPlot(
        graph,
        ec=aggregate_flow,
        ax=aggregate_ax,
        title="",
        cbar=False,
        edge_labels=dict(zip(graph.edges, aggregate_flow)),
        cmap=aggregate_cmap,
        norm=aggregate_norm,
        node_size=150,
        edgewith=3,
    )

    aggregate_bar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=aggregate_norm, cmap=aggregate_cmap),
        ax=aggregate_ax,
        shrink=1 / 2,
        extend="both",
        aspect=25,
        orientation="horizontal",
        pad=-0.02,
    )
    aggregate_bar.ax.set_xlabel(r"Combined flow $f_e$", labelpad=2.5, fontsize=12)

    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")


if __name__ == "__main__":
    main()

# %%
