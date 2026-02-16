# %%
"""Classic Braess paradox figure: social cost vs. central-edge travel time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src import TAPOptimization as tap
from src import SocialCost as sc
from src.figure_style import add_panel_label, apply_publication_style
from src.paper_examples import (
    build_classic_braess_social_cost_graph,
    default_node_colors,
    linear_travel_time_label,
)

OUTPUT_FIGURE = "figs/classic-braess-social-cost-scan.pdf"


def _social_cost_scan(
    graph: nx.DiGraph,
    demand: np.ndarray,
    scanned_edge: tuple[str, str],
    beta_values: np.ndarray,
) -> list[float]:
    values = []
    for beta in beta_values:
        graph.edges[scanned_edge]["beta"] = float(beta)
        flow = tap.user_equilibrium(graph, demand, positive_constraint=True)
        values.append(sc.total_social_cost(graph, flow))
    return values


def main() -> None:
    apply_publication_style(font_size=18)

    graph = build_classic_braess_social_cost_graph()
    demand = np.zeros(graph.number_of_nodes())
    demand[0] = 40
    demand[-1] = -40

    scanned_edge = ("C", "B")
    beta_values = np.linspace(0, 5, 100)
    social_cost_values = _social_cost_scan(graph, demand, scanned_edge, beta_values)

    graph.remove_edge(*scanned_edge)
    removed_flow = tap.user_equilibrium(graph, demand, positive_constraint=True)
    removed_edge_social_cost = sc.total_social_cost(graph, removed_flow)
    graph.add_edge(*scanned_edge, alpha=0.0, beta=float(beta_values[-1]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 2])

    axes[0].plot(beta_values, social_cost_values, color="black", linewidth=4)
    axes[0].scatter(
        beta_values[-1],
        removed_edge_social_cost,
        color="red",
        label="Deleted edge $(C,B)$",
        marker="x",
        zorder=3,
        s=250,
    )
    axes[0].set_xlabel(r"Free flow travel time $\beta_{CB}$")
    axes[0].set_ylabel(r"Total social cost $SC(f)$")
    axes[0].grid()
    axes[0].legend(loc="upper right")
    add_panel_label(axes[0], r"\textbf{a}", x=0.05, y=1.03, fontsize=30)

    nx.draw(
        graph,
        nx.get_node_attributes(graph, "pos"),
        node_size=1000,
        width=2,
        with_labels=True,
        node_color=list(default_node_colors().values()),
        font_weight="bold",
        arrowsize=18,
        ax=axes[1],
        font_size=22,
    )

    edge_labels = {}
    for edge in graph.edges:
        alpha = graph.edges[edge]["alpha"]
        beta = graph.edges[edge]["beta"]
        edge_labels[edge] = linear_travel_time_label(edge, alpha, beta)
    edge_labels[scanned_edge] = r"$t_{CB}(f_{CB})=\beta_{CB}$"

    nx.draw_networkx_edge_labels(
        graph,
        pos=nx.get_node_attributes(graph, "pos"),
        edge_labels=edge_labels,
        font_size=22,
        font_color="black",
        ax=axes[1],
    )
    add_panel_label(axes[1], r"\textbf{b}", x=0.1, y=1.0, fontsize=32)

    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

# %%
