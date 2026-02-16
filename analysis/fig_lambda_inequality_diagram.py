# %%
"""Two-node explanatory diagram for lambda inequality arguments."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from src.figure_style import apply_publication_style

OUTPUT_FIGURE = "figs/lambda-inequality-diagram.pdf"


def main() -> None:
    apply_publication_style(font_size=18)
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{stmaryrd}"

    graph = nx.DiGraph()
    graph.add_node("n")
    graph.add_node("m")
    graph.add_edge("n", "m", color="black", style="-", label=r"$\beta_{nm}$")
    graph.add_edge("m", "n", color="red", style="--", label=r"$\beta_{mn}$")

    pos = {"n": (0, 0), "m": (1, 0)}

    fig, ax = plt.subplots(figsize=(6, 3))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=600,
        node_color="lightgrey",
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=22,
        font_family="sans-serif",
        font_weight="bold",
        verticalalignment="center",
        ax=ax,
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=nx.get_edge_attributes(graph, "label"),
        font_size=22,
        connectionstyle="arc3,rad=0.4",
        font_color="black",
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        connectionstyle="arc3,rad=0.27",
        ax=ax,
        arrows=True,
        edge_color="white",
        alpha=0,
    )

    for u, v, attrs in graph.edges(data=True):
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[(u, v)],
            connectionstyle="arc3,rad=0.27",
            arrows=True,
            arrowsize=40,
            edge_color=attrs["color"],
            style=attrs["style"],
            arrowstyle="->",
            ax=ax,
        )

    ax.text(0, -0.15, r"$\lambda_n$", fontsize=22, ha="center")
    ax.text(1.02, -0.15, r"$\lambda_n+\beta_{nm}$", fontsize=22, ha="center")
    ax.text(0.0, 0.15, r"$\lambda_n+\beta_{nm}+\beta_{mn}$", fontsize=22, ha="center")
    ax.text(0.5, 0.3, r"$\lightning$", fontsize=150, ha="center", va="center")

    ax.axis("off")
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()

# %%
