# %%
"""Validation of analytic SCGC against numerical derivatives on Braess graph."""

from __future__ import annotations

import matplotlib as mpl


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from src import Plotting as pl
from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc
from src.figure_style import apply_publication_style
from src.paper_examples import build_classic_braess_validation_graph

OUTPUT_FIGURE = "figs/classic-braess-scgc-validation.pdf"


def _middle_value(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n % 2 == 1:
        return float(values[n // 2])
    return float(0.5 * (values[n // 2 - 1] + values[n // 2]))


def main() -> None:
    apply_publication_style(font_size=16)

    graph = build_classic_braess_validation_graph()

    load = 9.5
    demand = -np.ones(graph.number_of_nodes()) * load / (graph.number_of_nodes() - 1)
    demand[0] = load

    f_mat, _ = mc.solve_multicommodity_tap(graph, [demand], return_fw=True)

    derivatives = mcsc.derivative_social_cost(graph, f_mat, [demand], eps=1e-3)
    derivative_values = np.array([derivatives.get(edge, 0.0) for edge in graph.edges])
    derivative_labels = dict(zip(graph.edges, derivative_values))

    fig = plt.figure(figsize=(14, 6))
    panel_labels = [
        r"\textbf{a}",
        r"\textbf{b}",
        r"\textbf{c}",
        r"\textbf{d}",
        r"\textbf{e}",
        r"\textbf{f}",
    ]

    gs = GridSpec(4, 4, figure=fig, width_ratios=[2, 1, 1, 1])
    network_ax = fig.add_subplot(gs[:, 0])
    network_ax.text(
        0.25,
        0.95,
        panel_labels[0],
        transform=network_ax.transAxes,
        fontsize=22,
        fontweight="bold",
    )

    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.TwoSlopeNorm(
        vmin=float(derivative_values.min()),
        vmax=float(derivative_values.max()),
        vcenter=0,
    )

    pl.graphPlot(
        graph,
        ax=network_ax,
        ec=derivative_values,
        edge_labels=derivative_labels,
        cmap=cmap,
        norm=norm,
        edgewith=6,
        cbar=False,
        title="",
    )

    colorbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        orientation="horizontal",
        ax=network_ax,
        shrink=1 / 2,
        pad=-0.05,
        aspect=20,
    )
    colorbar.ax.set_xlabel(
        r"SCGC $\frac{\partial}{\partial\beta_e}SC(\mathbf{\beta_e})$"
    )

    for edge_index, edge in enumerate(graph.edges):
        if edge_index < 4:
            row = (edge_index % 2) * 2
            col = int(np.floor(1 + edge_index / 2))
            ax = fig.add_subplot(gs[row : row + 2, col])
        else:
            ax = fig.add_subplot(gs[1:3, 3])

        ax.text(
            0.01,
            1.02,
            panel_labels[edge_index + 1],
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
        )

        slopes, beta_values, social_cost_values = mcsc.numerical_derivative(
            graph,
            [demand],
            edge,
            var_percentage=0.25,
        )

        ax.plot(beta_values, social_cost_values, linewidth=2, marker=".", color="grey")

        beta_mid = beta_values[10:15]
        social_cost_mid = social_cost_values[10:15]
        slope = _middle_value(slopes)
        color = cmap(norm(slope))

        ax.plot(
            beta_mid,
            social_cost_mid,
            color=color,
            marker=".",
            markersize=10,
            linewidth=2.5,
            label=rf"$\partial SC / \partial\beta_e$ = {slope:.2f}",
        )
        ax.legend(loc="upper right")

        edge_label = (
            str(edge)
            .replace("(", "")
            .replace(")", "")
            .replace(",", r"$\rightarrow$")
            .replace("'", "")
        )
        ax.set_title(f"Edge {edge_label}")
        ax.grid()

    fig.text(
        2 / 3 + 0.04, 0.05, r"Free flow travel time $\beta_e$", ha="center", va="center"
    )
    fig.text(
        1 / 3 + 0.02, 0.5, r"Social cost", ha="center", va="center", rotation="vertical"
    )

    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

# %%
