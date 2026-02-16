# %%
"""Compare user-equilibrium and social-optimum flows on a toy network."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src import Plotting as pl
from src import SocialCost as sc
from src import multiCommodityTAP as mc
from src.figure_style import add_panel_label, apply_publication_style
from src.paper_examples import build_social_optimum_demo_graph

OUTPUT_FIGURE = "figs/user-equilibrium-vs-social-optimum.pdf"


def main() -> None:
    apply_publication_style(font_size=16)

    graph = build_social_optimum_demo_graph()
    graph.edges[("B", "C")]["beta"] = 1
    graph.edges[("A", "C")]["beta"] = 10
    graph.edges[("B", "D")]["beta"] = 10

    od_matrix = np.array([[10, 0, 0, -10]], dtype=float)

    flow_ue = mc.solve_multicommodity_tap(graph, od_matrix, pos_flows=True)
    flow_so = mc.solve_multicommodity_tap(
        graph,
        od_matrix,
        pos_flows=True,
        social_optimum=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    flows = [flow_ue, flow_so]
    titles = ["User equilibrium", "Social optimum"]
    labels = [r"\textbf{a}", r"\textbf{b}"]

    vmin = float(min(np.min(flow_ue), np.min(flow_so)))
    vmax = float(max(np.max(flow_ue), np.max(flow_so)))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for i, ax in enumerate(axes):
        flow = flows[i]
        pl.graphPlot(
            graph,
            ec=flow,
            edge_labels=dict(zip(graph.edges, flow)),
            title="",
            ax=ax,
            cbar=False,
            cmap=cmap,
            norm=norm,
            edgewith=7,
        )
        ax.set_title(titles[i], y=0.95, fontsize=22)
        add_panel_label(ax, labels[i], x=0.1, y=1.0, fontsize=30)

        total_sc = sc.total_social_cost(graph, flow)
        ax.text(
            0.5,
            -0.12,
            rf"$SC(f)$ = {total_sc:.0f}",
            ha="center",
            va="center",
            fontsize=20,
        )

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        orientation="horizontal",
        pad=-0.03,
        aspect=20,
        shrink=1 / 4,
    )
    colorbar.ax.set_xlabel(r"Flow $f_e$")

    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()

# %%
