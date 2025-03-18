# %%
from src import multiCommoditySocialCost as mcsc
from src import osmGraphs as og
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import cvxpy as cp

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# %%
location = "Innenstadt, Potsdam ,Germany"


G, bounds = og.osmGraph(
    location,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|residential|living"]',
    return_boundary=True,
    # heavy_boundary=True,
    # buffer_meter=2_500,
)
G.flows(num_sources="all", gamma=0.04, solver=cp.MOSEK)
nodes, edges = ox.graph_to_gdfs(G)


# %%

edges.explore(column="flow")


# %%
G.derivative_social_cost(num_sources=30, gamma=0.03, eps=1e-3, solver=cp.MOSEK)

nodes, edges = ox.graph_to_gdfs(G)


# %%
vmin, vmax = min(edges["derivative_social_cost"]), max(edges["derivative_social_cost"])
cmap = plt.get_cmap("cividis")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
edges.sort_values("derivative_social_cost", inplace=True, ascending=False)


def get_color(value):
    if value <= -1e-3:
        return "#FF0000"  # Red
    else:
        rgba = cmap(norm(value))
        return mpl.colors.to_hex(rgba)


edges["color"] = edges["derivative_social_cost"].apply(get_color)

# Now explore using the color column instead of the value
edges.explore(color=edges["color"])


# %%

nodes_list = []
edges_list = []

gamma_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

for gamma in gamma_list:
    G.derivative_social_cost(num_sources=25, gamma=gamma, eps=1e-3, solver=cp.MOSEK)

    nodes, edges = ox.graph_to_gdfs(G)
    nodes_list.append(nodes)
    edges_list.append(edges)

    braess_edges = edges[edges["derivative_social_cost"] < 0]
    total_braess_lengt = braess_edges["length"].sum()
    print("Gamma:", gamma)
    print("Braess Edges:", len(braess_edges))
    print("Braess Length:", total_braess_lengt)


# %%
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()
cmap = plt.get_cmap("Reds")
cmap.set_under("#0571b0")
# norm = mpl.colors.LogNorm(vmin=1e2, vmax=5e3)
norm = mpl.colors.Normalize(vmin=0, vmax=1e2)

labels = [
    r"\textbf{a}",
    r"\textbf{b}",
    r"\textbf{c}",
    r"\textbf{d}",
    r"\textbf{e}",
    r"\textbf{f}",
]

for i, ax in enumerate(axs):
    edges = edges_list[i]
    nodes = nodes_list[i]

    braess_edges = edges[edges["derivative_social_cost"] < 0]
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)

    edges.plot(
        column="derivative_social_cost",
        ax=ax,
        legend=False,
        cmap=cmap,
        norm=norm,
        linewidth=3,
    )

    bounds.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=2)
    nodes[nodes["source"]].plot(
        ax=ax, marker="x", color="black", markersize=25, zorder=3
    )
    ax.set_title(
        r"Traffic Volume $\gamma = $"
        + f"{gamma_list[i]:.2f} \n Braess edge length: {braess_edges['length'].sum():.0f} m"
    )
    ax.grid()

    ax.text(
        0.02,
        1.05,
        labels[i],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
    )


cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=axs,
    extend="both",
    shrink=1 / 2,
    pad=0.01,
    aspect=40,
)

cbar.ax.set_ylabel(r"Sensitivity $\partial_{\beta_e} \mathrm{sc}_e$")

red_patch = Patch(color="#0571b0", label="Braess Edges")
source_marker = Line2D(
    [],
    [],
    color="black",
    marker="x",
    linestyle="None",
    markersize=10,
    label="Source Nodes",
)
axs[2].legend(
    handles=[red_patch, source_marker],
    loc="upper right",
    # bbox_to_anchor=(1.4, 1.05),  # Adjust the position as needed
    borderaxespad=0.0,
)

fig.savefig(f"figs/{location}-braess.pdf", dpi=300, bbox_inches="tight")

# %%


fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

for i, ax in enumerate(axs.flatten()):
    edges = edges_list[i]
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)

    norm_flow = edges["flow"] / edges["flow"].max()
    norm_derivative = (
        edges["derivative_social_cost"] / edges["derivative_social_cost"].max()
    )

    diff = np.abs(norm_flow - norm_derivative)
    var = np.var(diff)
    ax.scatter(
        norm_flow,
        norm_derivative,
        marker="o",
        edgecolors="black",
        color="lightgrey",
        label=f"$\sigma= {var:.1e}$",
    )
    ax.set_title(r"Traffic Volume $\gamma = $" + f"{gamma_list[i]:.2f}")
    ax.set_xlabel(r"Traffic Flow $\|f_e\|$")
    ax.set_ylabel(r"Sensitivity $\|\partial_{\beta_e} \mathrm{sc}_e\|$")
    ax.legend()
    ax.grid()
    ax.text(
        0.02,
        1.05,
        labels[i],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
    )


fig.savefig(f"figs/{location}-braess-correlation.pdf", dpi=300, bbox_inches="tight")

# %%
