# %%
from src import Plotting as pl
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from src import multiCommoditySocialCost as mcsc

np.set_printoptions(precision=3, suppress=True)


# %%

G = nx.DiGraph()
edges = [(0, 1), (1, 0), (0, 2), (2, 0), (2, 3), (3, 2), (1, 3), (3, 1), (1, 2)]
G.add_edges_from(edges)
nx.set_edge_attributes(G, dict(zip(edges, np.ones(len(edges)))), "alpha")
nx.set_edge_attributes(G, dict(zip(edges, np.zeros(len(edges)))), "beta")
pos = {0: (0, 0.5), 3: (1, 0.5), 1: (0.5, 0), 2: (0.5, 1)}
nx.set_node_attributes(G, pos, "pos")
# relabel nodes to abcd
mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
G = nx.relabel_nodes(G, mapping)

od_matrix = np.array([[3, 0, 0, -3], [-1, -1, 3, -1], [-1, 3, -1, -1], [-3, 0, 0, 3]])

# od_matrix = [od_matrix[2]]
# %%

G.edges[("B", "C")]["beta"] = 1
G.edges[("A", "C")]["beta"] = 1
G.edges[("B", "D")]["beta"] = 1

fmat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_matrix, pos_flows=True, return_fw=True
)
F = np.sum(fmat, axis=0)
# pl.graphPlot(G, ec=F, edge_labels=dict(zip(G.edges, F)), title="")
pl.graphPlot(G, ec=F, edge_labels=dict(zip(G.edges, F)), title="")

# %%
mcsc.derivative_social_cost(G, fmat, od_matrix, eps=1e-4)

# %%

axlabels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}", r"\textbf{e}"]

fig = plt.figure(figsize=(7, 3.2), constrained_layout=True)
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], figure=fig)
axs = [fig.add_subplot(gs[i, j + 1]) for i in range(2) for j in range(2)]
cmap = plt.get_cmap("cividis")
# cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=0.5 - 1e-3, vmax=fmat.max())

cmap_n = plt.get_cmap("coolwarm_r")
norm_n = mpl.colors.Normalize(vmin=od_matrix.min() - 1e-3, vmax=od_matrix.max() + 1e-3)

for w, f in enumerate(fmat):
    axs[w].text(
        0.1, 1.05, axlabels[w + 1], ha="center", va="center", transform=axs[w].transAxes
    )
    U = gr.potential_subgraph(G, f)
    ff = nx.get_edge_attributes(U, "flow")

    nc = od_matrix[w]
    node_colors = {n: cmap_n(norm_n(nc[i])) for i, n in enumerate(U.nodes())}

    pl.graphPlot(
        U,
        ec=ff,
        edge_labels=ff,
        ax=axs[w],
        title="",
        cbar=False,
        cmap=cmap,
        norm=norm,
        node_size=150,
        edgewith=3,
        nc=node_colors,
    )
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs[1:3],
    shrink=1 / 2,
    extend="both",
    aspect=20,
    pad=0.01,
)
cbar.ax.set_ylabel(r"Flow $f_{e}^w$", labelpad=2.5, fontsize=12)

cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm_n, cmap=cmap_n),
    ax=axs[1:3],
    shrink=1 / 3,
    aspect=15,
    pad=-0.02,
    orientation="horizontal",
)
cbar.ax.set_xlabel(r"OD matrix $p_{n}^w$", labelpad=2.5, fontsize=12)


ax_top = fig.add_subplot(gs[0:2, 0])
ax_top.text(
    0.15, 0.95, axlabels[0], ha="center", va="center", transform=ax_top.transAxes
)

cmap_viridis = plt.get_cmap("viridis")
norm_viridis = mpl.colors.Normalize(vmin=F.min(), vmax=F.max())
pl.graphPlot(
    G,
    ec=F,
    ax=ax_top,
    title="",
    cbar=False,
    edge_labels=dict(zip(G.edges, F)),
    cmap=cmap_viridis,
    norm=norm_viridis,
    node_size=150,
    edgewith=3,
)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm_viridis, cmap=cmap_viridis),
    ax=ax_top,
    shrink=1 / 2,
    extend="both",
    aspect=25,
    orientation="horizontal",
    pad=-0.02,
)

cbar.ax.set_xlabel(r"Combined flow $f_{e}$", labelpad=2.5, fontsize=12)

fig.savefig("figs/multicom-fig.pdf", bbox_inches="tight")

# %%
# %%
