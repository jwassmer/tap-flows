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
mcsc.derivative_social_cost2(G, fmat, od_matrix, eps=1e-4)
# %%
mcsc.derivative_social_cost(G, fmat, od_matrix, eps=1e-4)

# %%

axlabels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}", r"\textbf{e}"]

fig = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], figure=fig)
axs = [fig.add_subplot(gs[i, j + 1]) for i in range(2) for j in range(2)]
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=0.5 - 1e-3, vmax=F.max())

for w, f in enumerate(fmat):
    axs[w].text(
        0.1, 1.05, axlabels[w + 1], ha="center", va="center", transform=axs[w].transAxes
    )
    U = gr.potential_subgraph(G, f)
    ff = nx.get_edge_attributes(U, "flow")
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
    )


ax_top = fig.add_subplot(gs[0:2, 0])
ax_top.text(
    0.15, 0.95, axlabels[0], ha="center", va="center", transform=ax_top.transAxes
)
pl.graphPlot(
    G,
    ec=F,
    ax=ax_top,
    title="",
    cbar=False,
    edge_labels=dict(zip(G.edges, F)),
    cmap=cmap,
    norm=norm,
    node_size=150,
)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs[1:3],
    shrink=1 / 2,
    extend="both",
    aspect=25,
    # orientation="horizontal",
)
cbar.ax.set_ylabel(r"Flow $f_{e}^w$", labelpad=2.5)
fig.savefig("figs/multicomm-fig.pdf", bbox_inches="tight")

# %%
# %%
