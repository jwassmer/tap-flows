# %%
from src import Plotting as pl
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc

import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import osmnx as ox

from src import multiCommoditySocialCost as mcsc

np.set_printoptions(precision=3, suppress=True)


# %%

G = nx.DiGraph()
edges = [(0, 1), (0, 2), (2, 3), (1, 3), (1, 2)]
G.add_edges_from(edges)
nx.set_edge_attributes(G, dict(zip(edges, np.ones(len(edges)))), "alpha")
nx.set_edge_attributes(G, dict(zip(edges, np.zeros(len(edges)))), "beta")
pos = {0: (0, 0.5), 3: (1, 0.5), 1: (0.5, 0), 2: (0.5, 1)}
nx.set_node_attributes(G, pos, "pos")
# relabel nodes to abcd
mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
G = nx.relabel_nodes(G, mapping)

od_matrix = np.array([[10, 0, 0, -10]])

# od_matrix = [od_matrix[2]]
# %%

G.edges[("B", "C")]["beta"] = 1
G.edges[("A", "C")]["beta"] = 10
G.edges[("B", "D")]["beta"] = 10


f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_matrix, pos_flows=True, return_fw=True
)


F_ue = mc.solve_multicommodity_tap(
    G,
    od_matrix,
    pos_flows=True,
)

F_so = mc.solve_multicommodity_tap(G, od_matrix, pos_flows=True, social_optimum=True)
# pl.graphPlot(G, ec=F, edge_labels=dict(zip(G.edges, F)), title="")
pl.graphPlot(G, ec=F_ue, edge_labels=dict(zip(G.edges, F_ue)), title="")

# %%

fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

F_list = [
    F_ue,
    F_so,
]

axlabels = [r"\textbf{a}", r"\textbf{b}"]

cmap = plt.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=2.25, vmax=7.25)

for i, ax in enumerate(axs):
    pl.graphPlot(
        G,
        ec=F_list[i],
        edge_labels=dict(zip(G.edges, F_list[i])),
        title="",
        ax=ax,
        cbar=False,
        cmap=cmap,
        norm=norm,
        edgewith=7,
    )
    ax.set_title(
        "User equilibrium" if i == 0 else "Social optimum", y=0.95, fontsize=22
    )
    ax.text(
        0.1,
        1.0,
        axlabels[i],
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=30,
    )

    total_social_cost = sc.total_social_cost(G, F_list[i])
    ax.text(
        0.5,
        -0.12,
        rf"$sc(f_e) =$ {total_social_cost:.0f}",
        ha="center",
        va="center",
        fontsize=22,
    )


cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    orientation="horizontal",
    pad=-0.03,
    aspect=20,
    shrink=1 / 4,
)
cbar.ax.set_xlabel(r"Flow $f_e$", fontsize=22)
cbar.set_ticks([2.5, 3.5, 4.5, 5.5, 6.5])
cbar.ax.tick_params(labelsize=18)


fig.savefig("figs/fig-social-optimum-user-eq.pdf", bbox_inches="tight", dpi=300)

# %%
