# %%
from src import Plotting as pl
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import multiCommoditySocialCost as mcsc
from src import SocialCost as sc
from src import TAPOptimization as tap
from src import osmGraphs as og


import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import cvxpy as cp
import matplotlib as mpl

import time


def demand_list(nodes, target_nodes_idx_list, gamma=0.1):
    # nodes, edges = ox.graph_to_gdfs(G)
    # nodes["source_node"] = False

    # sink_node_idx = sink_node.index[0]
    demands = []
    for t in target_nodes_idx_list:

        source_nodes_index = nodes.index.difference([t])

        P = dict(zip(nodes.index, np.zeros(len(nodes))))

        for s in source_nodes_index:
            P[s] = nodes.loc[s, "population"] * gamma / len(target_nodes_idx_list)

        P[t] = -np.sum(list(P.values()))

        demands.append(list(P.values()))
    # tot_sources = np.sum(list(P.values()))

    # for t in target_nodes_idx_list:
    # P[t] = nodes.loc[t, "population"] * (1 - gamma)
    #    P[t] = -tot_sources / len(target_nodes_idx_list)

    # print("Total Population:", tot_pop)
    return demands


# %%


G, boundary = og.osmGraph(
    "Munich,Germany",
    return_boundary=True,
    tolerance_meters=100,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]',
)

nodes, edges = ox.graph_to_gdfs(G)

# %%
adresss_list = [
    "Werner-Heisenberg-Allee 25, 80939 München",
]
target_node_idx_list = []
for a in adresss_list:
    location = ox.geocode(a)
    print(location)
    node_idx = ox.distance.nearest_nodes(G, location[1], location[0])
    target_node_idx_list.append(node_idx)

target_node_idx_list = np.unique(target_node_idx_list)


# %%
tot_pop = nodes["population"].sum()

seats_rhein_energie = 70_000

fraction_of_population = (seats_rhein_energie / tot_pop) * 0.15
od_mat = demand_list(nodes, target_node_idx_list, gamma=fraction_of_population)

od_mat = np.array(od_mat) * -1
# %%

nodes, edges = ox.graph_to_gdfs(G)
f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_mat, return_fw=True, pos_flows=True, solver=cp.OSQP, verbose=True
)
F = np.sum(f_mat, axis=0)

edges["flow"] = F
dsc = mcsc.derivative_social_cost(G, f_mat, od_mat, eps=1e-3)
dsc_vec = list(dsc.values())
edges["dsc"] = dsc_vec
# edges.sort_values("flow", ascending=True, inplace=True)
# %%

fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)

ax = axs[0]
boundary.boundary.plot(linewidth=1, ax=ax, color="black")
ax.grid()

ax.text(
    0.02,
    1.02,
    r"$\textbf{a}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=24,
)
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = mpl.colors.LogNorm(vmin=100, vmax=np.mean(np.sort(F)[-30:]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cb = fig.colorbar(
    sm, ax=ax, orientation="horizontal", pad=0.01, shrink=2 / 3, extend="both"
)


edges.sort_values("flow", ascending=True, inplace=True)
edges.plot(ax=ax, linewidth=2, column="flow", cmap=cmap, norm=norm, legend=False)

stadium_node = nodes.loc[target_node_idx_list]

stadium_node.plot(
    ax=ax,
    marker="s",
    color="black",
    markersize=100,
    zorder=3,
)
cb.ax.set_xlabel("Vehicle flow $f_{e}$")

ax = axs[1]
boundary.boundary.plot(linewidth=1, ax=ax, color="black")
ax.grid()

ax.text(
    0.02,
    1.02,
    r"$\textbf{b}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=24,
)

cmap = plt.get_cmap("Reds")
cmap.set_under("#0571b0")
norm = mpl.colors.Normalize(vmin=0, vmax=1e3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

edges.sort_values("dsc", ascending=True, inplace=True)
edges.plot(
    ax=ax,
    linewidth=2,
    column="dsc",
    cmap=cmap,
    norm=norm,
)

braess_edges = edges[edges["dsc"] < 0]
braess_edges.plot(
    ax=ax,
    linewidth=2,
    color="#0571b0",
    label="Braess edges",
)

stadium_node.plot(
    ax=ax,
    marker="s",
    color="black",
    markersize=100,
    zorder=3,
    label="Allianz Arena",
)

axs[1].legend(
    loc="upper right",
)

cb = fig.colorbar(
    sm,
    ax=ax,
    orientation="horizontal",
    pad=0.01,
    shrink=2 / 3,
    extend="min",
)
cb.ax.set_xlabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$")

axs[0].set_xlabel("Lon [°]")
axs[1].set_xlabel("Lon [°]")
axs[0].set_ylabel("Lat [°]")

fig.savefig("figs/munich-allianz.pdf", dpi=300, bbox_inches="tight")


# %%
