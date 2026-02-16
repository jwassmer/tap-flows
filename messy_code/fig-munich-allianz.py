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
import pandas as pd

import time


def demand_list(
    nodes,
    destination_nodes,
    gamma=0.05,
    demand_col="population",
):
    """
    Generate OD demand list from nodes to given destination nodes.

    Parameters
    ----------
    nodes : GeoDataFrame or DataFrame
        Node table with at least `id_col` and `demand_col` columns.
    commodity : list or array
        IDs (or indices) of destination nodes.
    gamma : float
        Fraction of each node's population that generates demand.
    demand_col : str
        Column name for demand size (e.g. population).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    od_list : DataFrame
        Columns: ['origin', 'destination', 'demand']
    """

    origins = np.array(nodes.index)
    destinations = np.array(destination_nodes.index)
    od_mat = pd.DataFrame(0, index=origins, columns=destinations, dtype=float)

    total_pop = destination_nodes[demand_col].sum()
    destination_attractiveness = destination_nodes[demand_col] / total_pop

    for d in destinations:

        for o in origins:
            pop = nodes.loc[o, demand_col]
            demand_total = pop * gamma
            od_mat.loc[o, d] = demand_total * destination_attractiveness[d]

        od_mat.loc[d, d] = od_mat.loc[d, d] - od_mat[d].sum()

    return np.array(od_mat).T


# %%


G, boundary = og.osmGraph(
    "Cologne,Germany",
    return_boundary=True,
    tolerance_meters=100,
    highway_filter='["highway"~"motorway|trunk|primary|secondary"]',
)

nodes, edges = ox.graph_to_gdfs(G)

num_destinations = len(nodes) // 10
destination_nodes = og.select_evenly_distributed_nodes(nodes, num_destinations)
od_mat = demand_list(
    nodes,
    destination_nodes,
    gamma=0.03,
    demand_col="population",
)

# %%
f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_mat, return_fw=True, pos_flows=True, solver=cp.OSQP, verbose=True
)
F = np.sum(f_mat, axis=0)

edges["flow"] = F
dsc = mcsc.derivative_social_cost(G, f_mat, od_mat, eps=1e-3, demands_to_sinks=False)
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

if len(braess_edges) > 0:
    braess_edges.plot(
        ax=ax,
        linewidth=2,
        color="#0571b0",
        label="Braess edges",
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
