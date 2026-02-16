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

# %%


G, boundary = og.osmGraph(
    "Cologne,Germany",
    return_boundary=True,
)

G.flows(gamma=0.1, num_sources=25)
nodes, edges = ox.graph_to_gdfs(G)
edges.sort_values(by="flow")


# %%

# edges.sort_values("flow", ascending=True, inplace=True)
# %%

fig, ax = plt.subplots(1, 1, figsize=(12, 10), sharex=True, sharey=True)

ax.grid()
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
cmap.set_over("red")
norm = mpl.colors.LogNorm(vmin=100, vmax=np.max(edges["flow"]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cb = fig.colorbar(
    sm, ax=ax, orientation="horizontal", pad=0.01, shrink=1 / 2, extend="min", aspect=30
)


edges.sort_values("flow", ascending=True, inplace=True)
edges.plot(ax=ax, linewidth=2, column="flow", cmap=cmap, norm=norm, legend=False)


cb.ax.set_xlabel("Vehicle flow $f_{e}$")

# boundary.plot(ax=ax, alpha=0.1, color="black", zorder=-1)
boundary.boundary.plot(linewidth=1, ax=ax, color="black")

# %%


# %%
