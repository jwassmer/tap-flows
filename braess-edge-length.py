# %%
# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import Plotting as pl
from src import osmGraphs as og
from src import multiCommoditySocialCost as mcsc

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import cvxpy as cp
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# %%

location = "Nippes, Cologne, Germany"

G = og.osmGraph(location)

gamma_list = np.linspace(0.01, 0.1, 16)

edges_list = []
for gamma in gamma_list:
    G.derivative_social_cost(num_sources=15, gamma=gamma, eps=1e-3, solver=cp.MOSEK)
    nodes, edges = ox.graph_to_gdfs(G)
    edges_list.append(edges)

# %%
braess_edge_length = []
for edges in edges_list:
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)
    braess_edges = edges[edges["derivative_social_cost"] < 0]
    braess_edge_length.append(braess_edges["length"].sum())

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

ax.plot(gamma_list, braess_edge_length, label="Braess edges length", marker="o")
ax.grid()
# %%

fig, axs = plt.subplots(4, 4, figsize=(18, 20), sharey=True, sharex=True)
axs = axs.flatten()

cmap = plt.get_cmap("Reds")
cmap.set_under("Blue")
norm = mpl.colors.Normalize(vmin=0, vmax=1e3)

for i, edges in enumerate(edges_list):
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)
    braess_edges = edges[edges["derivative_social_cost"] < 0]
    edges.plot(
        ax=axs[i], column="derivative_social_cost", cmap=cmap, norm=norm, legend=False
    )

    axs[i].set_title(f"$\gamma$ = {gamma_list[i]:.3f}")
    axs[i].axis("off")

# %%
