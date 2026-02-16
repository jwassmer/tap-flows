# %%
from src import Graphs as gr
from src import multiCommoditySocialCost as mcsc
import cvxpy as cp
import osmnx as ox
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

# %%

gamma = 0.04
G = gr.random_planar_graph(50, seed=42)

G.flows(num_sources="all", gamma=gamma, solver=cp.OSQP)
G.derivative_social_cost(num_sources="all", gamma=gamma, eps=1e-3, solver=cp.OSQP)
nodes, edges = ox.graph_to_gdfs(G)

# add edge length in metres. convert crs first
edges = edges.to_crs(epsg=3857)
edges["length"] = edges.geometry.length
edges = edges.to_crs(epsg=4326)


# %%
gamma_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
flow_dict = {}
dsc_dict = {}
for gamma in gamma_list:
    print(f"Computing flows and SCGC for gamma={gamma}")
    G.flows(num_sources="all", gamma=gamma, solver=cp.OSQP)
    G.derivative_social_cost(num_sources="all", gamma=gamma, eps=1e-3, solver=cp.OSQP)
    F = nx.get_edge_attributes(G, "flow")
    F_vec = np.array(list(F.values()))
    flow_dict[gamma] = F_vec
    dsc = nx.get_edge_attributes(G, "derivative_social_cost")
    dsc_vec = list(dsc.values())
    dsc_dict[gamma] = dsc_vec

# %%

gamma = 0.07
F = flow_dict[gamma]
dsc = dsc_dict[gamma]

edges["flow"] = F
edges["derivative_social_cost"] = dsc

braess_edges = edges[edges["derivative_social_cost"] < 0]

print("Length of braess edges:", np.sum(braess_edges["length"]))


fig = plt.figure(figsize=(10, 5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])

panellabels = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}"]

# Add subplots using GridSpec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
axs = [ax1, ax2, ax3]
for ax in [ax1, ax2]:
    ax.axis("off")


for ax, label in zip(axs, panellabels):
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=16)


cmap = plt.get_cmap("viridis")
cmap.set_under("lightgrey")
norm = plt.Normalize(vmin=0, vmax=max(edges["flow"]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

edges_sorted = edges.sort_values("flow", ascending=True)
edges_sorted.plot(ax=axs[0], column="flow", legend=False, cmap=cmap, norm=norm)
cb = fig.colorbar(
    sm,
    ax=axs[0],
    orientation="horizontal",
    pad=-0.12,
    aspect=20,
    shrink=0.5,
    # extend="min",
)
cb.ax.set_xlabel(r"Flow $f_e$")


cmap = plt.get_cmap("Reds")
cmap.set_under("blue")
vmin, vmax = min(edges["derivative_social_cost"]), max(edges["derivative_social_cost"])
norm = plt.Normalize(vmin=0, vmax=vmax)
edges_sorted = edges.sort_values("derivative_social_cost", ascending=False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
edges_sorted.plot(
    ax=axs[1], column="derivative_social_cost", legend=False, cmap=cmap, norm=norm
)
cb = fig.colorbar(
    sm,
    ax=axs[1],
    orientation="horizontal",
    pad=-0.12,
    aspect=20,
    shrink=0.5,
    extend="min",
)

cb.ax.set_xlabel(r"SCGC $\frac{\partial sc}{\partial \beta_e}$")

axs[2].scatter(
    edges["flow"],
    edges["derivative_social_cost"],
    c="lightgrey",
    edgecolors="black",
    s=10,
    alpha=1,
    marker="o",
)
axs[2].grid()
axs[2].set_xlabel(r"Flow $f_e$")
axs[2].set_ylabel(r"SCGC $\frac{\partial sc}{\partial \beta_e}$", labelpad=5)
# axs[2].tick_params(axis="y", which="both", pad=2)
axs[2].yaxis.tick_right()
axs[2].yaxis.set_label_position("right")

fig.savefig("figs/synthetic-graph-social-cost-gradient.pdf", bbox_inches="tight")
# %%
