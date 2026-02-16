# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cvxpy as cp

from src import Graphs as gr
from src import Plotting as pl
from src import SIBC as sibc
from src import TAPOptimization as tap

# %%
radius = 2

flows = []
for alpha in [1e-1, 1e1]:

    H = gr.triangularLattice(radius, beta="random", alpha=alpha, directed=True)

    source = 0  # int((len(H) - 1) / 2)
    P = np.zeros(H.number_of_nodes())
    P[source] = len(H)
    P[-1] = -len(H)

    f = tap.user_equilibrium(
        H, P, positive_constraint=True, solver=cp.SCS, eps_rel=1e-8
    )
    # f_norm = f / np.max(f)
    flows.append(f)
    # if len(flows) > 1:
    #    print(np.max((flows[-1] - flows[-2])))
    # print(np.min((flows[-1] - flows[-2])))
    print(sum(f))

s = sibc._single_source_interaction_betweenness_centrality(H, weight="beta", P=P)

flows.insert(0, s)

# %%

fig, axs = plt.subplots(1, len(flows), figsize=(12, 4))

maxf = max([max(f) for f in flows])
for i, f in enumerate(flows):
    cmap = plt.cm.cividis
    cmap.set_under("lightgrey")
    cmap.set_bad("lightgrey")
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=maxf)
    nc = {
        n: "red" if p > 0 else "lightblue" if p < 0 else "lightgrey"
        for n, p in zip(H.nodes, P)
    }
    pl.graphPlot(
        H,
        ec=f,
        norm=norm,
        cmap=cmap,
        show_labels=True,
        nc=nc,
        ax=axs[i],
        title=None,
        cbar=False,
    )


cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axs,
    orientation="horizontal",
    fraction=0.05,
    pad=0.05,
    extend="min",
)
cbar.set_label(r"traffic flow $f_e$")

fig.savefig("figs/loopyness.png", dpi=300, bbox_inches="tight")

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

G = gr.triangularLattice(2, beta="random", alpha=1e-1)
# G = gr.random_graph(20, 0.2, beta="random", alpha=1e0)

E = -nx.incidence_matrix(G, oriented=True)

load = 10

source = 0
sinks = [-1]
P = np.zeros(G.number_of_nodes())
P[source] = load
P[-1] = -load

f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS, eps_rel=1e-8)
s = sibc._single_source_interaction_betweenness_centrality(G, weight="beta", P=P)

maxf = max(np.abs(f - s))

cmap = plt.cm.coolwarm
norm = mpl.colors.SymLogNorm(linthresh=1e-2, linscale=1, vmin=-maxf, vmax=maxf)
nc = {
    n: "red" if p > 0 else "lightblue" if p < 0 else "lightgrey"
    for n, p in zip(G.nodes, P)
}
pl.graphPlot(G, ec=f - s, norm=norm, cmap=cmap, show_labels=True, nc=nc, ax=ax)

fig.savefig("figs/loopyness.png", dpi=300, bbox_inches="tight")

# %%

np.round(f - s, 2)
# %%

nx.minimum_cycle_basis(G.to_undirected())


def edge_cycle_incidence_matrix(G):
    """
    Returns the edge-cycle incidence matrix of a directed graph G.
    """
    cycles = nx.minimum_cycle_basis(G.to_undirected())
    E = nx.incidence_matrix(G, oriented=True)
    C = np.zeros((len(cycles), E.shape[1]))
    for i, cycle in enumerate(cycles):
        for u, v in zip(cycle, cycle[1:] + cycle[:1]):
            edge = (u, v)
            j = list(G.edges).index(edge)
            C[i, j] = 1
    return C


C = edge_cycle_incidence_matrix(G)

C

# %%


# Create a graph with 5 nodes forming a ring
G = nx.cycle_graph(6)
G = nx.to_directed(G)
E = -nx.incidence_matrix(G, oriented=True)
nx.set_edge_attributes(G, 1e-3, "alpha")
nx.set_edge_attributes(G, 1, "beta")
G.edges[(0, 5)]["beta"] = 5
pos = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (4, 1), 5: (5, 0)}
nx.set_node_attributes(G, pos, "pos")

source = 0
sink = 5
P = np.zeros(G.number_of_nodes())
P[source] = 10
P[sink] = -10

f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS, eps_rel=1e-8)
s = sibc._single_source_interaction_betweenness_centrality(G, weight="beta", P=P)

pl.graphPlot(G, ec=f, show_labels=True)
{e: np.round(v, 2) for e, v in zip(G.edges, f)}
# %%

# Create a graph with 5 nodes forming a ring
G = nx.cycle_graph(5)
G = nx.to_directed(G)
E = -nx.incidence_matrix(G, oriented=True)
nx.set_edge_attributes(G, 1e-3, "alpha")
nx.set_edge_attributes(G, 1, "beta")
G.edges[(0, 4)]["beta"] = 0
G.edges[(4, 3)]["beta"] = 3
pos = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (4, 0)}
nx.set_node_attributes(G, pos, "pos")

source = 0
sink = 3
load = 100
P = np.zeros(G.number_of_nodes())
P[source] = load
P[sink] = -load
nc = {
    n: "red" if p > 0 else "lightblue" if p < 0 else "lightgrey"
    for n, p in zip(G.nodes, P)
}

f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS, eps_rel=1e-8)
s = sibc._single_source_interaction_betweenness_centrality(G, weight="beta", P=P)

pl.graphPlot(G, ec=f, show_labels=True, nc=nc)
{e: np.round(v, 2) for e, v in zip(G.edges, f)}
# %%
s
# %%
