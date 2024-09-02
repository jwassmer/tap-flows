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
radius = 3

flows = []
for alpha in [1e-2, 1e-1, 1e0, 1e1]:

    H = gr.triangularLattice(radius, beta="random", alpha=alpha, directed=True)

    source = 0  # int((len(H) - 1) / 2)
    P = -np.ones(H.number_of_nodes())
    P[source] = np.sum(np.abs(P)) - 1

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

fig, axs = plt.subplots(len(flows), figsize=(5, 20))

maxf = max([max(f) for f in flows])
for i, f in enumerate(flows):
    cmap = plt.cm.cividis
    cmap.set_under("lightgrey")
    cmap.set_bad("lightgrey")
    norm = mpl.colors.LogNorm(vmin=1e-1, vmax=maxf)
    nc = {n: "red" if p > 0 else "lightblue" for n, p in zip(H.nodes, P)}
    pl.graphPlot(
        H, ec=f, norm=norm, cmap=cmap, show_labels=True, nc=nc, ax=axs[i], title=None
    )

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

G = gr.triangularLattice(3, beta="random", alpha=1e-3)
# G = gr.random_graph(20, 0.2, beta="random", alpha=1e0)

E = -nx.incidence_matrix(G, oriented=True)

load = 1000

source = 0
sinks = list(G.nodes())
sinks.remove(source)
P = np.zeros(G.number_of_nodes())
# P[source] = load
# P[sinks] = -load / len(sinks)
P = -np.ones(G.number_of_nodes())
P[source] = np.sum(np.abs(P)) - 1

f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS, eps_rel=1e-8)
s = sibc._single_source_interaction_betweenness_centrality(G, weight="beta", P=P)
print(sum(f))
print(sum(s))

maxf = max(np.abs(f - s))

cmap = plt.cm.coolwarm
norm = mpl.colors.SymLogNorm(linthresh=1e-2, linscale=1, vmin=-maxf, vmax=maxf)
nc = {n: "red" if p > 0 else "lightblue" for n, p in zip(G.nodes, P)}
pl.graphPlot(G, ec=f - s, norm=norm, cmap=cmap, show_labels=True, nc=nc, ax=ax)

# %%
C = gr.cycle_edge_incidence_matrix(G)
deltaf = f - s
# %%

v = C @ deltaf

v.min()

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
