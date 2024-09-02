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
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

G = gr.squareLattice(7, beta="random", alpha=1e0)

E = -nx.incidence_matrix(G, oriented=True)

load = 1000

source = 3
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

maxf = max(np.abs(f))

cmap = plt.cm.cividis
cmap.set_under("lightgrey")
cmap.set_bad("lightgrey")
norm = mpl.colors.LogNorm(vmin=1e0, vmax=maxf)
nc = {n: "red" if p > 0 else "lightblue" for n, p in zip(G.nodes, P)}
pl.graphPlot(G, ec=f, norm=norm, cmap=cmap, show_labels=True, nc=nc, ax=ax)

# %%

D = G.copy()
for v, e in zip(f, G.edges):
    if v < 1e-3:
        D.remove_edge(*e)


pos = {n: G.nodes[n]["pos"] for n in G.nodes}
nx.set_node_attributes(D, pos, "pos")
f0 = tap.user_equilibrium(D, P, solver=cp.SCS, eps_rel=1e-8, positive_constraint=False)
f00 = tap.linearTAP(D, P)[0]
pl.graphPlot(D, ec=f00, show_labels=True, nc=nc)
np.isclose(f0, f00, atol=1e-4).all()
# %%
f1_dict = {e: v for e, v in zip(G.edges, f)}
f0_dict = {e: v for e, v in zip(D.edges, f0)}

sim_arr = []
for k, v in f0_dict.items():
    sim_arr.append(v - f1_dict[k])


np.isclose(sim_arr, 0, atol=1e-4).all()


# %%
