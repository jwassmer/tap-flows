# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src import TAPOptimization as tap
from src import Plotting as pl
from src import ConvexOptimization as co
from src import Equilibirium as eq
from src import LinAlg as la


def positive_flow_graph(G: nx.DiGraph, F: list) -> nx.DiGraph:
    Fdict = dict(zip(G.edges, F))
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, **d)
    for i, j, d in G.edges(data=True):
        flow = Fdict[(i, j)]
        # b = beta_dict[(i, j)]
        if flow >= -1e-7:
            D.add_edge(i, j, **d)
            # D[i][j]["flow"] = flow
        else:
            D.add_edge(j, i, **d)
            # D[j][i]["flow"] = -flow
    return D


def positive_flow_constraint(G, lambd):
    E = -nx.incidence_matrix(G, oriented=True)
    beta = np.array([G.edges[e]["tt_function"](0) for e in G.edges()])
    edges = list(G.edges)
    f = E.T @ lambd - beta
    return dict(zip(edges, f))


num_nodes = 10
beta = 2
alpha = 1 / 2

# Example graph creation
G = tap.random_graph(
    seed=42,
    num_edges=5,
    num_nodes=num_nodes,
    alpha=alpha,
    beta=beta,
)
G = G.to_undirected()
E = -nx.incidence_matrix(G, oriented=True)
P = np.zeros(num_nodes)
load = 1
P[2] = load
P[7] = -load

nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

ftap = tap.user_equilibrium(G, P, positive_constraint=False)
# pl.graphPlotCC(G, cc=ftap, edge_labels=dict(zip(G.edges, ftap)))


f = co.convex_optimization_kcl_tap(G, P, positive_constraint=False)
# pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)))

np.allclose(ftap, f)

# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cbar = False


# nc = dict(zip(G.nodes, nc))
for i, ax in enumerate(axs):
    if i == 0:
        f, lamb = tap.linearTAP(G, P)
        # f += 36
    elif i == 1:
        f = tap.user_equilibrium(G, P, positive_constraint=False)
        cbar = True

    # fpos = tap.user_equilibrium(G, A, positive_constraint=True)
    pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)), ax=ax, cbar=cbar)

# pl.graphPlotCC(G, cc=fpos, edge_labels=dict(zip(G.edges, fpos)))

# %%
f, lamb = tap.linearTAP(G, P)
D = positive_flow_graph(G, f)
E = -nx.incidence_matrix(D, oriented=True)
f, lamb = tap.linearTAP(D, P)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, ax in enumerate(axs):
    if i == 0:
        f, lamb = tap.linearTAP(D, P)
        cbar = False
    elif i == 1:
        f = tap.user_equilibrium(D, P, positive_constraint=False)
        cbar = True
    pl.graphPlotCC(
        D, cc=f, edge_labels=dict(zip(D.edges, f)), ax=ax, nc=lamb, cbar=cbar
    )
    print(E @ f)


# %%
positive_flow_constraint(D, lamb)
# %%
lamb
# %%
