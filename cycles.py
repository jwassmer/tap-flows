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
    # ttf = nx.get_edge_attributes(G, "tt_function")
    # beta = np.array([ttf[e](0) for e in G.edges()])
    # beta_dict = dict(zip(G.edges, beta))
    # nx.set_edge_attributes(D, 0, "flow")

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
    # E @ lambd
    edges = list(G.edges)
    f = E.T @ lambd - beta
    return dict(zip(edges, f))


num_nodes = 7

# Example graph creation
G = tap.random_graph(
    seed=42,
    num_edges=5,
    num_nodes=num_nodes,
    alpha=1 / 10,
    beta=50,
)
# G = G.to_undirected()
E = -nx.incidence_matrix(G, oriented=True)
A = np.zeros((num_nodes, num_nodes))
load = 1000
A[6, 0] = load
# A[np.delete(range(num_nodes), 6), 0] = -1
A[3, 0] = -load
# A[4, 0] = 1
# A[0, 0] = -1
P = A[:, 0]
nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

ftap = tap.user_equilibrium(G, A, positive_constraint=True)
# flin = np.array(list(eq.linear_flow(G, weight="kappa").values()))
pl.graphPlotCC(G, cc=ftap, edge_labels=dict(zip(G.edges, ftap)))
# tap.ODmatrix(G)

# G = positive_flow_graph(G, ftap)
# G.add_edge(4, 0, **G[0][4])

C = la.cycle_link_incidence_matrix(G)
cycle_basis = [tuple(c) for c in la.cycle_basis(G)]

f, lamb = tap.linearTAP(G, A[:, 0])
pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)))


dict(zip(cycle_basis, np.round(C @ f, 2)))


# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
cbar = False

nc = ["lightgray" for _ in range(num_nodes)]
nc[6] = "lightblue"
nc[3] = "red"
nc = dict(zip(G.nodes, nc))
for i, ax in enumerate(axs):
    if i == 0:
        f, lamb = tap.linearTAP(G, A[:, 0])
    elif i == 1:
        f = tap.user_equilibrium(G, A, positive_constraint=True)
        cbar = True

    # fpos = tap.user_equilibrium(G, A, positive_constraint=True)
    pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)), ax=ax, cbar=cbar, nc=nc)

# pl.graphPlotCC(G, cc=fpos, edge_labels=dict(zip(G.edges, fpos)))

# %%
f, lamb = tap.linearTAP(G, A[:, 0])
D = positive_flow_graph(G, f)
E = -nx.incidence_matrix(D, oriented=True)
f, lamb = tap.linearTAP(D, A[:, 0])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, ax in enumerate(axs):
    if i == 0:
        f, lamb = tap.linearTAP(D, A[:, 0])
        cbar = False
    elif i == 1:
        f = tap.user_equilibrium(D, A, positive_constraint=False)
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
