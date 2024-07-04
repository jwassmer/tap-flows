# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src import TAPOptimization as tap
from src import Plotting as pl
from src import ConvexOptimization as co
from src import Equilibirium as eq
from src import LinAlg as la


import networkx as nx
import matplotlib.pyplot as plt
import random


def create_dag_from_graph(graph, start_node):
    C = graph.copy()

    def dfs(node, visited, stack):
        visited.add(node)
        stack.add(node)
        edges_to_remove = []
        for neighbor in list(C.neighbors(node)):
            if neighbor not in visited:
                dfs(neighbor, visited, stack)
            elif neighbor in stack:
                # If neighbor is in stack, it means there's a cycle. Mark the edge for removal.
                edges_to_remove.append((node, neighbor))
        stack.remove(node)
        for edge in edges_to_remove:
            C.remove_edge(*edge)

    visited = set()
    stack = set()
    dfs(start_node, visited, stack)
    return C


num_nodes = 10
beta = "random"
alpha = "random"

# Example graph creation
G = tap.random_graph(
    seed=42,
    num_edges=5,
    num_nodes=num_nodes,
    alpha=alpha,
    beta=beta,
)
P = np.zeros(num_nodes)
load = 1000
source = 5
P[source] = load
target_nodes = np.delete(np.arange(num_nodes), source)
P[target_nodes] = -load / len(target_nodes)

D = create_dag_from_graph(G, source)
# G = G.to_undirected()
E = -nx.incidence_matrix(G, oriented=True)


nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

ftap = tap.user_equilibrium(G, P, positive_constraint=False)
# pl.graphPlotCC(G, cc=ftap, edge_labels=dict(zip(G.edges, ftap)))


f, lamb = tap.linearTAP(G, P)
# pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)))

print(np.allclose(ftap, f))

E @ (f)


# %%
f, lamb = tap.linearTAP(G, P)
# D = positive_flow_graph(G, f)
E = -nx.incidence_matrix(D, oriented=True)
f, lamb = tap.linearTAP(D, P)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, ax in enumerate(axs):
    if i == 0:
        fg = tap.user_equilibrium(G, P, positive_constraint=True)
        cbar = False
        pl.graphPlotCC(G, cc=fg, edge_labels=dict(zip(G.edges, fg)), ax=ax, cbar=cbar)
    elif i == 1:
        fd = tap.user_equilibrium(D, P, positive_constraint=False)
        cbar = True
        pl.graphPlotCC(D, cc=fd, edge_labels=dict(zip(D.edges, fd)), ax=ax, cbar=cbar)
    # print(E @ f)


Cd = la.cycle_link_incidence_matrix(D)
Cd @ fd

# %%

fdict = dict(zip(D.edges(), fd))  # Note the use of D.edges() without data=True

edges_to_reverse = []

for e in D.edges(data=True):
    edge = (e[0], e[1])  # Extract the edge (u, v) from the tuple (u, v, data)
    f = fdict[edge]
    if f < 0:
        edges_to_reverse.append((edge, e[2]))  # Collect the edge and its data

# Perform the modifications after collecting edges to reverse
for edge, edge_data in edges_to_reverse:
    D.remove_edge(*edge)
    D.add_edge(edge[1], edge[0], **edge_data)


fd = tap.user_equilibrium(D, P, positive_constraint=False)
fd_so = tap.social_optimum(D, P, positive_constraint=False)
pl.graphPlotCC(D, cc=fd, edge_labels=dict(zip(D.edges, fd)))


# %%
E = -nx.incidence_matrix(D, oriented=True)
tt_func = nx.get_edge_attributes(D, "tt_function")
beta = np.array([tt_func[e](0) for e in D.edges()])
alpha = np.array([tt_func[e](1) - tt_func[e](0) for e in D.edges()])
nx.set_edge_attributes(D, dict(zip(D.edges, beta / alpha)), "gamma")
nx.set_edge_attributes(D, dict(zip(D.edges, 1 / alpha)), "kappa")

A = nx.adjacency_matrix(D, weight="gamma")
J = nx.laplacian_matrix(D, weight="kappa")
Gamma = A - A.T
L = (J + J.T).toarray()
np.fill_diagonal(L, 0)
np.fill_diagonal(L, -np.sum(L, axis=0))
one = np.ones(D.number_of_nodes())

E.T @ np.linalg.pinv(L) @ Gamma @ one - beta
# %%


dict(zip(D.edges, fd - fd_so))
# %%
