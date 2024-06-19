# %%
import numpy as np
import networkx as nx
from src import TAPOptimization as tap
from src import Plotting as pl
from src import ConvexOptimization as co


def positive_flow_graph(G, F):
    Fdict = dict(zip(G.edges, F))
    D = nx.DiGraph()
    nx.set_edge_attributes(D, 0, "flow")
    for n, d in G.nodes(data=True):
        D.add_node(n, **d)
    for i, j, d in G.edges(data=True):
        flow = Fdict[(i, j)]
        if flow >= 0:
            D.add_edge(i, j, **d)
            D[i][j]["flow"] = flow
        else:
            D.add_edge(j, i, **d)
            D[j][i]["flow"] = -flow
    return D


# %%
num_nodes = 20

# Example graph creation
G = tap.random_graph(
    seed=42,
    num_edges=5,
    num_nodes=num_nodes,
    alpha="random",
    beta="random",
)
E = -nx.incidence_matrix(G, oriented=True)
P = np.zeros(G.number_of_nodes())
A = np.zeros((num_nodes, num_nodes))
A[0, 0] = 100
A[1:, 0] = -A[0, 0] / (num_nodes - 1)
tfs = nx.get_edge_attributes(G, "tt_function")
alphas = np.array([tfs[e](1) - tfs[e](0) for e in G.edges()])


fpos = tap.user_equilibrium(G, A, positive_constraint=True)
pl.graphPlotCC(G, cc=fpos, norm="Normalize")

f = tap.user_equilibrium(G, A, positive_constraint=False)
pl.graphPlotCC(G, cc=f)
# tap.ODmatrix(G)
dict(zip(G.edges, f))

# %%
g = G.to_undirected()
flin = tap.linearTAP(g, A[:, 0])
ftap = tap.user_equilibrium(g, A, positive_constraint=False)
delta = np.abs(flin - ftap)
pl.graphPlotCC(g, cc=flin)
pl.graphPlotCC(g, cc=ftap)
E = -nx.incidence_matrix(g, oriented=True)
np.abs(flin - ftap) < 1e-5
# %%

np.round(E @ flin, 2)
# %%


flin = tap.linearTAP(G, A[:, 0])
ftap = tap.user_equilibrium(G, A, positive_constraint=False)
delta = np.abs(flin - ftap)

E = -nx.incidence_matrix(G, oriented=True)
np.abs(flin - ftap) < 1e-5

E @ ftap
# %%
