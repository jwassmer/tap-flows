# %%
import networkx as nx
import numpy as np
import pytest
import cvxpy as cp

from src import SIBC as sibc
from src import multiCommodityTAP as mc
from src import Graphs as gr
from src import TAPOptimization as tap


# %%
num_nodes = 30
num_edges = 15

G = gr.random_graph(
    seed=42,
    num_nodes=num_nodes,
    num_edges=num_edges,
    alpha=0,
    beta="random",
)
betas = np.random.rand(G.number_of_edges())
beta_dict = dict(zip(G.edges, betas))
nx.set_edge_attributes(G, beta_dict, "beta")

num_nodes = G.number_of_nodes()
od_matrix = -np.random.rand(num_nodes, num_nodes)
np.fill_diagonal(od_matrix, 0)
np.fill_diagonal(od_matrix, -np.sum(od_matrix, axis=0))


s = sibc._interaction_betweenness_centrality(
    G, weight="beta", od_matrix=np.abs(od_matrix)
)

ebc = list(nx.edge_betweenness_centrality(G, weight="beta", normalized=False).values())

print(np.all(np.isclose(s, ebc)))
####WRITE TESTS
# %%

demands = [od_matrix[:, i] for i in range(num_nodes)]
###if alpha 0 shortest path limit equal to ebc
fe = mc.solve_multicommodity_tap(G, demands)

print(np.all(np.isclose(fe, s)))
# %%


G = gr.random_graph(
    seed=42,
    num_nodes=10,
    num_edges=num_edges,
    alpha=0,
    beta="random",
)

P = -np.ones(G.number_of_nodes())
P[0] = np.sum(np.abs(P)) - 1

s = sibc._single_source_interaction_betweenness_centrality(G, weight="beta", P=P)

f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS)

f - s

# %%
fl = tap.linearTAP(G, P)
# %%
