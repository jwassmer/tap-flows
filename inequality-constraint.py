# %%
import numpy as np
import networkx as nx
import time
from src import TAPOptimization as tap
from src import Plotting as pl
import cvxpy as cp
import matplotlib.pyplot as plt

from src import sibc

# %%

G = tap.random_graph(
    seed=42,
    num_edges=10,
    num_nodes=5000,
    alpha="random",
    beta="random",
)

# G = braessGraph()

E = -nx.incidence_matrix(G, oriented=True).toarray()
P = np.zeros(G.number_of_nodes())
load = 500
source = 0
P[source] = load
targets = np.delete(np.arange(G.number_of_nodes()), source)
P[targets] = -load / len(targets)
demands = P

# %%
s = sibc._single_commodity_interaction_betweenness_centrality(G, weight="beta", P=P)

# E @ s - P


# %%

f_ue = tap.user_equilibrium(G, P, positive_constraint=True)
f_ue - s

# pl.graphPlotCC(G, cc=f_ue)
# %%
start_time = time.time()
E = -nx.incidence_matrix(G, oriented=True)  # .toarray()

alpha_d = nx.get_edge_attributes(G, "alpha")
beta_d = nx.get_edge_attributes(G, "beta")
alpha = np.array(list(alpha_d.values()))
beta = np.array(list(beta_d.values()))

num_edges = G.number_of_edges()

flows = cp.Variable(num_edges, nonneg=True)

constraints = [E @ flows == demands]

# Objective function
objective = cp.Minimize(cp.sum(cp.multiply(beta, flows)))

# Define the problem and solve it
prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.SCS)


# pl.graphPlotCC(G, cc=flows.value)
flows.value - s
# %%

E @ flows.value

# %%
tree = nx.dijkstra_predecessor_and_distance(G, source=0, weight="beta")

predecessors, distances = tree

T = nx.DiGraph()  # Create a directed graph for the spanning tree

# Add the edges from the predecessors dictionary
for node, preds in predecessors.items():
    for pred in preds:
        T.add_edge(pred, node, weight=G[pred][node]["beta"])

# %%
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))

# Draw the edges of G
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"))

# Draw the edges of T (spanning tree) with a different color
nx.draw_networkx_edges(T, pos, edge_color="red", width=3)

# Draw the nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

# %%
