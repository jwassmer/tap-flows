# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
import itertools


# %%


G1 = nx.DiGraph()

G1.add_nodes_from(
    [
        (0, {"pos": (0, 0)}),
        (1, {"pos": (0.25, 0.5)}),
        (2, {"pos": (0.5, 0.25)}),
        (3, {"pos": (1, 0.25)}),
    ]
)

G1.add_edges_from(
    [
        (0, 1, {"tt_function": lambda n: n}),
        (1, 2, {"tt_function": lambda n: n}),
        (1, 3, {"tt_function": lambda n: n}),
        (0, 3, {"tt_function": lambda n: n}),
        (2, 3, {"tt_function": lambda n: n}),
    ]
)

G2 = nx.DiGraph()
G2.add_nodes_from(
    [
        (0, {"pos": (0, 0.5)}),
        (1, {"pos": (0.5, 0)}),
        (2, {"pos": (0.5, 1)}),
        (3, {"pos": (1, 0.5)}),
    ]
)

G2.add_edges_from(
    [
        (0, 1, {"tt_function": lambda n: 12}),
        (0, 2, {"tt_function": lambda n: n / 100}),
        (1, 3, {"tt_function": lambda n: n / 100}),
        (2, 3, {"tt_function": lambda n: 12}),
        (2, 1, {"tt_function": lambda n: 2}),
    ]
)


start_node = [0]
end_node = 3

total_load = 1000

# %%
"""G = G2

for order in itertools.permutations(start_node):
    print(order)
    G = eq.assign_initial_loads(G, order, end_node, total_load)
    G, T_arr, sc_arr = eq.user_equilibrium(G, order, end_node)
    print(T_arr[-1])"""


# %%
G = G2
# G.remove_node(2)
# G.edges[(2, 3)]["tt_function"] = lambda n: alpha * n
G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, T_arr, sc_arr = eq.user_equilibrium(G, start_node, end_node)
nx.get_edge_attributes(G, "load")

eq.total_social_cost(G, "load")
# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(T_arr, marker="o", label="Total Potential Energy")
axs[1].plot(sc_arr, marker="o", label="Total Social Cost")
for a in axs:
    a.set_xlabel("Iteration")
    a.legend()
    a.grid()


fig = plt.figure()

pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos)
labels = nx.get_edge_attributes(G, "load")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


# %%
