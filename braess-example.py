# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
from src import Plotting as pl

pl.mpl_params(fontsize=32)


# %%
def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    nash_flow = nx.get_edge_attributes(G, "load")
    for e, f in lambda_funcs.items():
        beta = f(0)
        alpha = f(1) - beta
        x = nash_flow[e]
        if x != 0:
            K = x / (2 * beta + alpha * x)
        else:
            K = 0
        G.edges[e]["weight"] = K
    return G


def linear_function(alpha, beta, x):
    return alpha + beta * x


def build_graph(start_node, end_node, total_load):
    G = nx.DiGraph()

    a, b, c, d, e, f = "a", "b", "c", "d", "e", "f"

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0.5)}),
            (b, {"pos": (0.5, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 0.5)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: n / 100 + 10}),
            (b, d, {"tt_function": lambda n: 20}),
            (a, c, {"tt_function": lambda n: 20}),
            (c, d, {"tt_function": lambda n: n / 100 + 10}),
            (b, c, {"tt_function": lambda n: 1}),
        ]
    )

    nodes_dict = dict(zip(G.nodes, [i for i in range(G.number_of_nodes())]))
    P = np.zeros(G.number_of_nodes())

    start_node_int = nodes_dict[start_node]
    end_node_int = nodes_dict[end_node]

    P[start_node_int], P[end_node_int] = total_load, -total_load

    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    # G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    # for s in start_node:
    G.nodes[start_node]["color"] = "lightblue"
    # for e in end_node:
    G.nodes[end_node]["color"] = "red"

    planar = nx.check_planarity(G)
    if planar[0]:
        print("The graph is planar")
    else:
        print("The graph is NOT planar")

    return G


# %%

start_node, end_node = "a", "d"
total_load = 1000
G = build_graph(start_node, end_node, total_load)
G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)

G = updateEdgeWeights(G)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")


print("Total potential energy, linflow: ", eq.total_potential_energy(G, "flow"))
print("Total potential energy, nash TAP: ", eq.total_potential_energy(G, "load"))
print(eq.total_potential_energy(G, "load") <= eq.total_potential_energy(G, "load"))
print("The Social Cost is: ", eq.total_social_cost(G, "load"))


# %%

fig, ax = plt.subplots(1, figsize=(6, 4))
pl.graphPlot(G, ax=ax)
ax.annotate(
    r"$\textbf{(a)}$",
    xy=(0.15, 0.85),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
)

ax = fig.add_axes([1, 0.1, 0.33, 0.8])
ax.scatter(range(len(Utot)), Utot, label=r"$U_{tot}$")
ax.scatter(range(len(SCtot)), SCtot, label=r"$SC_{tot}$")
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xlabel("Iteration", fontsize=12)
ax.legend(fontsize=12)
ax.grid()
ax.ticklabel_format(useOffset=False)
ax.yaxis.tick_right()
ax.annotate(
    r"$\textbf{(b)}$",
    xy=(0.02, 1.03),
    xycoords="axes fraction",
    fontsize=16,
    fontweight="bold",
)


# fig.savefig("figs/braess-example.pdf", bbox_inches="tight")


# %%
