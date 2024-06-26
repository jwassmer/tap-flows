# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
from src import Plotting as pl
from src import ConvexOptimization as co
from src import multiCommodityTAP as mc

pl.mpl_params(fontsize=14)


# %%
def get_alpha(G):
    alpha = np.array([G[u][v]["tt_function"](0) for u, v in G.edges()])
    return dict(zip(G.edges(), alpha))


def get_beta(G):
    beta = np.array(
        [G[u][v]["tt_function"](1) - G[u][v]["tt_function"](0) for u, v in G.edges()]
    )
    return dict(zip(G.edges(), beta))


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


def braessGraph():
    G = nx.DiGraph()

    a, b, c, d = 1, 2, 3, 4

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
            (b, d, {"tt_function": lambda n: 25}),
            (a, c, {"tt_function": lambda n: 25}),
            (c, d, {"tt_function": lambda n: n / 100 + 10}),
            (b, c, {"tt_function": lambda n: 10}),
        ]
    )

    # nodes_dict = dict(zip(G.nodes, [i for i in range(G.number_of_nodes())]))
    # P = np.zeros(G.number_of_nodes())

    # start_node_int = nodes_dict[start_node]
    # end_node_int = nodes_dict[end_node]

    # P[start_node_int], P[end_node_int] = total_load, -total_load

    # nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    # G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    # for s in start_node:
    # G.nodes[start_node]["color"] = "lightblue"
    # for e in end_node:
    # G.nodes[end_node]["color"] = "red"

    return G


# %%


G = braessGraph()
G.add_edge(3, 2, **G[2][3])
load = 500

P = [load, 0, 0, -load]
demand = [P]


tapflow = mc.solve_multicommodity_tap(G, demand, social_optimum=False)
nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")

pl.graphPlotCC(G, cc=tapflow, edge_labels=dict(zip(G.edges, np.round(tapflow, 2))))
# print(eq.total_social_cost(G, kwd="tapflow") / load)
print(mc.price_of_anarchy(G, demand))
# %%
fig, ax = plt.subplots(figsize=(6, 4))

betas = np.linspace(0, 15, num=100)
labels = ["Social Optimum", "User Equilibrium"]
for i, social_optimum in enumerate([True, False]):
    scs = []
    energies = []
    for beta in betas:
        G.edges[(2, 3)]["tt_function"] = lambda n: beta
        tapflow = mc.solve_multicommodity_tap(G, demand, social_optimum=social_optimum)
        nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")
        social_cost = eq.total_social_cost(G, kwd="tapflow") / load
        # energy = eq.total_potential_energy(G, kwd="tapflow") / load
        scs.append(social_cost)
        # energies.append(energy)

    G.remove_edges_from([(2, 3)])
    tapflow = mc.solve_multicommodity_tap(G, demand, social_optimum=social_optimum)
    nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")
    social_cost = eq.total_social_cost(G, kwd="tapflow") / load
    G.add_edge(2, 3, tt_function=lambda n: beta)

    ax.plot(betas, scs, label=labels[i])
    plt.scatter(beta, [social_cost], color="red")

ax.set_ylabel("Social Cost")
ax.set_xlabel("Beta")
ax.legend()
ax.grid()

# %%
