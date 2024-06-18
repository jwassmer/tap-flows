# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
from src import Plotting as pl
from src import ConvexOptimization as co

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


def build_graph(start_node, end_node, total_load):
    G = nx.DiGraph()
    G.source_nodes = [start_node]
    G.target_nodes = [end_node]
    G.total_load = total_load

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
            (b, c, {"tt_function": lambda n: 0}),
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

tapflow = co.convex_optimization_kcl_tap(G)
nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")

pl.graphPlotCC(G, cc=tapflow)
print(eq.total_social_cost(G, kwd="tapflow"))
tapflow

# %%

scs = []
energies = []
betas = np.linspace(0, 15, num=100)
for beta in betas:
    G.edges[("b", "c")]["tt_function"] = lambda n: beta
    tapflow = co.convex_optimization_kcl_tap(G)
    nx.set_edge_attributes(G, dict(zip(G.edges, tapflow)), "tapflow")
    social_cost = eq.total_social_cost(G, kwd="tapflow") / G.total_load
    energy = eq.total_potential_energy(G, kwd="tapflow") / G.total_load
    scs.append(social_cost)
    energies.append(energy)


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(betas, scs, label="Social Cost")
ax.plot(betas, energies, label="Potential Energy")
ax.legend()
ax.grid()
ax.set_xlabel("Beta")
# %%

import cvxpy as cp

start_node, end_node = "a", "d"
total_load = 1000
G = build_graph(start_node, end_node, total_load)


# Get the number of edges and nodes
num_edges = G.number_of_edges()

# Create the edge incidence matrix E
E = -nx.incidence_matrix(G, oriented=True).toarray()

tt_funcs = nx.get_edge_attributes(G, "tt_function")
betas = np.array([tt_funcs[e](0) for e in G.edges()])
alphas = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges()])

P = list(nx.get_node_attributes(G, "P").values())

# Define the flow variable f_e
f = cp.Variable(num_edges)

# Objective function: minimize sum of (f_e^2 / (2 * K_e))
# objective = cp.Minimize(alphas @ f**2 / 2 + betas @ f)
objective = cp.Minimize(cp.sum(cp.multiply(f**2, alphas) + cp.multiply(f, betas)))

# Constraints: E @ f = p
constraints = [E @ f == P, f >= np.zeros(num_edges)]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Get the results
flow_values = f.value

nx.set_edge_attributes(G, dict(zip(G.edges, flow_values)), "flow")
print(eq.total_social_cost(G, kwd="flow") / G.total_load)
flow_values
# %%

tapflow / flow_values
# %%


# %%
