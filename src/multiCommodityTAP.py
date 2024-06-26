# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx

from src import ConvexOptimization as co
from src import Plotting as pl


def price_of_anarchy(G, demands):
    """
    Compute the price of anarchy for a given graph and demands.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    Fso = solve_multicommodity_tap(G, demands, social_optimum=True)
    Fue = solve_multicommodity_tap(G, demands, social_optimum=False)

    Fso_dict = dict(zip(G.edges, Fso))
    Fue_dict = dict(zip(G.edges, Fue))

    cost_funcs = nx.get_edge_attributes(G, "tt_function")
    if isinstance(G, nx.MultiDiGraph):
        so_cost = sum(
            [Fso_dict[e] * cost_funcs[e](Fso_dict[e]) for e in G.edges(keys=True)]
        )
        ue_cost = sum(
            [Fue_dict[e] * cost_funcs[e](Fue_dict[e]) for e in G.edges(keys=True)]
        )
    else:
        so_cost = sum([Fso_dict[e] * cost_funcs[e](Fso_dict[e]) for e in G.edges()])
        ue_cost = sum([Fue_dict[e] * cost_funcs[e](Fue_dict[e]) for e in G.edges()])
    # print(so_cost, ue_cost)
    return ue_cost - so_cost


def solve_multicommodity_tap(G, demands, social_optimum=False):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    start_time = time.time()
    A = -nx.incidence_matrix(G, oriented=True).toarray()

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    if isinstance(G, nx.MultiDiGraph):
        beta = np.array([tt_funcs[e](0) for e in G.edges(keys=True)])
        alpha = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges(keys=True)])
    else:
        beta = np.array([tt_funcs[e](0) for e in G.edges()])
        alpha = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges()])

    # Number of edges
    num_edges = G.number_of_edges()

    # Number of commodities
    num_commodities = len(demands)

    # Variables for the flow on each edge for each commodity
    flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
    # flows = cp.Variable((num_commodities, num_edges))  # , nonneg=True)
    # Combine the constraints for flow conservation
    constraints = []
    for k in range(num_commodities):
        constraints.append(A @ flows[k] == demands[k])

    if social_optimum:
        Q = 1
    elif not social_optimum:
        Q = 1 / 2

    # Objective function
    total_flow = cp.sum(flows)
    objective = cp.Minimize(
        cp.sum((cp.multiply(Q * alpha, total_flow**2)) + cp.multiply(beta, total_flow))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve(eps_rel=1e-7)

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    conv_time = time.time() - start_time
    # print("Time:", conv_time, "s")

    return total_flow.value


def random_graph(
    num_nodes=10,
    num_edges=15,
    seed=42,
    alpha=1,
    beta=0,
):
    connected = False
    if num_edges < num_nodes - 1:
        num_edges = num_nodes - 1

    while not connected:
        U = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
        connected = nx.is_connected(U)
        num_edges += 1

    G = U.to_directed()

    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(G.number_of_edges())

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())

    tt_func = {
        edge: (lambda alpha, beta: lambda n: alpha * n + beta)(alpha[e], beta[e])
        for e, edge in enumerate(G.edges)
    }
    nx.set_edge_attributes(G, tt_func, "tt_function")

    pos = nx.spring_layout(G, seed=seed)
    nx.set_node_attributes(G, pos, "pos")

    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


# %%
if __name__ == "__main__":

    num_nodes = 50
    num_edges = int(num_nodes * 1.5)
    # Example graph creation
    G = random_graph(
        seed=42,
        num_nodes=num_nodes,
        num_edges=num_edges,
        alpha=1,
        beta=3,
    )
    load = 1000
    P0 = np.zeros(num_nodes)
    P0[13], P0[14] = load, -load
    P1 = np.zeros(num_nodes)
    # P1[10], P1[7] = load, -load

    demands = [P0, P1]
    Fso = solve_multicommodity_tap(G, demands, social_optimum=True)
    Fue = solve_multicommodity_tap(G, demands, social_optimum=False)
    pl.graphPlotCC(G, cc=Fue)  # , edge_labels=dict(zip(G.edges, Fue)))
    print(price_of_anarchy(G, demands))
    # %%

    P = P0 + P1
    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")
    Fco = co.convex_optimization_kcl_tap(G)
    pl.graphPlotCC(G, cc=Fco)  # , edge_labels=dict(zip(G.edges, Fco)))
    print(np.round(Fue - Fco, 5))
    # WRITE TEST

    # %%
    def OD_matrix(G):
        num_nodes = G.number_of_nodes()
        A = -np.ones((num_nodes, num_nodes))
        np.fill_diagonal(A, num_nodes - 1)
        return A

    A = OD_matrix(G)
    demands = [A[:, i] for i in range(num_nodes)]
    Fue = solve_multicommodity_tap(G, demands)
    Fso = solve_multicommodity_tap(G, demands, social_optimum=True)

    print(price_of_anarchy(G, demands))
    # %%
