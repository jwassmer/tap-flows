# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx


def solve_multicommodity_tap(G, demands):
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

    # Objective function
    total_flow = cp.sum(flows)
    objective = cp.Minimize(
        cp.sum(cp.multiply(alpha, total_flow**2))
        + cp.sum(cp.multiply(beta, total_flow))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    conv_time = time.time() - start_time
    print("Time:", conv_time, "s")

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

    num_nodes = 1000
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
    P0[2], P0[7] = load, -load
    P1 = np.zeros(num_nodes)
    P1[5], P1[2] = load, -load

    demands = [P0, P1]
    # demands = [P0 + P1]
    F = solve_multicommodity_tap(G, demands)
    # pl.graphPlotCC(G, cc=F)  # , edge_labels=dict(zip(G.edges, F)))

    # WRITE TEST


# %%
