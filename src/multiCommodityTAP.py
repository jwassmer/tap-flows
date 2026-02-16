# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx

from src import ConvexOptimization as co
from src import Plotting as pl
from src._graph_utils import (
    edge_attribute_array,
    random_directed_cost_graph,
    validate_node_balance,
)


def solve_tap(
    G,
    gamma=0.1,
    num_sources="all",
    social_optimum=False,
    pos_flows=True,
    alpha=None,
    beta=None,
    **kwargs
):
    from src import osmGraphs as og

    if num_sources == "all":
        num_sources = G.number_of_nodes()

    demand_list = og.demands(G, num_sources, gamma=gamma)

    f = solve_multicommodity_tap(
        G, demand_list, social_optimum, pos_flows, alpha=alpha, beta=beta, **kwargs
    )
    nx.set_edge_attributes(G, dict(zip(G.edges, np.asarray(f).reshape(-1))), "flow")


nx.DiGraph.flows = solve_tap
nx.MultiDiGraph.flows = solve_tap


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

    # cost_funcs = nx.get_edge_attributes(G, "tt_function")
    alpha = nx.get_edge_attributes(G, "alpha")
    beta = nx.get_edge_attributes(G, "beta")
    cost_funcs = {e: lambda n: alpha[e] * n + beta[e] for e in G.edges}

    so_cost = sum([Fso_dict[e] * cost_funcs[e](Fso_dict[e]) for e in G.edges])
    ue_cost = sum([Fue_dict[e] * cost_funcs[e](Fue_dict[e]) for e in G.edges])
    # print(so_cost, ue_cost)
    return ue_cost - so_cost


def solve_multicommodity_tap(
    G, demands, social_optimum=False, pos_flows=True, alpha=None, beta=None, **kwargs
):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    start_time = time.time()
    A = -nx.incidence_matrix(G, oriented=True)  # .toarray()

    alpha = edge_attribute_array(G, "alpha", alpha)
    beta = edge_attribute_array(G, "beta", beta)

    # Number of edges
    num_edges = G.number_of_edges()

    # Number of commodities
    if isinstance(demands, np.ndarray):
        if demands.ndim == 1:
            demand_items = [demands]
        elif demands.ndim == 2:
            demand_items = [demands[i, :] for i in range(demands.shape[0])]
        else:
            raise ValueError(f"Expected 1D or 2D demand array, got shape {demands.shape}.")
    else:
        demand_items = list(demands)

    if len(demand_items) == 0:
        raise ValueError("Demands must contain at least one commodity vector.")

    demand_vectors = [validate_node_balance(G, d) for d in demand_items]
    num_commodities = len(demand_vectors)

    # Variables for the flow on each edge for each commodity
    if pos_flows:
        flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
    else:
        flows = [cp.Variable(num_edges) for _ in range(num_commodities)]

    # Combine the constraints for flow conservation
    constraints = []
    for k in range(num_commodities):
        constraints.append(A @ flows[k] == demand_vectors[k])

    if social_optimum:
        Q = 1
    elif not social_optimum:
        Q = 1 / 2

    # Objective function
    flow_matrix = (
        cp.vstack(flows)
        if num_commodities > 1
        else cp.reshape(flows[0], (1, num_edges), order="F")
    )
    total_flow = cp.sum(flow_matrix, axis=0)
    objective = cp.Minimize(
        cp.sum((cp.multiply(Q * alpha, total_flow**2)) + cp.multiply(beta, total_flow))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    # Extracting specific kwargs if provided, otherwise setting default values

    return_fw = kwargs.pop("return_fw", False)
    prob.solve(**kwargs)

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    print_time = kwargs.pop("print_time", False)
    if print_time:
        conv_time = time.time() - start_time
        print("Time:", conv_time, "s")

    if return_fw:
        fw = [np.asarray(flow.value).reshape(-1) for flow in flows]

        lambda_s = [c.dual_value for c in constraints]
        return np.array(fw), np.array(lambda_s)

    return np.asarray(total_flow.value).reshape(-1)


def random_graph(
    num_nodes=10,
    num_edges=15,
    seed=42,
    alpha=1,
    beta=0,
):
    return random_directed_cost_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        alpha=alpha,
        beta=beta,
    )


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
    pl.graphPlot(G, ec=Fue)  # , edge_labels=dict(zip(G.edges, Fue)))
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
