# %%
import networkx as nx
import numpy as np
import osmnx as ox
import cvxpy as cp
import time

from src import multiCommodityTAP as mc
from src import Graphs as gr
from src import osmGraphs as og

from src import multiCommoditySocialCost as mcsc

# %%


def solve_tap(
    G, demands, social_optimum=False, pos_flows=True, alpha=None, beta=None, **kwargs
):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Additionally counts the parameters used in the problem.
    """
    print_time = kwargs.pop("print_time", False)
    return_fw = kwargs.pop("return_fw", False)

    start_time = time.time()
    A = -nx.incidence_matrix(G, oriented=True)  # Incidence matrix

    if alpha is None:
        alpha_d = nx.get_edge_attributes(G, "alpha")
        alpha = np.array(list(alpha_d.values()))

    if beta is None:
        beta_d = nx.get_edge_attributes(G, "beta")
        beta = np.array(list(beta_d.values()))

    # Number of edges
    num_edges = G.number_of_edges()

    # Number of commodities
    num_commodities = len(demands)

    # Variables for the flow on each edge for each commodity
    if pos_flows:
        flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
    else:
        flows = [cp.Variable(num_edges) for _ in range(num_commodities)]

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

    # Solve the problem
    prob.solve(**kwargs)

    # Optional: Print time
    if print_time:
        conv_time = time.time() - start_time
        print("Time:", conv_time, "s")

    # Optional: Return flows and dual variables
    if return_fw:
        fw = []
        for k, flow in enumerate(flows):
            fw.append(flow.value)
        lambda_s = [c.dual_value for c in constraints]
        return np.array(fw), np.array(lambda_s)

    return total_flow.value


def count_parameters(G, demands, alpha=None, beta=None):
    # Number of edges and nodes
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()

    # Number of commodities
    num_commodities = len(demands)

    # Alpha and Beta parameters
    if alpha is None:
        alpha_d = nx.get_edge_attributes(G, "alpha")
        alpha = np.array(list(alpha_d.values()))
    if beta is None:
        beta_d = nx.get_edge_attributes(G, "beta")
        beta = np.array(list(beta_d.values()))

    # Parameter counts
    num_alpha_beta = len(alpha) + len(beta)
    num_demand = sum(len(d) for d in demands)
    num_flow_variables = num_commodities * num_edges
    num_constraints = num_commodities * num_nodes

    total_parameters = (
        num_alpha_beta + num_demand + num_flow_variables + num_constraints
    )

    return {
        "num_alpha_beta": num_alpha_beta,
        "num_demand": num_demand,
        "num_flow_variables": num_flow_variables,
        "num_constraints": num_constraints,
        "total_parameters": total_parameters,
    }


# %%

G = og.osmGraph("Berlin,Germany")
nodes, edges = ox.graph_to_gdfs(G)
# %%
demands = og.demands(G, num_commodities=len(G), gamma=0.02)
# %%
count_parameters(G, demands)
# %%
f_mat, lambda_mat = solve_tap(
    G, demands, pos_flows=True, solver=cp.OSQP, print_time=True, return_fw=True
)

# %%


mcsc.derivative_social_cost(G, f_mat, demands, eps=1e-3)


# %%

N = 100

A = np.random.rand(N * N, N * N)

np.linalg.inv(A)
# %%
