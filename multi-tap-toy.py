# %%
from src import osmGraphs as og
import cvxpy as cp
import time
import networkx as nx
import numpy as np
import pickle


# %%
def tap_solver(G, demands, **kwargs):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    start_time = time.time()
    E = -nx.incidence_matrix(G, oriented=True)

    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha = np.array(list(alpha_d.values()))
    beta = np.array(list(beta_d.values()))

    # Number of edges
    num_edges = G.number_of_edges()

    # Number of commodities
    num_origins = len(demands)

    # Variables for the flow on each edge for each commodity
    # if pos_flows:
    nonneg = kwargs.pop("nonneg", False)
    flows = [cp.Variable(num_edges, nonneg=nonneg) for _ in range(num_origins)]

    constraints = []
    for k in range(num_origins):
        constraints.append(E @ flows[k] == demands[k])

    Q = 1 / 2

    # Objective function
    total_flow = cp.sum(flows)
    objective = cp.Minimize(
        cp.sum((cp.multiply(Q * alpha, total_flow**2)) + cp.multiply(beta, total_flow))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    # Extracting specific kwargs if provided, otherwise setting default values

    prob.solve(**kwargs)

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    conv_time = time.time() - start_time
    print("Time:", conv_time, "s")

    return total_flow.value


# %%
G = og.osmGraph(
    "Cologne, Germany",
    heavy_boundary=True,
    buffer_meter=10_000,
)

# Save graph as pickles
# with open("data/vienna_graph.pkl", "wb") as f:
#    pickle.dump(G, f)

# Read graph from pickle
# with open("data/vienna_graph.pkl", "rb") as f:
#    G = pickle.load(f)

n = len(G.nodes)

# assume every node has equal demand for testing purposes
# in reality chose according to population data (given as node attribute)
demand_matrix = np.full((n, n), -1)
np.fill_diagonal(demand_matrix, (n - 1) * np.ones(G.number_of_nodes()))
demand_list = [demand_matrix[i] for i in range(n)]

demands = demand_list[:10]


# %%
#
# PLAY AROUND WITH SIZE OF DEMAND LIST.
# ON MY LOCAL MACHINE AT len(demand_list) = 25 IT ALREADY TAKES A FEW MINUTES TO SOLVE.
#
# tap_solver(G, demand_list, verbose=True, nonneg=True, solver=cp.MOESK)
# fmosek = tap_solver(G, demand_list[:20], verbose=True, solver=cp.MOSEK, nonneg=True)
fosqp = tap_solver(G, demands, verbose=True, solver=cp.OSQP, nonneg=True)
# %%
fosqp
# %%

variables = len(G.edges) * len(demands)
print("Variables:", variables)
# %%
constraints = len(G.nodes) * len(demands)
print("Constraints:", constraints)
# %%
