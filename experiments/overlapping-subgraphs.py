# %%
from src import Graphs as gr
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import Plotting as pl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import cvxpy as cp

from src import Plotting as pl


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
    # flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
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
        fw = []
        for k, flow in enumerate(flows):
            fw.append(flow.value)

        lambda_s = [c.dual_value for c in constraints]
        return fw, lambda_s
        # return fw

    return total_flow.value


# %%


def subgraph_coupling_matrix(G, od_matrix):
    # Number of edges in the graph
    num_edges = G.number_of_edges()

    num_layers = len(od_matrix)

    # Identity matrix for each layer
    I_N = np.eye(num_edges)

    # Construct the interaction matrix K
    # Initialize K as a block matrix of size (num_layers * num_edges) x (num_layers * num_edges)
    K = np.zeros((num_layers * num_edges, num_layers * num_edges))

    for w in range(num_layers):
        for v in range(num_layers):
            if w != v:
                # Place -I_N in off-diagonal blocks of K
                K[
                    w * num_edges : (w + 1) * num_edges,
                    v * num_edges : (v + 1) * num_edges,
                ] = I_N

    # Construct I + K by adding the identity matrix to K
    I_plus_K = np.eye(num_layers * num_edges) + K
    return I_plus_K


def check_condition_for_edge_layer(
    edge, layer, lambdas, I_plus_K_inv, alphas, betas, num_layers, num_edges
):
    """
    Checks if the condition is satisfied for a specific edge and layer.

    Parameters:
    - edge (tuple): The edge (n, m) being checked.
    - layer (int): The subgraph layer (w) being checked.
    - lambdas (list of np.ndarray): Potentials for each layer.
    - I_plus_K_inv (np.ndarray): Inverse of (I + K) matrix.
    - alphas (dict): Dictionary with edge (n, m) keys and alpha values.
    - betas (dict): Dictionary with edge (n, m) keys and beta values.
    - edge_index (int): Index of the edge in the I_plus_K_inv matrix.
    - num_layers (int): Number of layers (W).
    - num_edges (int): Number of edges in the graph.

    Returns:
    - bool: True if the condition is fulfilled for the specified edge and layer, False otherwise.
    """
    n, m = edge
    sum_v = 0

    K_tens_inv = I_plus_K_inv.reshape(num_layers, num_layers, num_edges, num_edges)

    for v in range(num_layers):
        # Calculate lambda difference and beta term for the edge and layer
        lambda_diff = (lambdas[v][m] - lambdas[v][n]) / alphas[(n, m)]
        beta_term = betas[(n, m)] / alphas[(n, m)]

        # Compute the corresponding term in the sum for each v in relation to given layer (w)
        # term = I_plus_K_inv[
        #    edge_index + v * num_edges, edge_index + layer * num_edges
        # ] * (lambda_diff - beta_term)

        term = K_tens_inv[layer, v, n, m] * (lambda_diff - beta_term)
        sum_v += term

    # Check if the condition is satisfied for this edge and layer
    return sum_v  # True if condition is met, False otherwise


def generate_subgraph_adjacencies(
    graph,
    od_matrix,
    lambdas,
    I_plus_K_inv,
):
    """
    Generates W matrices of size num_edges x num_edges where each entry is 1 if the condition is satisfied,
    and 0 otherwise.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - num_layers (int): Number of layers (W).
    - lambdas (list of np.ndarray): Potentials for each layer.
    - I_plus_K_inv (np.ndarray): Inverse of (I + K) matrix.

    Returns:
    - condition_matrices (list of np.ndarray): List of W matrices of size num_edges x num_edges with entries 1 or 0.
    """

    alphas = nx.get_edge_attributes(graph, "alpha")
    betas = nx.get_edge_attributes(graph, "beta")
    num_edges = graph.number_of_edges()
    edge_list = list(graph.edges)

    num_layers = od_matrix.shape[0]

    num_nodes = graph.number_of_nodes()

    # Initialize W matrices of size num_edges x num_edges
    condition_matrices = [
        np.zeros((num_nodes, num_nodes), dtype=int) for _ in range(num_layers)
    ]

    # Check condition for each edge and layer
    for w in range(num_layers):
        for edge_index, edge in enumerate(edge_list):
            n, m = edge
            # Check the condition for this edge and layer
            v = check_condition_for_edge_layer(
                edge,
                w,
                lambdas,
                I_plus_K_inv,
                alphas,
                betas,
                num_layers,
                num_edges,
            )
            # If condition is satisfied, set entry to 1 in the corresponding nodes
            if v >= -1e-7:
                condition_matrices[w][n, m] = 1

    return condition_matrices


def generate_d_vec(G, lambda_w):
    alpha = nx.get_edge_attributes(G, "alpha")
    beta = nx.get_edge_attributes(G, "beta")
    d = []
    for lambda_ in lambda_w:
        d_w = np.zeros(G.number_of_edges())
        for i, e in enumerate(G.edges):
            n, m = e
            d_e_w = (lambda_[m] - lambda_[n]) / alpha[e] - beta[e] / alpha[e]
            d_w[i] = d_e_w

        d.append(d_w)
    return np.array(d).reshape(-1)


# %%
G = gr.random_graph(3, num_edges=2, beta="random", alpha="random", directed=True)
# pl.graphPlot(G)
n = 2
od_matrix = -n * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, n * (G.number_of_nodes() - 1))

od_matrix = [od_matrix[:, i] for i in range(1)]

fw, lambda_w = solve_multicommodity_tap(G, od_matrix, return_fw=True)
K = subgraph_coupling_matrix(G, od_matrix)

print(G)

f_vec = np.array(fw).reshape(-1)
d = generate_d_vec(G, lambda_w)
# %%
np.round(f_vec @ K, 2)
# %%
np.round(d, 2)
# %%
alpha = nx.get_edge_attributes(G, "alpha")
beta = nx.get_edge_attributes(G, "beta")

for i, e in enumerate(G.edges):
    v = 1 / alpha[e] * (lambda_w[0][e[1]] - lambda_w[0][e[0]]) - beta[e] / alpha[e]
    print(f"{e}: {v}")


# %%
lambda_w
# %%
print(K.shape)
K_inv = np.linalg.pinv(K)
K_inv


K_tens_inv = K_inv.reshape(
    od_matrix.shape[0], od_matrix.shape[0], G.number_of_edges(), G.number_of_edges()
)
# %%
edge_idx = 3
edge = list(G.edges)[edge_idx]
w = 0

check_condition_for_edge_layer(
    edge,
    w,
    lambda_w,
    K_inv,
    nx.get_edge_attributes(G, "alpha"),
    nx.get_edge_attributes(G, "beta"),
    od_matrix.shape[0],
    G.number_of_edges(),
)


# %%

subgraphh_adjacencies = generate_subgraph_adjacencies(G, od_matrix, lambda_w, K_inv)
np.round(subgraphh_adjacencies, 1)
# %%

g = nx.DiGraph(subgraphh_adjacencies[0])
pos = nx.get_node_attributes(G, "pos")
nx.set_node_attributes(g, pos, "pos")
pl.graphPlot(g)

# %%

num_edges = G.number_of_edges()
num_layers = od_matrix.shape[0]


kr = K.reshape(num_layers, num_layers, num_edges, num_edges)
# %%
kr[0, 0]
# %%
