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
from scipy.linalg import block_diag

np.set_printoptions(precision=3, suppress=True)

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
        return np.array(fw), np.array(lambda_s)
        # return fw

    return total_flow.value


# %%


def subgraph_incidence_matrix(G, few):
    """
    Create a coupling matrix for subgraphs based on edge existence.

    Parameters:
    G : NetworkX graph
        The original graph.
    lambda_w : list
        List of thresholds for each subgraph layer.

    Returns:
    np.ndarray
        The coupling matrix K with shape (num_layers, num_layers, num_edges, num_edges).
    """
    # Number of edges in the graph
    num_edges = G.number_of_edges()
    num_layers = len(few)

    # num_unique_layer_combinations = num_layers * (num_layers - 1) // 2

    # Initialize the coupling matrix
    K = np.zeros((num_layers, num_edges), dtype=int)

    # List of edges in the graph for indexing
    edges = list(G.edges)

    for w in range(num_layers):
        Uw = potential_subgraph(G, few[w])  # edges in subgraph w

        for e_idx, edge in enumerate(edges):
            # Check if the edge exists in both subgraphs
            if edge in Uw.edges:
                K[w, e_idx] = 1  # few[w][e_idx]  # (
            # if edge in Uw.in_edges:
            #    K[w, e_idx] = -1

        # k += 1

    # column_sums = np.sum(K, axis=0)
    # Divide only if the column sum is nonzero
    # K = np.divide(K, column_sums, where=column_sums != 0)
    return K


def subgraph_coupling_matrix(G, few):
    """
    Create a coupling matrix for subgraphs based on edge existence.

    Parameters:
    G : NetworkX graph
        The original graph.
    lambda_w : list
        List of thresholds for each subgraph layer.

    Returns:
    np.ndarray
        The coupling matrix K with shape (num_layers, num_layers, num_edges, num_edges).
    """
    # Number of edges in the graph
    num_edges = G.number_of_edges()
    num_layers = len(few)

    # num_unique_layer_combinations = num_layers * (num_layers - 1) // 2

    # Initialize the coupling matrix
    K = np.zeros((num_layers, num_edges, num_layers, num_edges), dtype=int)

    # List of edges in the graph for indexing
    edges = list(G.edges)

    for w in range(num_layers):
        for v in range(num_layers):
            # if v != w:
            # Generate the subgraphs U_v and U_w based on thresholds lambda_w[v] and lambda_w[w]
            Uv = potential_subgraph(G, few[v])  # edges in subgraph v
            Uw = potential_subgraph(G, few[w])  # edges in subgraph w

            for e_idx, edge in enumerate(edges):
                # Check if the edge exists in both subgraphs
                if edge in Uw.edges and edge in Uv.edges:
                    K[w, e_idx, v, e_idx] = 1  # (
                    # Uw.edges[edge]["flow"] / Uv.edges[edge]["flow"]
                    # )
                    # print(f"Edge {edge} exists in both subgraphs {v} and {w}")
                    # print(Uv.edges[edge]["flow"])
                    # print(Uw.edges[edge]["flow"])

        # k += 1

    return K.reshape(num_layers * num_edges, num_layers * num_edges)


def potential_subgraph(G, few):
    U = G.copy()

    for i, e in enumerate(G.edges):
        if few[i] < 1e-2:
            U.remove_edge(*e)
            # A[n, m] = 1
        else:
            U.edges[e]["flow"] = few[i]
    return U


def generate_d_vec0(G, lambda_w, few):
    alpha = nx.get_edge_attributes(G, "alpha")
    beta = nx.get_edge_attributes(G, "beta")
    d = []
    for k, lambda_ in enumerate(lambda_w):
        U = potential_subgraph(G, few[k])
        d_w = np.zeros(G.number_of_edges())
        for i, e in enumerate(G.edges):
            if e in U.edges:
                n, m = e
                d_e_w = (lambda_[m] - lambda_[n]) / alpha[e] - beta[e] / alpha[e]
                d_w[i] = d_e_w

        d.append(d_w)
    return np.array(d)  # .reshape(-1)


def sub_incidence_matrix(G, few):
    U = potential_subgraph(G, few)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    for n in range(E.shape[0]):
        for e in range(E.shape[1]):
            edg = list(G.edges)[e]
            if edg not in U.edges:
                E[n, e] = 0
    return E


def delta_incidence_matrix(G, fe0, fe1):
    # num_edges = G.number_of_edges()
    E0 = sub_incidence_matrix(G, fe0)
    E1 = sub_incidence_matrix(G, fe1)
    return E0 - E1


def generate_d_vec(G, lambda_w, few):
    alpha_vec = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta_vec = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    d = []
    for k, lambda_ in enumerate(lambda_w):
        fk = few[k]
        E = sub_incidence_matrix(G, fk)  # .toarray()

        d_w = -E.T @ lambda_ / alpha_vec - beta_vec / alpha_vec

        d.append(d_w)
    return np.array(d)  # .reshape(-1)


def subgraph_mattrices(G, few):
    U = potential_subgraph(G, few)
    alpha_vec = np.array(list(nx.get_edge_attributes(U, "alpha").values()))
    beta_vec = np.array(list(nx.get_edge_attributes(U, "beta").values()))

    E = -nx.incidence_matrix(U, oriented=True)

    kappa = 1 / alpha_vec
    nx.set_edge_attributes(U, dict(zip(U.edges, kappa)), "kappa")
    L = nx.laplacian_matrix(U, weight="kappa").toarray()

    gamma = beta_vec / alpha_vec
    nx.set_edge_attributes(U, dict(zip(U.edges, gamma)), "gamma")
    A = nx.adjacency_matrix(U, weight="gamma").toarray()

    Gamma = A - A.T
    L = E @ np.diag(kappa) @ E.T

    return L, Gamma


def full_matrices(G, f_mat, lambda_mat):
    num_layers = len(f_mat)
    d = []
    for i in range(num_layers):
        lamb = lambda_mat[i]
        L_i, Gamma_i = subgraph_mattrices(G, f_mat[i])

        d_i = L_i @ lamb + Gamma_i @ np.ones(G.number_of_nodes())
        d.append(d_i)
    return np.array(d)  # .reshape(-1)


# %%

G = gr.random_graph(4, num_edges=5, beta=0, alpha=1, directed=True)
alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
print(G)
num_edges = G.number_of_edges()

num_layers = 2  # len(G.nodes)
n = 10
od_matrix = -n * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, n * (G.number_of_nodes() - 1))

od_matrix = [od_matrix[:, i] for i in range(num_layers)]

f_mat, lambda_mat = solve_multicommodity_tap(
    G, od_matrix, return_fw=True, solver=cp.OSQP
)
f_vec = np.array(f_mat).reshape(-1)
F = solve_multicommodity_tap(G, od_matrix, return_fw=False)

E = -nx.incidence_matrix(G, oriented=True).toarray()

K = subgraph_coupling_matrix(G, f_mat)
d = generate_d_vec0(G, lambda_mat, f_mat)
# generate_d_vec(G, lambda_mat, f_mat) # .reshape(-1)
# d.reshape(num_layers, num_edges)

S = subgraph_incidence_matrix(G, f_mat)
# pl.graphPlot(G)

# %%


result = np.zeros_like(f_mat)
for i in range(S.shape[0]):
    result[i] = sum(f_mat[j] * S[i] for j in range(f_mat.shape[0]))

result = np.sum(f_mat[:, None, :] * S[None, :, :], axis=0)

np.allclose(result, d)

# %%
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs = axs.flatten()

cmap = plt.get_cmap("viridis")
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1, vmax=15)
for w in range(num_layers):
    Uw = potential_subgraph(G, f_mat[w])  # edges in subgraph w
    fw = nx.get_edge_attributes(Uw, "flow")
    pl.graphPlot(Uw, ec=fw, cmap=cmap, norm=norm, ax=axs[w], edge_labels=fw, title=None)

print(G.edges)
# f_mat
f_mat

# %%

np.round(S * d - f_mat, 3)

# %%
Ktimesf = K @ f_mat.reshape(-1)
d_vec = d.reshape(-1)

# %%
if not np.allclose(d_vec, Ktimesf, atol=1e-7):
    print(np.max(np.abs(d_vec - Ktimesf)))
else:
    print("Equal")

# %%


def generate_delta(G, lambda_w, few):

    d = []
    for w in range(len(few)):
        U = potential_subgraph(G, few[w])
        alpha = nx.get_edge_attributes(U, "alpha")
        beta = nx.get_edge_attributes(U, "beta")
        lambda_ = lambda_w[w]
        d_n = np.zeros(U.number_of_nodes())
        for n in U.nodes:
            for e in U.out_edges(n):
                m = e[1]
                d_n[n] += (lambda_[n] - lambda_[m]) / alpha[e] + beta[e] / alpha[e]
            for e in U.in_edges(n):
                m = e[0]
                d_n[n] += (lambda_[n] - lambda_[m]) / alpha[e] - beta[e] / alpha[e]
        d.append(d_n)
    return np.array(d)  # .reshape(-1)


# %%

generate_delta(G, lambda_mat, f_mat)  # - od_matrix[1]
# %%
np.sum(od_matrix, axis=0)
# %%
f_mat, lambda_mat = solve_multicommodity_tap(
    G, od_matrix, return_fw=True, pos_flows=True
)
np.sum(f_mat, axis=0)
# %%
x0 = -E.T @ lambda_mat[0] / alpha_arr - beta_arr / alpha_arr
x1 = -E.T @ lambda_mat[1] / alpha_arr - beta_arr / alpha_arr
print(x0)
print(x1)
print(np.sum(f_mat, axis=0))

# %%

num_matrices = len(f_mat)  # Number of matrices

# Step 1: Compute the sub-incidence matrices
sub_matrices = [sub_incidence_matrix(G, f_mat[i]) for i in range(num_matrices)]

# Step 2: Iteratively compute ee, x, and so on
# Compare the new matrix to all prior matrices
intermediate_matrices = []  # Initialize empty list for intermediate results

for i in range(num_matrices):
    # Start with the current sub-matrix
    current_matrix = sub_matrices[i]
    for j in range(i):  # Compare to all previous matrices
        ee = np.where(current_matrix == sub_matrices[j], sub_matrices[j], 0)
        current_matrix = current_matrix - ee  # Subtract overlapping parts
    intermediate_matrices.append(current_matrix)

# Step 3: Stack all intermediate matrices
EE = np.vstack(intermediate_matrices)


# Step 3: Construct the general condition
lhs = -EE.T @ lambda_mat.reshape(-1) / alpha_arr - beta_arr / alpha_arr

# Step 4: Verify the condition
result = np.isclose(lhs, F)
print(f"Condition satisfied: {result}")

# %%
LL = -EE @ EE.T
LL
LL @ lambda_mat.reshape(-1)

# %%
for i, e in enumerate(sub_matrices):
    print(e.T @ lambda_mat[i])
F


# %%

G0 = potential_subgraph(G, f_mat[0])
G1 = potential_subgraph(G, f_mat[1])
# %%
E0 = sub_incidence_matrix(G, f_mat[0])
E1 = sub_incidence_matrix(G, f_mat[1])
# %%

-E0.T @ lambda_mat[0] / alpha_arr - beta_arr / alpha_arr
# %%
-E1.T @ lambda_mat[1] / alpha_arr + beta_arr / alpha_arr

# %%
F
# %%
