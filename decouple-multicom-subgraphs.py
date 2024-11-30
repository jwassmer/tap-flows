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
import random

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


def potential_subgraph(G, few, eps=1e-2):
    """
    Returns subgraph with positive flows f_e^w
    """
    U = G.copy()

    for i, e in enumerate(G.edges):
        if few[i] < eps:
            # U.remove_edge(*e)
            # A[n, m] = 1
            U.edges[e]["flow"] = 0
            U.edges[e]["alpha"] = np.inf
        else:
            U.edges[e]["flow"] = few[i]
    return U


def decoupled_subgraphs(G, f_mat, eps=1e-3, pos_flows=True):
    num_layers = len(f_mat)
    U_list = []

    # alpha_dict = nx.get_edge_attributes(G, "alpha")

    if pos_flows:
        for w in range(num_layers):
            Uw = potential_subgraph(G, f_mat[w], eps=eps)
            U_list.append(Uw)
    else:
        for w in range(num_layers):
            U = G.copy()
            f = f_mat[w]
            nx.set_edge_attributes(U, dict(zip(G.edges, f)), "flow")
            U_list.append(U)

    # counts = np.sum(f_mat.T > eps, axis=1)
    # shared_edge_indices = np.where(counts > 1)[0]  # Find indices where  count > 1
    # counts_at_indices = counts[shared_edge_indices]  # Get the counts for those indices
    # subgraph_indices = [np.where(f_mat[:, col] > eps)[0] for col in shared_edge_indices]

    # shared_edge_indices_dict = dict(zip(shared_edge_indices, subgraph_indices))
    # print(shared_edge_indices_dict)

    for edge_idx in range(G.number_of_edges()):
        edge = list(G.edges)[edge_idx]
        tot_f_e = np.sum(f_mat[:, edge_idx])
        for w in range(num_layers):
            U = U_list[w]

            if np.isfinite(U.edges[edge]["alpha"]):
                U.edges[edge]["alpha"] *= tot_f_e / np.sum(f_mat[w, edge_idx])

    # for edge_idx, layer_idices in shared_edge_indices_dict.items():
    #    edge = list(G.edges)[edge_idx]
    #    for subgraph_idx in layer_idices:
    #        U = U_list[subgraph_idx]

    #        tot_f_e = np.sum(f_mat[:, edge_idx])

    #       U.edges[edge]["alpha"] *= tot_f_e / f_mat[subgraph_idx, edge_idx]
    return U_list


# %%
# inititalize graph etc
G = gr.random_graph(4, num_edges=5, beta="random", alpha="random", directed=True)


alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
print(G)
num_edges = G.number_of_edges()

num_layers = 1  # len(G.nodes)
n = 10
od_matrix = -n * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, n * (G.number_of_nodes() - 1))

od_matrix = [od_matrix[:, i] for i in range(num_layers)]


f_mat, lambda_mat = solve_multicommodity_tap(
    G, od_matrix, return_fw=True, pos_flows=False
)

F = np.sum(f_mat, axis=0)

E = -nx.incidence_matrix(G, oriented=True).toarray()
# %%

U_list = decoupled_subgraphs(G, f_mat, eps=1e-2, pos_flows=False)

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
for i in range(num_layers):
    U = U_list[i]
    f = nx.get_edge_attributes(U, "flow")
    # alpha = nx.get_edge_attributes(U, "alpha")
    # print(alpha)
    pl.graphPlot(U, title=None, ec=f, edge_labels=f, ax=axs.flatten()[i])
# %%

decoupled_flows = []
for i, U in enumerate(U_list):
    f, l = tap.linearTAP(U, od_matrix[i])
    decoupled_flows.append(f)


decoupled_flows


# %%

# predict braess edges

s_list = []
for i, U in enumerate(U_list):
    E = -nx.incidence_matrix(U, oriented=True).toarray()
    eff_alpha = np.array(list(nx.get_edge_attributes(U, "alpha").values()))
    L = E @ np.diag(1 / eff_alpha) @ E.T
    gamma = np.linalg.pinv(L) @ od_matrix[i]
    f_beta0 = E.T @ gamma / eff_alpha
    # f_dict = dict(zip(U.edges, f_beta0))
    s_list.append(f_beta0)

np.sum(np.array(s_list), axis=0)


# %%

edge_idx = 3
edge = list(G.edges)[edge_idx]
print("edge", edge)
beta_ = G.edges[edge]["beta"]
eps = 2
beta_list = np.linspace(beta_ - eps, beta_ + eps, 10)


social_cost_list = []

f_w_list = []
f_list = []
lambda_list = []

delta_lamb = E.T @ lambda_mat.flatten()  # [edge_idx]

for beta in beta_list:
    G.edges[edge]["beta"] = beta

    # f0, lamb0 = tap.linearTAP(G, od_matrix[0])

    f, lamb = solve_multicommodity_tap(G, od_matrix, return_fw=True, pos_flows=False)

    lambda_list.append(-lamb[0])
    # f = tap.user_equilibrium(G, od_matrix)
    s = sc.total_social_cost(G, f[0])
    social_cost_list.append(s)

# %%
fig, ax = plt.subplots()
ax.plot(beta_list, lambda_list)
ax.grid()
ax.set_xlabel("Beta")
ax.set_ylabel("Social Cost")

lamb_slope = []
for l in np.array(lambda_list).T:
    slope = np.mean(np.diff(l) / np.diff(beta_list))
    lamb_slope.append(slope)
# %%
A = sum((2 * (E.T @ lamb_slope) * delta_lamb) / alpha_arr)

B = sum((E.T @ lamb_slope * beta_arr) / alpha_arr)


C = delta_lamb[edge_idx] / alpha_arr[edge_idx]

A - 2 * C

# %%
sum((E.T @ lamb_slope) * delta_lamb)
# %%

L = gr.directed_laplacian(G)
Linv = np.linalg.pinv(L)
a, b = edge[0], edge[1]
lambda_derivative = 1 / alpha_arr[edge_idx] * (Linv[:, a] - Linv[:, b])
lambda_derivative
# %%
E.T @ lambda_derivative
# %%
