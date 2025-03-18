# %%
import networkx as nx
import osmnx as ox
import numpy as np
import cvxpy as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import cg
from numpy.linalg import eig

from src import Plotting as pl
from src import Graphs as gr
from src import osmGraphs as og
from src import multiCommodityTAP as mc
from src import multiCommoditySocialCost as mcsc


# %%
eps = 1e-3
G = gr.random_planar_graph(50, seed=5, alpha=1)
nodes, edges = ox.graph_to_gdfs(G)
alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
print(G)
num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()

num_layers = len(G)

selected_nodes = og.select_evenly_distributed_nodes(nodes, num_layers)
demands = og.demand_list(nodes, commodity=selected_nodes, gamma=0.04)
od_matrix = np.array(demands)

p_vec = np.array(od_matrix).flatten()

f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_matrix, return_fw=True, pos_flows=True, solver=cp.MOSEK
)
F = np.sum(f_mat, axis=0)
F_dict = dict(zip(G.edges, F))
print("mean flow per edge:", np.mean(f_mat) / num_layers / num_edges)
lambda_vec = np.hstack(lambda_mat)
f_vec = np.hstack(f_mat)
beta_vec = np.array([beta_arr for _ in range(num_layers)]).flatten()
alpha_vec = np.array([alpha_arr for _ in range(num_layers)]).flatten()


K = mcsc.generate_coupling_matrix(G, f_mat, eps=eps).toarray()

d_sc = mcsc.derivative_social_cost(G, f_mat, od_matrix, eps=eps)

d_sc_vec = np.array(list(d_sc.values()))

d_sc

# %%

source_nodes = np.where(p_vec > 0)[0]

num_layers = len(f_mat)
p_vec = np.hstack(od_matrix)
f_vec = np.hstack(f_mat)
alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))

binary_f_mat = np.where(f_mat > eps, 1, 0)


rows_per_column = {
    col: np.where(binary_f_mat[:, col] == 1)[0] for col in range(binary_f_mat.shape[1])
}


kappa = lil_matrix((num_layers * num_edges, num_layers * num_edges))
mu = 10000

for key, value in rows_per_column.items():
    for i in value:
        for j in value:
            if i == j:
                kappa[i * num_edges + key, j * num_edges + key] = -(
                    mu * (len(value) - 1) / (len(value))
                ) + 1 / (len(value) ** 2 * alpha_[key])

            else:
                kappa[i * num_edges + key, j * num_edges + key] = mu / (
                    len(value)
                ) + 1 / (len(value) ** 2 * alpha_[key])

# Convert kappa to CSR format for efficient row and column operations
kappa_csr = kappa.tocsr()

# Identify rows and columns that contain only zeros
nonzero_row_mask = kappa_csr.getnnz(axis=1) > 0
nonzero_col_mask = kappa_csr.getnnz(axis=0) > 0

# Keep only non-zero rows and columns
kappa_reduced = kappa_csr[nonzero_row_mask][:, nonzero_col_mask]

kappa = kappa_reduced
# %%

EE = mcsc.layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()

LL = -EE @ kappa @ EE.T  # Use sparse matrix multiplication

# Efficiently delete rows and columns corresponding to source_nodes
mask = np.ones(LL.shape[0], dtype=bool)
mask[source_nodes] = False

LL_tilde = LL[mask][:, mask]  # Submatrix without source_nodes
EE_tilde = EE[mask, :]  # Efficient row deletion using slicing

# I_tilde = identity(LL_tilde.shape[0]).tocsc()  # Sparse identity matrix
# D = spsolve(LL_tilde.tocsc(), I_tilde)


start_time = time.time()
D = np.linalg.inv(LL_tilde.toarray())
print("INVERSION TIME --- %s seconds ---" % (time.time() - start_time))


C = -D @ EE_tilde @ kappa


slopes = C.T @ p_vec[p_vec < 0]

# Compute slopes
p_vec_neg = p_vec[p_vec < 0]
slopes = C.T @ p_vec_neg


pos_flow_edge_indices = np.where(f_vec >= eps)[0]
full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

slope_edge_dict = {}
for i, idx in enumerate(pos_flow_edge_indices):
    # if key exsists add to it
    if full_edge_list[idx] in slope_edge_dict:
        slope_edge_dict[full_edge_list[idx]] += slopes[i]
    # if key does not exist create new key
    else:
        slope_edge_dict[full_edge_list[idx]] = slopes[i]

slopes1 = pd.Series(slope_edge_dict)
slopes1


# %%
mu = 1e-5
K_tilde = K - mu * np.eye(K.shape[0])
K_inv = np.linalg.inv(K_tilde)

# K_inv[np.abs((K_inv) < 5) & (K_inv > 0)] = 1
# K_inv[K_inv < 100] = 0


LL2 = -EE @ K_inv @ EE.T
LL2_tilde = np.delete(LL2, source_nodes, axis=0)
LL2_tilde = np.delete(LL2_tilde, source_nodes, axis=1)


D = np.linalg.inv(LL2_tilde)

C = -D @ EE_tilde @ K_inv


slopes2 = C.T @ p_vec[p_vec < 0]

pos_flow_edge_indices = np.where(f_vec >= eps)[0]
full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

slope_edge_dict = {}
for i, idx in enumerate(pos_flow_edge_indices):
    # if key exsists add to it
    if full_edge_list[idx] in slope_edge_dict:
        slope_edge_dict[full_edge_list[idx]] += slopes2[i]
    # if key does not exist create new key
    else:
        slope_edge_dict[full_edge_list[idx]] = slopes2[i]

slopes2 = pd.Series(slope_edge_dict)

# %%

np.max(np.abs(slopes1 - slopes2))

# %%


# Check positive definiteness of LL_tilde
def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)


is_pd = is_positive_definite(LL_tilde.toarray())
print("LL_tilde is positive definite:", is_pd)

is_sym = is_symmetric(LL_tilde.toarray())
print("LL_tilde is symmetric:", is_sym)

# %%
