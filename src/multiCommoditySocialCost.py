# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import Plotting as pl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import cvxpy as cp
from scipy.linalg import block_diag
from scipy.linalg import null_space

# from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as spla


from src import SocialCost as sc
import warnings
from tqdm import tqdm
import osmnx as ox
from scipy.sparse import csr_matrix, block_diag, bmat, diags, eye, lil_matrix
from memory_profiler import memory_usage
from types import MethodType

np.set_printoptions(precision=3, suppress=True)

from src import osmGraphs as og
from src import Plotting as pl


def _derivative_social_cost_and_flow(
    G, gamma=0.02, num_sources="all", eps=1e-3, **kwargs
):
    nodes, edges = ox.graph_to_gdfs(G)

    if num_sources == "all":
        num_sources = len(G)

    selected_nodes = og.select_evenly_distributed_nodes(nodes, num_sources)

    nodes["source"] = nodes.index.isin(selected_nodes.index)

    nx.set_node_attributes(G, nodes["source"], "source")

    demands = og.demand_list(nodes, commodity=selected_nodes, gamma=gamma)

    solver = kwargs.pop("solver", cp.OSQP)

    f_mat, lambda_mat = mc.solve_multicommodity_tap(
        G, demands, return_fw=True, solver=solver
    )
    F = np.sum(f_mat, axis=0)
    F_dict = dict(zip(G.edges, F))
    nx.set_edge_attributes(G, F_dict, "flow")

    d_sc = derivative_social_cost(G, f_mat, demands, eps=eps)
    edges["derivative_social_cost"] = d_sc

    edges["derivative_social_cost"] = edges["derivative_social_cost"].fillna(0)
    nx.set_edge_attributes(G, edges["derivative_social_cost"], "derivative_social_cost")


nx.DiGraph.derivative_social_cost = _derivative_social_cost_and_flow
nx.MultiDiGraph.derivative_social_cost = _derivative_social_cost_and_flow


def potential_subgraph(G, few, eps=1e-1):
    """
    Returns subgraph with positive flows f_e^w
    """
    U = G.copy()

    for i, e in enumerate(G.edges):
        if few[i] <= eps:
            U.remove_edge(*e)
            # A[n, m] = 1
            # U.edges[e]["flow"] = 0
            # U.edges[e]["alpha"] = np.inf
        else:
            U.edges[e]["flow"] = few[i]
    return U


def sub_incidence_matrix(G, few, eps=1e-1):
    """
    Generates a sparse incidence matrix for the graph G, zeroing out columns
    where `few` values are <= eps.

    Parameters:
        G (networkx.Graph): Input graph.
        few (np.ndarray): Array of values associated with edges.
        eps (float): Threshold for zeroing columns.

    Returns:
        csr_matrix: Sparse incidence matrix.
    """
    # Generate a sparse oriented incidence matrix
    E = -nx.incidence_matrix(G, oriented=True)  # Sparse matrix in CSC format
    E = E.tolil()

    # Find indices where `few <= eps` and zero out corresponding columns
    zero_indices = np.where(few <= eps)[0]
    E[:, zero_indices] = 0  # Set the entire column to 0 in sparse format

    return E.tocsr()  # Convert to CSR format for efficient operations


def generate_coupling_matrix(G, f_mat, eps=1e-1):
    """
    Generate the D1 matrix for a given number of sources and edges.

    Parameters:
        G (networkx.Graph): The graph with edge attribute "alpha".
        f_mat (np.ndarray): The flow matrix.
        eps (float): The threshold for positive flows.

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse D1 matrix.
    """
    # Extract alpha values from the graph's edges
    alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    num_edges = len(alpha)  # Number of edges (E)
    num_sources = len(f_mat)  # Number of sources (N)

    # Create a sparse diagonal matrix for alpha
    alpha_diag = diags(alpha, format="csr")

    # Construct the full block matrix as a sparse grid
    D1_blocks = [[alpha_diag for j in range(num_sources)] for i in range(num_sources)]
    D1 = bmat(D1_blocks, format="csr")

    # Generate the flow mask
    mask = flow_mask(f_mat, eps=eps)

    # Apply the mask to D1
    D1 = D1.multiply(mask)

    # Remove rows and columns that are entirely zero
    D1 = D1[D1.getnnz(1) > 0][:, D1.getnnz(0) > 0]

    return D1


def flow_mask(f_mat, eps=1e-1):
    """
    Returns a sparse mask for positive flows in the flow matrix.

    Parameters:
        f_mat (np.ndarray): The flow matrix.
        eps (float): The threshold for positive flows.

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse mask.
    """
    f_mat_binary = (f_mat > eps).astype(int)
    num_sources, num_edges = f_mat_binary.shape

    # Construct the mask as a sparse grid
    mask_blocks = [
        [
            diags(f_mat_binary[i] * f_mat_binary[j], format="csr")
            for j in range(num_sources)
        ]
        for i in range(num_sources)
    ]
    mask = bmat(mask_blocks, format="csr")

    return mask


def layered_edge_incidence_matrix(G, few, eps=1e-1):
    """
    Generate the layered edge incidence matrix for a given graph and number of layers.

    Parameters:
        G (nx.DiGraph): The graph.
        num_layers (int): The number of layers.

    Returns:
        np.ndarray: The resulting layered edge incidence matrix.
    """
    num_layers = len(few)
    num_nodes = G.number_of_nodes()

    # E_list = [sub_incidence_matrix(G, few[i], eps=eps) for i in range(num_layers)]

    # print(np.shape(E_list))

    # EE = np.block(
    #    [
    #        [
    #            E_list[i] if i == j else np.zeros_like(E_list[i])
    #            for j in range(num_layers)
    #        ]
    #        for i in range(num_layers)
    #    ]
    # )
    # EE = EE[~np.all(EE == 0, axis=1)]
    # EE = EE[:, ~np.all(EE == 0, axis=0)]

    E_list = [sub_incidence_matrix(G, few[i], eps=eps) for i in range(num_layers)]

    # Construct the sparse block matrix
    EE_sparse = block_diag(E_list, format="csr")

    # Remove all-zero rows and columns

    # non_zero_rows = np.any(EE_sparse.toarray() != 0, axis=1)
    non_zero_rows = np.diff(EE_sparse.indptr) > 0
    non_zero_cols = np.diff(EE_sparse.tocsc().indptr) > 0
    # non_zero_cols = np.any(EE_sparse.toarray() != 0, axis=0)
    EE_sparse = EE_sparse[non_zero_rows][:, non_zero_cols]

    # warn if EE.shape[0] != num_layers*num_nodes
    if EE_sparse.shape[0] != num_layers * num_nodes:
        warnings.warn(
            f"Mismatch detected: EE.shape[0] ({EE_sparse.shape[0]}) != num_layers * num_nodes ({num_layers * num_nodes}).",
            UserWarning,
        )

    return EE_sparse.astype(int)


def generate_M_matrix(G, f_mat, eps=1e-1):
    num_nodes = G.number_of_nodes()
    num_layers = len(f_mat)

    # if pos_flow_mask:
    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)
    D1 = csr_matrix(generate_coupling_matrix(G, f_mat, eps=eps))

    D2 = csr_matrix(np.zeros((EE.shape[0], EE.shape[0])))

    M = bmat([[D1, EE.T], [EE, D2]])  # .tocsr()
    # remove zero rows and columns
    # M = M[~np.all(M == 0, axis=1)]
    # M = M[:, ~np.all(M == 0, axis=0)]
    return M


def coupled_laplacian(G, f_mat, eps=1e-1, return_incidences=False):
    """
    Generate the coupled Laplacian matrix for a given graph and flow matrix.

    Parameters:
        G (nx.DiGraph): The graph.
        f_mat (np.ndarray): The flow matrix.
        eps (float): The threshold for positive flows.

    Returns:
        np.ndarray: The resulting coupled Laplacian matrix.
    """

    # Generate the layered edge incidence matrix
    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)

    # Generate the D1 matrix
    K = generate_coupling_matrix(G, f_mat, eps=eps)
    mu = 1e-7
    Ktilde = K - mu * eye(K.shape[0], format="csr")
    Ktilde_inv = np.linalg.inv(Ktilde.toarray())

    # Dtilde_inv = np.linalg.pinv(D1.toarray())

    # Dtilde_inv = inverse_coupling_matrix(G, f_mat, eps=eps)

    # Generate the coupled Laplacian matrix
    LL = -EE @ Ktilde_inv @ EE.T
    if return_incidences:
        return LL, EE

    return LL


def generate_od_matrix(G, num_sources="all", generate_random=False):
    """
    Generates a random OD matrix for a given graph.

    Parameters:
        G (nx.DiGraph): The graph.

    Returns:
        np.ndarray: The resulting OD matrix.
    """
    num_nodes = G.number_of_nodes()
    if num_sources == "all":
        num_sources = G.number_of_nodes()

    # generate vector of sources
    if generate_random:
        sources = np.random.randint(1, num_nodes, num_sources)
        od_matrix = []
        for i, source in enumerate(sources):
            breakpoints = sorted(
                np.random.uniform(0, source) for _ in range(num_nodes - 2)
            )
            breakpoints = [0] + breakpoints + [source]
            parts = [
                -(breakpoints[i + 1] - breakpoints[i]) for i in range(num_nodes - 1)
            ]
            # parts -= source
            parts = np.insert(parts, i, source)
            od_matrix.append(parts)
    if not generate_random:
        od_matrix = -1 * np.ones((num_sources, num_nodes))
        # fill diagonal with sum of all other entries
        for i in range(num_sources):
            od_matrix[i, i] = -np.sum(od_matrix[i]) - 1

    return np.array(od_matrix)


def delete_sparse_rows(sparse_matrix, row_indices):
    """
    Removes specific rows from a sparse matrix.

    Parameters:
        sparse_matrix (csr_matrix): The input sparse matrix.
        row_indices (array-like): Indices of rows to be removed.

    Returns:
        csr_matrix: Sparse matrix with specified rows removed.
    """
    mask = np.ones(sparse_matrix.shape[0], dtype=bool)
    mask[row_indices] = False
    return sparse_matrix[mask]


def delete_sparse_columns(sparse_matrix, col_indices):
    """
    Removes specific columns from a sparse matrix.

    Parameters:
        sparse_matrix (csr_matrix): The input sparse matrix.
        col_indices (array-like): Indices of columns to be removed.

    Returns:
        csr_matrix: Sparse matrix with specified columns removed.
    """
    mask = np.ones(sparse_matrix.shape[1], dtype=bool)
    mask[col_indices] = False
    return sparse_matrix[:, mask]


def derivative_social_cost_OLD(G, f_mat, od_matrix, eps=1e-1):
    """
    Compute the derivative of the social cost function with respect to the flow matrix.

    Parameters:
        G (nx.DiGraph): The graph.
        f_mat (np.ndarray): The flow matrix.

    Returns:
        np.ndarray: The resulting derivative.
    """
    p_vec = np.hstack(od_matrix)

    num_layers = len(f_mat)
    source_nodes = np.where(p_vec > 0)[0]

    alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    alpha_vec = np.array([alpha_arr for _ in range(num_layers)]).flatten()
    f_vec = np.hstack(f_mat)

    LL, EE = coupled_laplacian(G, f_mat, eps=eps, return_incidences=True)
    LL_tilde = np.delete(LL, source_nodes, axis=0)
    LL_tilde = np.delete(LL_tilde, source_nodes, axis=1)
    print("Effective Laplacian shape:", LL.shape)

    # EE_tilde = np.delete(EE, source_nodes, axis=0)
    EE_tilde = delete_sparse_rows(EE, source_nodes)

    LL_tilde_inv = np.linalg.inv(LL_tilde)
    # Linv = np.linalg.pinv(LL)

    p_tilde = p_vec[p_vec <= 0]
    slopes = -EE_tilde.T @ (LL_tilde_inv @ p_tilde) / alpha_vec[f_vec > eps]
    full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]
    pos_flow_edge_indices = np.where(f_vec >= eps)[0]

    slope_edge_dict = {}
    for i, idx in enumerate(pos_flow_edge_indices):
        slope_edge_dict[full_edge_list[idx]] = slopes[i]

    # slope_edge_dict = {}
    # k = 0
    # for w in range(num_layers):
    #    for i, edge in enumerate(edge_list[w]):
    #        edge_tuple = edge + (w,)
    #        slope_edge_dict[edge] = slopes[k]
    #        k += 1
    return slope_edge_dict


def inverse_coupling_matrix(G, f_mat, eps=1e-1):
    num_layers = len(f_mat)
    num_edges = G.number_of_edges()
    alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))

    binary_f_mat = np.where(f_mat > eps, 1, 0)

    rows_per_column = {
        col: np.where(binary_f_mat[:, col] == 1)[0]
        for col in range(binary_f_mat.shape[1])
    }

    kappa = lil_matrix((num_layers * num_edges, num_layers * num_edges))
    mu = 100000

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
    return kappa_reduced  # .toarray()


def derivative_social_cost(G, f_mat, od_matrix, eps=1e-2):
    """Compute the derivative of the social cost efficiently."""
    start_time = time.time()
    num_layers = len(f_mat)

    p_vec = np.hstack(od_matrix)
    f_vec = np.hstack(f_mat)

    source_nodes = np.flatnonzero(p_vec > 0)
    sink_nodes = np.flatnonzero(p_vec < 0)

    kappa = inverse_coupling_matrix(G, f_mat, eps=eps)

    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps).tocsc()
    LL = -EE @ kappa @ EE.T

    print("Effective Laplacian shape:", LL.shape)

    # Efficient masking to remove source_nodes
    mask = np.ones(LL.shape[0], dtype=bool)
    mask[source_nodes] = False

    LL_tilde = LL[mask][:, mask]
    EE_tilde = EE[mask, :]

    # Efficient sparse inversion via linear solve (avoiding dense inversion)
    rhs = -(EE_tilde @ kappa).tocsc()
    D = spla.spsolve(LL_tilde.tocsc(), rhs)

    # Compute slopes
    slopes = D.T @ p_vec[sink_nodes]

    pos_flow_edge_indices = np.flatnonzero(f_vec > eps)
    full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

    slope_edge_dict = {}
    for idx, slope_value in zip(pos_flow_edge_indices, slopes):
        edge = full_edge_list[idx]
        slope_edge_dict[edge] = slope_edge_dict.get(edge, 0) + slope_value

    print("Total computation time:", time.time() - start_time, "s")

    return slope_edge_dict


def derivative_social_cost_(G, f_mat, od_matrix, eps=1e-2):
    """ """
    start_time = time.time()
    num_layers = len(f_mat)
    p_vec = np.hstack(od_matrix)
    f_vec = np.hstack(f_mat)
    source_nodes = np.where(p_vec > 0)[0]

    kappa = inverse_coupling_matrix(G, f_mat, eps=eps)

    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()
    LL = -EE @ kappa @ EE.T

    # Efficiently delete rows and columns corresponding to source_nodes
    mask = np.ones(LL.shape[0], dtype=bool)
    mask[source_nodes] = False

    LL_tilde = LL[mask][:, mask]  # Submatrix without source_nodes
    EE_tilde = EE[mask, :]  # Efficient row deletion using slicing

    print("Effective Laplacian shape:", LL.shape)

    D = np.linalg.inv(LL_tilde.toarray())
    print("Time to invert:", time.time() - start_time, "s")
    C = -D @ EE_tilde @ kappa

    # b = -(EE_tilde @ kappa).tocsc()
    # C = spsolve(LL_tilde.tocsc(), b)

    slopes = C.T @ p_vec[p_vec < 0]

    pos_flow_edge_indices = np.where(f_vec > eps)[0]
    full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

    slope_edge_dict = {}
    for i, idx in enumerate(pos_flow_edge_indices):
        # if key exsists add to it
        if full_edge_list[idx] in slope_edge_dict:
            slope_edge_dict[full_edge_list[idx]] += slopes[i]
        # if key does not exist create new key
        else:
            slope_edge_dict[full_edge_list[idx]] = slopes[i]

    return slope_edge_dict


def numerical_derivative(G, od_matrix, edge, num=25, var_percentage=0.1, **kwargs):
    beta = nx.get_edge_attributes(G, "beta")
    beta_e = beta[edge]
    eps = beta_e * var_percentage
    beta_list = np.linspace(beta_e - eps, beta_e + eps, num)

    social_cost_list = []
    solver = kwargs.pop("solver", cp.MOSEK)
    for beta_e in tqdm(beta_list):
        beta[edge] = beta_e
        beta_arr = np.array(list(beta.values()))
        f = mc.solve_multicommodity_tap(
            G, od_matrix, pos_flows=True, beta=beta_arr, solver=solver
        )
        social_cost = sc.total_social_cost(G, f, beta=beta_arr)
        social_cost_list.append(social_cost)

    slopes = np.gradient(social_cost_list, beta_list)
    return slopes, beta_list, social_cost_list


def test_system(G, f_mat, lambda_mat, od_matrix, eps=1e-2):
    num_layers = len(f_mat)

    beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    beta_vec = np.array([beta_arr for _ in range(num_layers)]).flatten()
    f_vec = np.hstack(f_mat)
    lambda_vec = np.hstack(lambda_mat)
    p_vec = np.hstack(od_matrix)

    M = generate_M_matrix(G, f_mat, eps=eps)
    x = np.hstack([f_vec[f_vec >= eps], lambda_vec])
    y = np.hstack([-beta_vec[f_vec >= eps], p_vec])

    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)
    print("Test system:")
    print(np.max(np.abs(M @ x - y)))

    K = generate_coupling_matrix(G, f_mat, eps=eps)
    mu = 1e-7
    Ktilde = K - mu * np.eye(K.shape[0])
    Ktilde_inv = np.linalg.inv(Ktilde)
    # K_inv = inverse_coupling_matrix(G, f_mat, eps=eps)

    LL = -EE @ Ktilde_inv @ EE.T

    D = np.linalg.pinv(LL)

    C = -D @ EE @ Ktilde_inv

    rhs = -C @ beta_vec[f_vec >= eps] + D @ p_vec
    lhs = lambda_vec

    rhs_vec = np.array(rhs).flatten()
    lhs_vec = np.array(lhs).flatten()

    print("Test inverse:")
    print(np.max(np.abs(EE.T @ rhs_vec - EE.T @ lhs_vec)))

    return M, x, y


# %%

if __name__ == "__main__":
    # inititalize graph etc
    # G = gr.random_graph(
    #    15, num_edges=12, beta="random", alpha="random", directed=True, seed=2
    # )
    G = gr.random_planar_graph(4, seed=5)
    nodes, edges = ox.graph_to_gdfs(G)
    alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    print(G)
    num_edges = G.number_of_edges()

    num_nodes = G.number_of_nodes()

    num_layers = len(G)

    selected_nodes = og.select_evenly_distributed_nodes(nodes, num_layers)
    demands = og.demand_list(nodes, commodity=selected_nodes, gamma=0.1)
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
    pl.graphPlot(G, ec=F)
    # %%
    G.derivative_social_cost(eps=1e-3)
    nodes, edges = ox.graph_to_gdfs(G)
    # %%
    edges.sort_values("derivative_social_cost", ascending=False)
    # %%
    eps = 1e-3
    kappa0 = inverse_coupling_matrix(G, f_mat, eps=eps)
    EE0 = layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()
    LL0 = (-EE0 @ kappa0 @ EE0.T).toarray()

    eps = 1e-4
    kappa1 = inverse_coupling_matrix(G, f_mat, eps=eps)
    EE1 = layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()
    LL1 = (-EE1 @ kappa1 @ EE1.T).toarray()

    # %%
    start_time = time.time()
    derivative_v2 = derivative_social_cost_OLD(G, f_mat, od_matrix, eps=1e-3)
    print("Elapsed time:", time.time() - start_time)
    derivative_v2
    # %%
    start_time = time.time()
    derivative_v1 = derivative_social_cost(G, f_mat, od_matrix, eps=1e-3)
    print("Elapsed time:", time.time() - start_time)
    derivative_v1

    diff = {key: derivative_v1[key] - derivative_v2[key] for key in derivative_v1}
    print("Max difference:", np.max(np.abs(list(diff.values()))))

    # %%

    usage = memory_usage((derivative_social_cost, (G, f_mat, od_matrix), {"eps": 1e-3}))
    print(f"Memory usage of the new function: {max(usage) - min(usage):.2f} MiB")

    usage = memory_usage(
        (derivative_social_cost_OLD, (G, f_mat, od_matrix), {"eps": 1e-3})
    )
    print(f"Memory usage of the old function: {max(usage) - min(usage):.2f} MiB")
    # %%

    edge = list(G.edges)[3]
    print(edge)

    s0 = derivative_v1[edge]

    s1 = derivative_v2[edge]

    s2 = numerical_derivative(G, od_matrix, edge)[0]

    print(s0)
    print(s1)
    print(s2)
    # %%

    M, x, y = test_system(G, f_mat, lambda_mat, od_matrix, eps=1e-1)
    # %%

    Minv = np.linalg.pinv(M.toarray())

    print(np.allclose(M @ x, y, atol=1e0))
    print(np.allclose(Minv @ y, x, atol=1e0))
    np.max(np.abs(M @ x - y))

    # %%
    EE = layered_edge_incidence_matrix(G, f_mat, eps=1e-4)
    K = generate_coupling_matrix(G, f_mat, eps=1e-4)
    mu = 1e-7
    Ktilde = K - mu * np.eye(K.shape[0])
    K_inv = np.linalg.inv(Ktilde)
    # K_inv = inverse_coupling_matrix(G, f_mat, eps=1e-4)

    LL = -EE @ K_inv @ EE.T

    to_delete = [len(G.nodes) * i for i in range(num_layers)]

    LL_tilde = np.delete(LL, to_delete, axis=0)
    LL_tilde = np.delete(LL_tilde, to_delete, axis=1)

    Linv = np.linalg.inv(LL_tilde)

    EE_tilde = delete_sparse_rows(EE, to_delete)

    C = -Linv @ EE_tilde @ K_inv

    # %%
    p_tilde = np.delete(p_vec, to_delete)
    rhs = -C @ beta_vec[f_vec >= 1e-2] + Linv @ p_tilde
    rhs_vec = np.array(rhs).flatten()

    EE_tilde.T @ rhs_vec - EE.T @ lambda_vec
    # %%

    layersxedges = EE.shape[1]

    # get bottom left block of M
    M11 = Minv[:layersxedges, :layersxedges]
    M12 = Minv[:layersxedges, layersxedges:]

    M21 = Minv[layersxedges:, :layersxedges]
    M22 = Minv[layersxedges:, layersxedges:]

    print(np.allclose(np.block([[M11, M12], [M21, M22]]), Minv))

    # test against schur method

    if not np.allclose(Linv, M22):
        diff = Linv - M22
        print("Not true")
        print(np.max(np.abs(diff)))

    if not np.allclose(C, M21):
        diff = C - M21
        print("Not true")
        print(np.max(np.abs(diff)))

    rhs = -C @ beta_vec[f_vec >= 1e-2] + Linv @ p_vec

    rhs_mat = np.reshape(rhs, (num_layers, num_nodes))
    np.diff(rhs_mat) - np.diff(lambda_mat)

    # %%

    pos_flow_edge_indices = np.where(f_vec >= 1e-4)[0]
    slopes = M21.T @ p_vec
    full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

    slope_edge_dict = {}
    for i, idx in enumerate(pos_flow_edge_indices):
        # if key exsists add to it
        if full_edge_list[idx] in slope_edge_dict:
            slope_edge_dict[full_edge_list[idx]] += slopes[i]
        # if key does not exist create new key
        else:
            slope_edge_dict[full_edge_list[idx]] = slopes[i]

    slope_edge_dict

    # %%

    beta_ = beta_vec[f_vec >= 1e-4]
    kappa = inverse_coupling_matrix(G, f_mat, eps=1e-4)

    D = np.linalg.pinv((-EE @ kappa @ EE.T))
    C = -D @ EE @ kappa

    alpha_ = alpha_vec[f_vec >= 1e-4]

    # derivative =
    a = 2 * C.T @ EE @ kappa @ EE.T @ (C @ beta_ - D @ p_vec)
    b = -C.T @ EE @ kappa @ beta_
    c = -kappa @ EE.T @ (C @ beta_ - D @ p_vec)
    derivative = a + b + c
    derivative - C.T @ p_vec

    # %%

    mu = 1e-7
    K = generate_coupling_matrix(G, f_mat, eps=1e-4)
    K_tilde = K - mu * np.eye(K.shape[0])
    Kinv_tilde = np.linalg.inv(K_tilde)

    D = np.linalg.pinv((-EE @ Kinv_tilde @ EE.T))
    C = -D @ EE @ Kinv_tilde

    C.T @ p_vec  # - derivative

# %%
