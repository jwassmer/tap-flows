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
from src import SocialCost as sc
import warnings

np.set_printoptions(precision=3, suppress=True)

from src import Plotting as pl


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
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    zero_indices = np.where(few <= eps)[0]
    E[:, zero_indices] = 0
    # for n in G.nodes:
    #    for e in range(len(G.edges)):
    # edg = list(G.edges)[e]
    #        if few[e] <= eps:
    #            E[n, e] = 0
    return E.astype(int)


def generate_coupling_matrix(G, f_mat, eps=1e-1):
    """
    Generate the D1 matrix for a given number of sources and edges.

    Parameters:
        alpha (list or np.ndarray): A list of alpha values (\(\alpha_e\)) of length E.
        num_sources (int): The number of sources.

    Returns:
        np.ndarray: The resulting D1 matrix.
    """
    alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    num_edges = len(alpha)  # Number of edges (E)
    num_sources = len(f_mat)  # Number of sources (N)

    # Create a diagonal matrix for alpha
    alpha_diag = np.diag(alpha)

    # Tile the diagonal matrix for each source
    D1 = np.block(
        [[alpha_diag for j in range(num_sources)] for i in range(num_sources)]
    )

    mask = flow_mask(f_mat, eps=eps)
    D1 = D1 * mask
    D1 = D1[~np.all(D1 == 0, axis=1)]
    D1 = D1[:, ~np.all(D1 == 0, axis=0)]

    return D1


def flow_mask(f_mat, eps=1e-1):
    """
    Returns a mask for positive flows in the flow matrix.

    Parameters:
        f_mat (np.ndarray): The flow matrix.
        eps (float): The threshold for positive flows.

    Returns:
        np.ndarray: The resulting mask.
    """
    f_mat_binary = np.where(f_mat > eps, 1, 0)
    mask = np.block(
        [
            [
                np.diag(f_mat_binary[i] * f_mat_binary[j])
                for i in range(f_mat_binary.shape[0])
            ]
            for j in range(f_mat_binary.shape[0])
        ]
    )
    # mask negative flows
    # mask = (f_mixed > eps).astype(int)

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

    E_list = [sub_incidence_matrix(G, few[i], eps=eps) for i in range(num_layers)]

    EE = np.block(
        [
            [
                E_list[i] if i == j else np.zeros_like(E_list[i])
                for j in range(num_layers)
            ]
            for i in range(num_layers)
        ]
    )
    EE = EE[~np.all(EE == 0, axis=1)]
    EE = EE[:, ~np.all(EE == 0, axis=0)]

    # warn if EE.shape[0] != num_layers*num_nodes
    if EE.shape[0] != num_layers * num_nodes:
        warnings.warn(
            f"Mismatch detected: EE.shape[0] ({EE.shape[0]}) != num_layers * num_nodes ({num_layers * num_nodes}).",
            UserWarning,
        )

    return EE.astype(int)


def generate_M_matrix(G, f_mat, eps=1e-1):
    num_nodes = G.number_of_nodes()
    num_layers = len(f_mat)

    # if pos_flow_mask:
    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)
    D1 = generate_coupling_matrix(G, f_mat, eps=eps)

    D2 = np.zeros((EE.shape[0], EE.shape[0]))

    M = np.block([[D1, EE.T], [EE, D2]])
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
    D1 = generate_coupling_matrix(G, f_mat, eps=eps)
    mu = 1e-7
    Dtilde = D1 - mu * np.eye(D1.shape[0])
    Dtilde_inv = np.linalg.inv(Dtilde)
    # Dtilde_inv = inverse_coupling_matrix(G, f_mat, eps=eps)

    # Generate the coupled Laplacian matrix
    LL = -EE @ Dtilde_inv @ EE.T
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


def derivative_social_cost(G, f_mat, od_matrix, eps=1e-1):
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

    EE_tilde = np.delete(EE, source_nodes, axis=0)

    LL_tilde_inv = np.linalg.inv(LL_tilde)
    # Linv = np.linalg.pinv(LL)

    p_tilde = p_vec[p_vec < 0]
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
    alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    num_layers = len(f_mat)
    binary_f_mat = np.where(f_mat > eps, 1, 0)

    flow_count = 1 / (np.sum(binary_f_mat, axis=0) - (1 - 1e-5))

    kappa_count = flow_count / alpha_
    layered_kappa_count = np.array([kappa_count for _ in range(num_layers)]).flatten()

    layered_kappa_count = layered_kappa_count * binary_f_mat.flatten()
    layered_kappa_count = layered_kappa_count[layered_kappa_count > 0]
    kappa = np.diag(layered_kappa_count)
    return kappa


def derivative_social_cost2(G, f_mat, od_matrix, eps=1e-2):
    """ """
    p_vec = np.hstack(od_matrix)
    alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))

    binary_f_mat = np.where(f_mat > eps, 1, 0)

    flow_count = 1 / (np.sum(binary_f_mat, axis=0) - (1 - 1e-8))

    kappa_count = flow_count / alpha_
    layered_kappa_count = np.array([kappa_count for _ in range(num_layers)]).flatten()

    layered_kappa_count = layered_kappa_count * binary_f_mat.flatten()
    layered_kappa_count = layered_kappa_count[layered_kappa_count > eps]
    kappa = np.diag(layered_kappa_count)

    EE = layered_edge_incidence_matrix(G, f_mat, eps=eps)

    D = np.linalg.pinv((-EE @ kappa @ EE.T))
    C = -D @ EE @ kappa

    slopes = C.T @ p_vec

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

    return slope_edge_dict


def numerical_derivative(G, od_matrix, edge):
    beta = nx.get_edge_attributes(G, "beta")
    beta_e = beta[edge]
    eps = 1
    beta_list = np.linspace(beta_e - eps, beta_e + eps, 20)

    social_cost_list = []
    for beta_e in beta_list:
        beta[edge] = beta_e
        beta_arr = np.array(list(beta.values()))
        f = mc.solve_multicommodity_tap(G, od_matrix, pos_flows=True, beta=beta_arr)
        social_cost = sc.total_social_cost(G, f, beta=beta_arr)
        social_cost_list.append(social_cost)

    slope = np.gradient(social_cost_list, beta_list)
    return slope


# %%

if __name__ == "__main__":
    # inititalize graph etc
    G = gr.random_graph(
        50, num_edges=3, beta="random", alpha="random", directed=True, seed=2
    )
    alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    print(G)
    num_edges = G.number_of_edges()

    num_layers = len(G.nodes)
    num_nodes = G.number_of_nodes()

    od_matrix = generate_od_matrix(G, num_sources=num_layers)
    p_vec = np.array(od_matrix).flatten()

    f_mat, lambda_mat = mc.solve_multicommodity_tap(
        G, od_matrix, return_fw=True, pos_flows=True
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
    derivative_v2 = derivative_social_cost2(G, f_mat, od_matrix, eps=1e-2)
    derivative_v1 = derivative_social_cost(G, f_mat, od_matrix, eps=1e-2)

    np.max(
        np.abs(
            np.array(list(derivative_v1.values()))
            - np.array(list(derivative_v2.values()))
        )
    )
    # %%
    derivative_v1
    # %%

    edge = (29, 23)
    print(edge)

    s0 = derivative_v1[edge]

    s1 = derivative_v2[edge]

    s2 = numerical_derivative(G, od_matrix, edge)

    print(s0)
    print(s1)
    print(s2)
    # %%

    M = generate_M_matrix(G, f_mat, eps=1e-4)
    x = np.hstack([f_vec[f_vec >= 1e-4], lambda_vec])
    y = np.hstack([-beta_vec[f_vec >= 1e-4], p_vec])
    M.shape

    # %%

    Minv = np.linalg.pinv(M)

    print(np.allclose(M @ x, y, atol=1e0))
    print(np.allclose(Minv @ y, x, atol=1e0))
    np.max(np.abs(M @ x - y))

    # %%
    EE = layered_edge_incidence_matrix(G, f_mat, eps=1e-4)
    LL = coupled_laplacian(G, f_mat, eps=1e-4)
    Linv = np.linalg.pinv(LL)
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
        print(np.max(np.abs(diff)))

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
