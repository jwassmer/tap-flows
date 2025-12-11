# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import osmnx as ox
import cvxpy as cp
from matplotlib.gridspec import GridSpec
import time
import warnings
from geopy.distance import geodesic
from scipy.spatial import Delaunay

from scipy.sparse import block_diag, lil_matrix, diags, bmat

from src import Plotting as pl


def random_planar_graph(n_nodes, seed=42, alpha=1):
    """
    Create a Delaunay triangulation for n_nodes randomly distributed points.
    """

    np.random.seed(seed)
    # Generate random points in 2D space
    latitudes = np.random.uniform(
        low=52.9, high=53.0, size=n_nodes
    )  # Latitude range for a region like Berlin
    longitudes = np.random.uniform(
        low=13.9, high=14.0, size=n_nodes
    )  # Longitude range for a region like Berlin

    points = np.column_stack((longitudes, latitudes))

    # Compute the Delaunay triangulation
    tri = Delaunay(points)

    # Create a graph from the triangulation
    G = nx.MultiGraph()

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                node1, node2 = simplex[i], simplex[j]
                point1 = points[node1]
                point2 = points[node2]

                # Calculate geodesic distance between points using EPSG:4326
                speed_kph = 50
                beta = geodesic(
                    (point1[1], point1[0]), (point2[1], point2[0])
                ).meters / (speed_kph * 1000 / 3600)

                G.add_edge(node1, node2, weight=beta)

    # convert to multiddigraph
    G = G.to_directed()
    # remove parallel edges
    new_G = nx.MultiDiGraph()

    for u, v, keys in G.edges(keys=True):

        if not new_G.has_edge(u, v):

            new_G.add_edge(u, v, key=keys, **G[u][v][keys])

    G = new_G

    # a = 2
    # values = np.array([1, 2, 3, 4, 5]) / 2
    # probabilities_scale_free = np.array([x**a for x in values])
    # probabilities_scale_free = probabilities_scale_free / probabilities_scale_free.sum()

    # alpha_vec = np.random.choice(values, size=len(G.edges), p=probabilities_scale_free)
    # alpha_dict = dict(zip(G.edges, alpha_vec))

    nx.set_node_attributes(G, {i: points[i] for i in range(len(points))}, "pos")
    nx.set_node_attributes(G, {i: latitudes[i] for i in range(len(points))}, "x")
    nx.set_node_attributes(G, {i: longitudes[i] for i in range(len(points))}, "y")

    if alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
        nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    else:
        nx.set_edge_attributes(G, alpha, "alpha")

    beta = nx.get_edge_attributes(G, "weight")
    nx.set_edge_attributes(G, beta, "beta")

    # draw population vals from agussian
    # nx.set_node_attributes(
    #    G, {i: np.random.normal(100, 10) for i in range(len(points))}, "population"
    # )
    # draw population vals from a scale free distribution
    a = -1
    values = np.linspace(100, 5000, num=1000)
    probabilities_scale_free = np.array([x**a for x in values])
    probabilities_scale_free = probabilities_scale_free / probabilities_scale_free.sum()
    nx.set_node_attributes(
        G,
        {
            i: np.random.choice(values, p=probabilities_scale_free)
            for i in range(len(points))
        },
        "population",
    )

    G.graph["crs"] = "epsg:4326"
    return G


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

    # Construct the sparse block matrix
    EE_sparse = block_diag(E_list, format="csr")

    # Remove all-zero rows and columns

    # non_zero_rows = np.any(EE_sparse.toarray() != 0, axis=1)
    non_zero_rows = np.diff(EE_sparse.indptr) > 0
    non_zero_cols = np.diff(EE_sparse.tocsc().indptr) > 0
    # non_zero_cols = np.any(EE_sparse.toarray() != 0, axis=0)
    EE_sparse = EE_sparse[non_zero_rows][:, non_zero_cols]

    # warn if EE.shape[0] != num_layers*num_nodes
    while EE_sparse.shape[0] != num_layers * num_nodes:
        eps /= 1.25
        E_list = [sub_incidence_matrix(G, few[i], eps=eps) for i in range(num_layers)]
        EE_sparse = block_diag(E_list, format="csr")
        non_zero_rows = np.diff(EE_sparse.indptr) > 0
        non_zero_cols = np.diff(EE_sparse.tocsc().indptr) > 0
        EE_sparse = EE_sparse[non_zero_rows][:, non_zero_cols]
        if EE_sparse.shape[0] == num_layers * num_nodes:
            warnings.warn(
                f"Reduced eps to {eps:.2e} to match expected incidence matrix size."
            )
        if eps < 1e-10:
            # EE_sparse.shape[0] != num_layers * num_nodes,
            raise ValueError(
                f"Could not match expected incidence matrix size. Current shape: {EE_sparse.shape}, Expected: ({num_layers * num_nodes}, {EE_sparse.shape[1]})"
            )

    return EE_sparse.astype(int), eps


def derivative_social_cost(G, f_mat, od_matrix, eps=1e-2):
    """ """
    start_time = time.time()
    num_layers = len(f_mat)
    p_vec = np.hstack(od_matrix)
    f_vec = np.hstack(f_mat)
    source_nodes = np.where(p_vec > 0)[0]

    EE, eps = layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()
    kappa = inverse_coupling_matrix(G, f_mat, eps=eps)
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

    slope_edge_dict = {edge: 0 for edge in G.edges}
    for i, idx in enumerate(pos_flow_edge_indices):
        # if key exsists add to it
        if full_edge_list[idx] in slope_edge_dict:
            slope_edge_dict[full_edge_list[idx]] += slopes[i]
        # if key does not exist create new key
        else:
            slope_edge_dict[full_edge_list[idx]] = slopes[i]

    return slope_edge_dict


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


def middle_value(arr):
    n = len(arr)
    if n % 2 == 1:  # Odd length
        return arr[n // 2]
    else:  # Even length
        return (arr[n // 2 - 1] + arr[n // 2]) / 2


# %%

G = random_planar_graph(4, seed=42)
nx.draw(G, with_labels=True)


# %%

od_matrix = np.array(
    [
        [-3, 9, -3, -3],
        [-3, -3, 9, -3],
        [-3, -3, -3, 9],
    ]
)

f_mat, lambda_mat = solve_multicommodity_tap(G, od_matrix, return_fw=True)
F = np.sum(f_mat, 0)

# %%

derivatives = derivative_social_cost(G, f_mat, od_matrix, eps=1e-3)

derivative_list = [derivatives.get(e, 0) for e in G.edges]
derivatives = dict(zip(G.edges, derivative_list))
derivatives


# %%


Kinv = inverse_coupling_matrix(G, f_mat, eps=1e-2)
K = generate_coupling_matrix(G, f_mat, eps=1e-2)

(Kinv @ K).todense()

# %%
eps = 1e-2

num_layers = len(f_mat)

f_vec = np.hstack(f_mat)
p_vec = od_matrix.flatten()
beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
beta_vec = np.array([beta_arr for _ in range(num_layers)]).flatten()
lambda_vec = np.hstack(np.diff(lambda_mat))

# %%

source_nodes = np.where(p_vec > 0)[0]

EE, _ = layered_edge_incidence_matrix(G, f_mat, eps=eps)  # .toarray()
kappa = inverse_coupling_matrix(G, f_mat, eps=eps)
LL = -EE @ kappa @ EE.T

# Efficiently delete rows and columns corresponding to source_nodes
mask = np.ones(LL.shape[0], dtype=bool)
mask[source_nodes] = False

LL_tilde = LL[mask][:, mask]  # Submatrix without source_nodes
EE_tilde = EE[mask, :]  # Efficient row deletion using slicing


D = np.linalg.inv(LL_tilde.toarray())

C = -D @ EE_tilde @ kappa


slopes = C.T @ p_vec[p_vec < 0]

# %%
pos_flow_edge_indices = np.where(f_vec > eps)[0]
full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

slope_edge_dict = {edge: 0 for edge in G.edges}
for i, idx in enumerate(pos_flow_edge_indices):
    # if key exsists add to it
    if full_edge_list[idx] in slope_edge_dict:
        slope_edge_dict[full_edge_list[idx]] += slopes[i]
    # if key does not exist create new key
    else:
        slope_edge_dict[full_edge_list[idx]] = slopes[i]

slope_edge_dict
# %%
