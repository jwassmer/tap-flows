# %%
import pickle
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
from scipy.sparse.linalg import splu
import tqdm


from sksparse.cholmod import cholesky
from scipy.sparse import block_diag, lil_matrix, diags, bmat

from scipy.linalg import eigvals

# from src import Plotting as pl


def create_braess_graph():
    """
    Create a simple, but asymmertic graph which has a braess edge.
    """
    G = nx.MultiGraph()
    G.add_edge(0, 1, beta=1.1, alpha=9)
    G.add_edge(0, 2, beta=49, alpha=0.9)
    G.add_edge(1, 2, beta=0.95, alpha=10.05)
    G.add_edge(2, 3, beta=1.2, alpha=12)
    G.add_edge(1, 3, beta=51, alpha=1)
    G = G.to_directed()
    return G


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

    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
        nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    elif isinstance(alpha, float) or isinstance(alpha, int):
        nx.set_edge_attributes(G, alpha, "alpha")
    elif isinstance(alpha, np.ndarray):
        nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    else:
        assert "Type of alpha not allowed"

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


# Only used for testing not in final computation
def generate_coupling_matrix(G, edge_mask):
    """
    Generate the D1 matrix for a given number of sources and edges.

    Parameters:
        G (networkx.Graph): The graph with edge attribute "alpha".
        edge_mask (np.ndarray): The flat edge mask whitch excludes edges with no flows

    Returns:
        scipy.sparse.csr_matrix: The resulting sparse D1 matrix.
    """
    # Extract alpha values from the graph's edges
    alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    num_sources = len(f_mat)  # Number of sources (N)

    # Create a sparse diagonal matrix for alpha
    alpha_diag = diags(alpha, format="csr")

    # Construct the full block matrix as a sparse grid
    D1_blocks = [[alpha_diag for j in range(num_sources)] for i in range(num_sources)]
    D1 = bmat(D1_blocks, format="csr")

    return D1[edge_mask][:, edge_mask]


def inverse_coupling_matrix_new(G, stacked_edge_mask, delta=1e-2):
    """
    Create the inverse matrix of the K block in the original problem.

    Parameters:
        G (networkx.Graph): The graph with edge attribute "alpha".
        stacked_edge_mask (np.ndarray): The stacked edge mask whitch excludes edges with no flows
        delta (float): The original K matrix is not invertable therefore the inverse of K + delta 1 is returned.
    """
    kappa_x_delta = get_kappa_x_delta(G, stacked_edge_mask, delta=delta)
    return 1 / delta * kappa_x_delta


def get_kappa_x_delta(G, stacked_edge_mask, delta=1e-2):
    """
    Create the inverse matrix of the K block in the original problem times delta.

    Parameters:
        G (networkx.Graph): The graph with edge attribute "alpha".
        stacked_edge_mask (np.ndarray): The stacked edge mask whitch excludes edges with no flows
        delta (float): The original K matrix is not invertable therefore the inverse of K + delta 1 is returned.
    """
    num_layers = len(stacked_edge_mask)
    num_edges = G.number_of_edges()
    alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))

    rows_per_column = {
        col: np.where(stacked_edge_mask[:, col])[0]
        for col in range(stacked_edge_mask.shape[1])
    }

    kappa = lil_matrix((num_layers * num_edges, num_layers * num_edges))

    for key, value in rows_per_column.items():
        for i in value:
            for j in value:
                if i == j:
                    kappa[i * num_edges + key, j * num_edges + key] = 1 - alpha_[
                        key
                    ] / (delta + alpha_[key] * len(value))

                else:
                    kappa[i * num_edges + key, j * num_edges + key] = -alpha_[key] / (
                        delta + alpha_[key] * len(value)
                    )

    # Convert kappa to CSR format for efficient row and column operations
    kappa_csr = kappa.tocsr()

    # Keep only non-zero rows and columns
    kappa_reduced = kappa_csr[stacked_edge_mask.flatten()][
        :, stacked_edge_mask.flatten()
    ]
    return kappa_reduced  # .toarray()


def layered_edge_incidence_matrix(G, note_mask, edge_mask):
    """
    Generate the layered edge incidence matrix for a given graph and number of layers.

    Parameters:
        G (nx.DiGraph): The graph.
        note_mask (np.ndarray): The flat edge mask whitch excludes notes
        edge_mask (np.ndarray): The flat edge mask whitch excludes edges with no flows

    Returns:
        csr_array: The resulting layered edge incidence matrix.
        np.ndarray: The updated note_mask
    """
    num_layers = len(edge_mask) // G.size()
    E = -nx.incidence_matrix(G, oriented=True)  # Sparse matrix in CSC format

    # Construct the sparse block matrix
    EE_sparse = block_diag([E] * num_layers, format="csr")[:, edge_mask]

    unconnected_notes = np.diff(EE_sparse.indptr) == 0
    note_mask[unconnected_notes] = False
    EE_sparse = EE_sparse[note_mask]

    return EE_sparse.astype(int), note_mask


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

    print(flows)

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


def derivative_social_cost_new(G, f_mat, od_matrix, eps=1e-3, reg=1e-5):
    """
    Calculate all derivatives

    Parameters:
        G (nx.DiGraph): The graph.
        f_mat (np.ndarray): The flow on every edge of every subgraph from the numerical solver.
        od_matrix (np.ndarray): The origin destination matrix.
        eps (float): Threshold defining which flow on an edge is
           considered no flow, these edges are removed from the corresponding subgraph.
        reg (float): Small number which is used to make the problem invertable.

    """

    num_layers = len(f_mat)
    p_vec = od_matrix.flatten()

    edge_mask_stacked = f_mat > eps
    edge_mask = edge_mask_stacked.flatten()
    source_nodes = np.where(p_vec > 0)[0]
    note_mask_raw = np.ones(len(G) * num_layers, dtype=bool)
    note_mask_raw[source_nodes] = False

    EE, note_mask = layered_edge_incidence_matrix(G, note_mask_raw, edge_mask)

    # K = generate_coupling_matrix(G, edge_mask)
    # K = K + diags(np.full(K.shape[0], reg))
    kappa_x_delta = get_kappa_x_delta(G, edge_mask_stacked, delta=reg)

    mLL_x_delta = EE @ kappa_x_delta @ EE.T
    p_filtered = p_vec[note_mask]

    # Use only scipy
    # solveA = splu(mLL_x_delta, permc_spec="COLAMD").solve
    # approx 4x faster than splu
    solveA = cholesky(mLL_x_delta).solve_A
    blocksize = 512
    n_LL = mLL_x_delta.shape[0]
    p_x_D = np.empty(n_LL)
    for i in tqdm.tqdm(range(0, n_LL, blocksize)):
        i_start = i
        i_end = min(i_start + blocksize, n_LL)
        rhs = np.zeros((n_LL, i_end - i_start))
        rhs[np.arange(i_start, i_end), np.arange(i_end - i_start)] = 1
        D_prime_cols = solveA(-rhs)  # minues from lhs to make LL positive definit
        p_x_D[i_start:i_end] = p_filtered @ D_prime_cols

    slopes = -p_x_D @ EE @ kappa_x_delta
    pos_flow_edge_indices = np.where(edge_mask)[0]
    full_edge_list = [edge for _ in range(num_layers) for edge in G.edges]

    slope_edge_dict = {edge: 0 for edge in G.edges}
    for i, idx in enumerate(pos_flow_edge_indices):
        # if key exsists add to it
        if full_edge_list[idx] in slope_edge_dict:
            slope_edge_dict[full_edge_list[idx]] += slopes[i]
        # if key does not exist create new key
        else:
            slope_edge_dict[full_edge_list[idx]] = slopes[i].item()

    return slope_edge_dict


# %%

eps = 1e-3
reg = 1e-5

case = 2
save = True

if case == 0:  # Random but small planar graph
    # random but fixed alpha
    alpha = np.array(
        [
            0.4370861069626263,
            0.9556428757689246,
            0.7587945476302645,
            0.6387926357773329,
            0.24041677639819287,
            0.2403950683025824,
            0.15227525095137953,
            0.8795585311974417,
            0.6410035105688879,
            0.737265320016441,
        ]
    )
    G = random_planar_graph(4, seed=42, alpha=alpha)  # alpha=1 / 2

    od_matrix = np.array(
        [
            [-3, 9, -3, -3],
            [-1, -1, 3, -1],
            [-6, -6, -6, 18],
        ]
    )
elif case == 1:  # Braess Graph
    G = create_braess_graph()
    od_matrix = np.array(
        [[6, -1 / 3, -2 / 3, -5], [4, -2 / 3, -1 / 3, -3], [-2, -2, -2, 6]]
    )
elif case == 2:  # Large Graph
    n = 835
    n_o = 5  # number origins
    G = random_planar_graph(n, seed=40, alpha="random")
    od_matrix = -50 / n_o * np.random.rand(n_o, n) - 1e-2
    od_matrix[np.arange(n_o), np.arange(n_o)] = 0
    od_matrix[np.arange(n_o), np.arange(n_o)] = -np.sum(od_matrix, 1)

# nx.draw(G, with_labels=True)
print("Notes", len(G), "Edges", G.size())
# %%
ts = time.time()
f_mat, lambda_mat = solve_multicommodity_tap(G, od_matrix, return_fw=True)
F = np.sum(f_mat, 0)
t_num = time.time() - ts
print("Solved numerically in", t_num)

# %%
ts = time.time()
slope_edge_dict = derivative_social_cost_new(G, f_mat, od_matrix, eps, reg)
t_inv = time.time() - ts
print("Time to invert ", t_inv)

# %%
if save:
    with open(f"save_n{n}_od{n_o}.pkl", "wb") as f:
        pickle.dump(
            {
                "G": G,
                "od_matrix": od_matrix,
                "reg": reg,
                "eps": eps,
                "slope_edge_dict": slope_edge_dict,
                "f_mat": f_mat,
                "lambda_mat": lambda_mat,
                "F": F,
                "t_num": t_num,
                "t_inv": t_inv,
            },
            f,
        )
