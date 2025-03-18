# %%
import networkx as nx
import numpy as np
from networkx.algorithms.cycles import simple_cycles
from src import Plotting as pl
from scipy.spatial import Delaunay
from geopy.distance import geodesic


def remove_parallel_edges(G):

    # Create a new graph to hold unique edges

    new_G = nx.MultiDiGraph()

    for u, v, keys in G.edges(keys=True):

        if not new_G.has_edge(u, v):

            new_G.add_edge(u, v, key=keys, **G[u][v][keys])

    return new_G


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
    G = remove_parallel_edges(G)

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


def random_graph(
    num_nodes=10,
    num_edges=15,
    seed=42,
    alpha=1,
    beta=0,
    directed=True,
):
    connected = False
    if num_edges < num_nodes - 1:
        num_edges = num_nodes - 1

    while not connected:
        U = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
        connected = nx.is_connected(U)
        num_edges += 1

    if isinstance(alpha, str) and alpha == "random_symmetric":
        # doesnt work atm
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, U.number_of_edges())
    if isinstance(beta, str) and beta == "random_symmetric":
        # doesnt work atm
        np.random.seed(seed)
        beta = 100 * np.random.rand(U.number_of_edges())

    if directed:
        G = U.to_directed()
    else:
        G = U

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())
    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(G.number_of_edges())

    # nx.set_edge_attributes(G, tt_func, "tt_function")
    nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    nx.set_edge_attributes(G, dict(zip(G.edges, beta)), "beta")

    pos = nx.spring_layout(G, seed=seed)
    nx.set_node_attributes(G, pos, "pos")

    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


def triangularLattice(radius, alpha=1, beta=0, seed=42, directed=True):
    G = nx.Graph()

    # Add nodes in a triangular grid pattern
    for x in range(-radius, radius + 1):
        for y in range(max(-radius, -x - radius), min(radius, -x + radius) + 1):
            G.add_node((x, y))
            if (x - 1, y) in G:
                G.add_edge((x, y), (x - 1, y))
            if (x, y - 1) in G:
                G.add_edge((x, y), (x, y - 1))
            if (x - 1, y + 1) in G:
                G.add_edge((x, y), (x - 1, y + 1))

    if directed:
        G = G.to_directed()
    # Calculate positions for drawing
    pos = {
        node: (node[0] + 0.5 * node[1], node[1] * (3**0.5 / 2)) for node in G.nodes()
    }
    nx.set_node_attributes(G, pos, "pos")

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())
    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.normal(1, 1e-1, G.number_of_edges())

    nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    nx.set_edge_attributes(G, dict(zip(G.edges, beta)), "beta")

    return G


def squareLattice(n, alpha=1, beta=0, seed=42, directed=True):
    G = nx.grid_2d_graph(n, n)
    if directed:
        G = G.to_directed()
    pos = {node: node for node in G.nodes()}
    nx.set_node_attributes(G, pos, "pos")

    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())
    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = np.random.normal(1, 1e-1, G.number_of_edges())

    nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    nx.set_edge_attributes(G, dict(zip(G.edges, beta)), "beta")

    return G


def is_tuple_in_array(tup, array):
    n = len(array)

    # Check if tuple is in the array in normal order
    for i in range(n - 1):
        if array[i] == tup[0] and array[i + 1] == tup[1]:
            return True

    # Check for cyclic case
    if len(array) > 2:
        if array[-1] == tup[0] and array[0] == tup[1]:
            return True

    return False


def cycle_basis_directed(graph):
    # Step 1: Convert directed graph to undirected graph
    undirected_graph = graph.to_undirected()

    # Step 2: Compute cycle basis of the undirected graph
    cycle_basis = nx.cycle_basis(undirected_graph)

    for e in undirected_graph.edges:
        cycle_basis.append([e[0], e[1]])

    return cycle_basis


def cycle_edge_incidence_matrix(gra):
    """Construct the edge cycle incidence matrix from a given networkx.MultiDiGraph.
    C_{e, c} = 1, iff edge is in the oriented cycle c and -1 if the edge is in the opposite direction.

    Args:
        gra (networkx.Graph): Graph encoding the traffic network

    Returns:
        C_matrix, simple_cycles
    """
    # Create a list of all simple cycles
    simple_cycles = cycle_basis_directed(gra)
    # add multiedge cycles
    # simple_cycles = nx.cycle_basis(gra.to_undirected())

    C_matrix = np.zeros((len(gra.edges), len(simple_cycles)))

    for cc, cycle in enumerate(simple_cycles):
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            edge = (i, j)
            # reversed_edge = (j, i)

            if edge in gra.edges:
                edge_idx = list(gra.edges).index(edge)
                C_matrix[edge_idx, cc] = 1
            # if reversed_edge in gra.edges:
            #    edge_idx = list(gra.edges).index(reversed_edge)
            #    C_matrix[edge_idx, cc] = -1

    simple_cycles = [tuple(cycle) for cycle in simple_cycles]
    return C_matrix, simple_cycles


def cycle_edge_incidence_matrix__(G):
    """
    Compute the cycle edge incidence matrix C for a given directed graph G.

    Parameters:
    - G (networkx.DiGraph): A directed graph.

    Returns:
    - C (numpy.ndarray): The cycle edge incidence matrix.
    """
    # Get the number of edges
    m = G.number_of_edges()

    # Find a spanning tree of the graph
    T = nx.minimum_spanning_tree(G.to_undirected())

    # Identify fundamental cycles by adding back the edges not in the spanning tree
    fundamental_cycles = []
    remaining_edges = set(G.edges()) - set(T.edges())

    # Mapping edges to their indices
    edge_index = {edge: idx for idx, edge in enumerate(G.edges())}

    for edge in remaining_edges:
        # Add the edge back to the tree to form a cycle
        H = T.copy()
        H.add_edge(*edge)
        try:
            # Find the cycle formed by this addition
            cycle = nx.find_cycle(H, orientation="ignore")
            fundamental_cycles.append(cycle)
        except nx.NetworkXNoCycle:
            # No cycle found, continue
            continue

    # Create the cycle edge incidence matrix C
    f = len(fundamental_cycles)  # Number of fundamental cycles
    C = np.zeros((m, f))

    # Fill the cycle edge incidence matrix C with direction handling
    for cycle_idx, cycle in enumerate(fundamental_cycles):
        for u, v, direction in cycle:
            edge = (u, v) if (u, v) in edge_index else (v, u)
            edge_pos = edge_index[edge]

            # Check the direction to assign +1 or -1
            if direction == "forward":  # Edge is in the same direction as the cycle
                C[edge_pos, cycle_idx] = 1
            else:  # Edge is in the opposite direction to the cycle
                C[edge_pos, cycle_idx] = -1

    fundamental_cycles = [tuple(cycle) for cycle in fundamental_cycles]
    return C, fundamental_cycles


def flow_subgraph(G, f, eps=1e-4):
    Gs = nx.DiGraph()
    Gs.add_nodes_from(G.nodes)

    for e, v in f.items():
        if v > eps:
            Gs.add_edge(*e, alpha=G.edges[e]["alpha"], beta=G.edges[e]["beta"])
        if v < -eps:
            l = e[::-1]
            Gs.add_edge(*l, alpha=G.edges[l]["alpha"], beta=G.edges[l]["beta"])

    pos = nx.get_node_attributes(G, "pos")
    nx.set_node_attributes(Gs, pos, "pos")

    return Gs


def potential_subgraph(G, few):
    U = G.copy()

    for i, e in enumerate(G.edges):
        if few[i] < 1e-2:
            U.remove_edge(*e)
            # A[n, m] = 1
        else:
            U.edges[e]["flow"] = few[i]
    return U


def directed_laplacian(G):
    E = nx.incidence_matrix(G, oriented=True)
    alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    L = E @ np.diag(1 / alpha_arr) @ E.T
    return L


# %%
if __name__ == "__main__":
    G = random_graph(10, 5, alpha=1, beta=3)
    pl.graphPlot(G)
    C, cycles = cycle_edge_incidence_matrix(G)
    E = -nx.incidence_matrix(G, oriented=True)
    # print(C)
    # print(list(simple_cycles(G)))

    print(np.all(np.abs(E @ C < 1e-7)))
# %%
