# %%
import networkx as nx
import numpy as np
from networkx.algorithms.cycles import simple_cycles
from src import Plotting as pl


def random_graph(
    num_nodes=10,
    num_edges=15,
    seed=42,
    alpha=1,
    beta=0,
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

    G = U.to_directed()

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
        beta = np.random.normal(1, 1e-1, G.number_of_edges())

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


def cycle_edge_incidence_matrix(graph):
    # Get all the cycles in the graph
    cycles = list(simple_cycles(graph))

    # Number of edges and cycles
    edges = list(graph.edges())
    num_edges = len(edges)
    num_cycles = len(cycles)

    # Create the incidence matrix
    incidence_matrix = np.zeros((num_cycles, num_edges), dtype=int)

    for i, edge in enumerate(edges):
        for j, cycle in enumerate(cycles):
            if is_tuple_in_array(edge, cycle):
                incidence_matrix[j, i] = 1
            elif is_tuple_in_array((edge[1], edge[0]), cycle):
                incidence_matrix[j, i] = -1

    return incidence_matrix


# %%
if __name__ == "__main__":
    G = random_graph(20, 5, alpha=1, beta=3)
    pl.graphPlot(G)
    C = cycle_edge_incidence_matrix(G)
    print(C)
    print(list(simple_cycles(G)))
