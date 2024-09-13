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

    # simple_cycles = nx.recursive_simple_cycles(gra)

    C_matrix = np.zeros((len(gra.edges), len(simple_cycles)))

    for cc, cycle in enumerate(simple_cycles):
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            edge = (i, j)
            reversed_edge = (j, i)

            if edge in gra.edges:
                edge_idx = list(gra.edges).index(edge)
                C_matrix[edge_idx, cc] = 1
            # if reversed_edge in gra.edges:
            #    edge_idx = list(gra.edges).index(reversed_edge)
            #    C_matrix[edge_idx, cc] = -1

    return C_matrix, simple_cycles


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


# %%
if __name__ == "__main__":
    G = random_graph(20, 5, alpha=1, beta=3)
    pl.graphPlot(G)
    C = cycle_edge_incidence_matrix(G)
    print(C)
    print(list(simple_cycles(G)))
