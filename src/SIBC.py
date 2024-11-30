# %%
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function


import time
import networkx as nx
import numpy as np


def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc


def _rescale_e(betweenness, normalized):
    """
    Rescale the edge SIBC values based on population and normalization options.

    Parameters
    ----------
        betweenness (dict): Dictionary of betweenness values.
        population (dict): Dictionary of node populations.
        normalized (bool): Flag indicating whether to normalize the betweenness values.

    Returns
    -------
        dict: The rescaled betweenness centrality values.
    """
    if normalized:
        # total_pop = sum(population.values())
        total_betweenness = sum(betweenness.values())
        betweenness = {k: v / total_betweenness for k, v in betweenness.items()}

    return betweenness


def _single_source_dijkstra_path_basic(G, s, weight, cutoff=None):
    """
    Compute the shortest paths and related information from a single source using Dijkstra's algorithm.

    Parameters
    ----------
        G (NetworkX graph): The graph in which to find the shortest paths.
        s (node): The source node from which to start the computation.
        weight (str or callable): The edge weight attribute or function used for the shortest path calculations.
        cutoff (float or None, optional): If specified, the algorithm will terminate if the shortest path
            to a node exceeds this cutoff value. Defaults to None.

    Returns
    -------
        tuple: A tuple containing the following elements:
            - S (list): List of nodes in the order they were visited during the search.
            - P (dict): Dictionary of lists of predecessors for each node.
            - sigma (dict): Dictionary of node counts representing the number of shortest paths.
            - D (dict): Dictionary of shortest path distances from the source node.
    """
    weight = _weight_function(G, weight)
    S = []
    P = {}
    for v in G:
        P[v] = []

    sigma = dict.fromkeys(G, 0.0)
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        if cutoff is not None:
            if dist > cutoff:
                continue

        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)

    return S, P, sigma, D


def _accumulate_edges(SIBC, seen, pred, sigma, P):
    """
    Accumulate the edge weights based on the populations, shortest path distances, and travel function.

    Parameters
    ----------
        SIBC (dict): Dictionary to store the spatial interaction edge betweennness centrality (SIBC).
        seen (list): List of nodes in the order they were visited.
        pred (dict): Dictionary of lists of predecessors for each node.
        sigma (dict): Dictionary of node counts representing the number of shortest paths.
        od_matrix (np.ndarray): Origin-Destination matrix.
    Returns
    -------
        dict: The updated SIBC dictionary with the accumulated edge weights.
    """
    delta = dict.fromkeys(seen, 0)
    source_node = seen[0]
    while seen:
        w = seen.pop()
        flow = P[w]
        # flow = od_matrix[source_node, w]
        coeff = (flow + delta[w]) / sigma[w]
        for v in pred[w]:
            c = sigma[v] * coeff
            if (v, w) not in SIBC:
                SIBC[(w, v)] += c
            else:
                SIBC[(v, w)] += c
            delta[v] += c
    return SIBC


def _interaction_betweenness_centrality(
    graph,
    weight="beta",
    normalized=False,
    cutoff=None,
    od_matrix=None,
):
    """
    Computes the spatial interaction betweenness centrality for each edge in the graph.

    Parameters
    ----------
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths. Either 'length or' 'travel_time.
                    'Default is 'travel_time'.
        normalized (bool): Flag indicating whether to normalize the betweenness centrality values. Default is True.
        cutoff (float or str): Cutoff value for the maximum shortest path length. Default is 'default'.
        cache (bool): Flag indicating whether to cache and reuse previously computed results. Default is True.
        return_graph (bool): Flag indicating whether to return the graph with updated edge attributes. Default is True.
        **kwargs: Additional keyword arguments for parallel computation.

    Returns
    -------
    dict
        A dictionary where keys are edges in the graph and values are dictionaries containing the accumulated
        spatial interaction betweenness centrality values for the respective edges.
    """
    if od_matrix is None:
        num_nodes = len(graph)
        od_matrix = 1 * np.ones((num_nodes, num_nodes))
    else:
        if not np.all(od_matrix >= 0):
            od_matrix = np.abs(od_matrix)

    start = time.time()

    SIBC = {e: 0 for e in graph.edges()}

    for o in graph.nodes:
        seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
            graph, o, weight, cutoff=cutoff
        )

        SIBC = _accumulate_edges(SIBC, seen, pred, sigma, P=od_matrix[:, o])

    SIBC = _rescale_e(SIBC, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)
    end = time.time()

    print("Single-threaded Time:", round(end - start, 1), "seconds")
    return np.array(list(SIBC.values()))


def _single_source_interaction_betweenness_centrality(
    graph,
    weight="beta",
    normalized=False,
    cutoff=None,
    P=None,
):
    """
    Computes the spatial interaction betweenness centrality for each edge in the graph.

    Parameters
    ----------
        graph (networkx.Graph): The input graph.
        weight (str): The edge weight attribute to use for computing shortest paths. Either 'length or' 'travel_time.
                    'Default is 'travel_time'.
        normalized (bool): Flag indicating whether to normalize the betweenness centrality values. Default is True.
        cutoff (float or str): Cutoff value for the maximum shortest path length. Default is 'default'.
        cache (bool): Flag indicating whether to cache and reuse previously computed results. Default is True.
        return_graph (bool): Flag indicating whether to return the graph with updated edge attributes. Default is True.
        **kwargs: Additional keyword arguments for parallel computation.

    Returns
    -------
    dict
        A dictionary where keys are edges in the graph and values are dictionaries containing the accumulated
        spatial interaction betweenness centrality values for the respective edges.
    """

    start = time.time()

    SIBC = {e: 0 for e in graph.edges()}

    source = np.where(np.array(P) > 0)[0]

    if len(source) != 1:
        raise ValueError("There must be exactly one source node.")

    seen, pred, sigma, dists = _single_source_dijkstra_path_basic(
        graph, source[0], weight, cutoff=cutoff
    )

    demands = np.abs(np.where(P <= 0, P, 0))

    SIBC = _accumulate_edges(SIBC, seen, pred, sigma, P=demands)

    SIBC = _rescale_e(SIBC, normalized)

    if graph.is_multigraph():
        SIBC = _add_edge_keys(graph, SIBC, weight=weight)
    end = time.time()

    print("Single-threaded Time:", round(end - start, 1), "seconds")
    return np.array(list(SIBC.values()))


# %%
if __name__ == "__main__":

    from src import TAPOptimization as tap
    import cvxpy as cp
    from src import Plotting as pl
    from src import SocialCost as sc

    G = nx.cycle_graph(6)
    G = nx.to_directed(G)
    G = G.copy()
    E = -nx.incidence_matrix(G, oriented=True)
    nx.set_edge_attributes(G, 0, "alpha")
    nx.set_edge_attributes(G, 1, "beta")
    G.edges[(0, 5)]["beta"] = 3
    pos = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (4, 1), 5: (5, 0)}
    nx.set_node_attributes(G, pos, "pos")

    source = 0
    sink = 4
    P = np.zeros(G.number_of_nodes())
    P[source] = 10
    P[sink] = -10
    # %%

    f = tap.user_equilibrium(G, P, positive_constraint=True, solver=cp.SCS)
    s = _single_source_interaction_betweenness_centrality(G, weight="beta", P=P)

    {e: np.round(v, 2) for e, v in zip(G.edges, f)}

    pl.graphPlot(G, ec=f, show_labels=True)
    # %%
    sc.total_social_cost(G, f)
    # %%
    sc.total_social_cost(G, s)
    # %%
