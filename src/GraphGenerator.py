# %%
import networkx as nx
import numpy as np
from src import Equilibirium as eq
from src import Plotting as pl


def random_graph(
    source_nodes, target_nodes, total_load, num_nodes=10, num_edges=15, seed=42
):
    connected = False
    while not connected:
        U = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
        connected = nx.is_connected(U)

    U = nx.relabel_nodes(U, lambda x: chr(x + 97))

    tt_func = {edge: lambda n: n for edge in U.edges}
    nx.set_edge_attributes(U, tt_func, "tt_function")
    weights = {edge: 1 for edge in U.edges}
    nx.set_edge_attributes(U, weights, "weight")

    # G = to_directed(U)
    pos = nx.spring_layout(U, seed=seed)
    nx.set_node_attributes(U, pos, "pos")

    nodes_dict = dict(zip(U.nodes, [i for i in range(U.number_of_nodes())]))
    P = np.zeros(U.number_of_nodes())

    start_node_int = [nodes_dict[s] for s in source_nodes]
    end_node_int = [nodes_dict[t] for t in target_nodes]

    P = np.zeros(U.number_of_nodes())
    P[start_node_int], P[end_node_int] = total_load / len(
        source_nodes
    ), -total_load / len(target_nodes)
    nx.set_node_attributes(U, dict(zip(U.nodes, P)), "P")

    F = eq.linear_flow(U)
    nx.set_edge_attributes(U, F, "flow")

    nx.set_edge_attributes(U, "black", "color")
    nx.set_node_attributes(U, "lightgrey", "color")

    for s in source_nodes:
        U.nodes[s]["color"] = "lightblue"
    for t in target_nodes:
        U.nodes[t]["color"] = "red"
    planar = nx.check_planarity(U)
    if planar[0]:
        print("The graph is planar")
    else:
        print("The graph is NOT planar")

    U.source_nodes = source_nodes
    U.target_nodes = target_nodes
    U.total_load = total_load
    return U


def to_directed_flow_graph(G):
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, **d)
    for i, j, d in G.edges(data=True):
        f = d["flow"]
        if f >= 0:
            D.add_edge(i, j, **d)
        elif f < 0:
            D.add_edge(j, i, **d)
            D[j][i]["flow"] = -f
    D.source_nodes = G.source_nodes
    D.target_nodes = G.target_nodes
    D.total_load = G.total_load
    return D


# %%
if __name__ == "__main__":
    sources = ["a", "b"]
    targets = ["c", "h"]
    G = random_graph(sources, targets, 1000)
    G = to_directed_flow_graph(G)
    pl.graphPlotCC(G)
# %%
