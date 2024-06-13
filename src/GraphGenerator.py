# %%
import networkx as nx
import numpy as np
import itertools
from src import Equilibirium as eq
from src import Plotting as pl


def source_sink_dict(G):
    sources = G.source_nodes
    targets = G.target_nodes
    total_load = G.total_load

    num_od_pairs = 0
    od_has_path = {}
    for s in sources:
        for d in targets:
            if nx.has_path(G, s, d):
                num_od_pairs += 1
                od_has_path[(s, d)] = True
            else:
                od_has_path[(s, d)] = False

    # pl.graphPlotCC(G, cc="flow")

    Ptot = dict(zip(G.nodes, np.zeros(G.number_of_nodes())))
    for (s, d), has_path in od_has_path.items():
        if has_path:
            # print(f"Path from {s} to {d} exists")
            # P = nx.get_node_attributes(G, "P")
            for k, v in Ptot.items():
                if k == s:
                    Ptot[k] += total_load / num_od_pairs
                elif k == d:
                    Ptot[k] += -total_load / num_od_pairs

        # else:
        #    print(f"Path from {s} to {d} does not exist")

    return Ptot


def random_graph(
    source_nodes, target_nodes, total_load, num_nodes=10, num_edges=15, seed=42
):
    connected = False
    if num_edges < num_nodes - 1:
        num_edges = num_nodes - 1

    while not connected:
        U = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
        connected = nx.is_connected(U)
        num_edges += 1

    U = nx.relabel_nodes(U, lambda x: chr(x + 97))
    U.source_nodes = source_nodes
    U.target_nodes = target_nodes
    U.total_load = total_load

    tt_func = {edge: lambda n: n for edge in U.edges}
    nx.set_edge_attributes(U, tt_func, "tt_function")
    weights = {edge: 1 for edge in U.edges}
    nx.set_edge_attributes(U, weights, "weight")

    # G = to_directed(U)
    pos = nx.spring_layout(U, seed=seed)
    nx.set_node_attributes(U, pos, "pos")

    # nodes_dict = dict(zip(U.nodes, [i for i in range(U.number_of_nodes())]))

    # start_node_int = [nodes_dict[s] for s in source_nodes]
    # end_node_int = [nodes_dict[t] for t in target_nodes]

    P = source_sink_dict(U)
    nx.set_node_attributes(U, P, "P")

    F = eq.linear_flow(U)
    nx.set_edge_attributes(U, F, "flow")

    nx.set_edge_attributes(U, "black", "color")
    nx.set_node_attributes(U, "lightgrey", "color")

    for s in source_nodes:
        U.nodes[s]["color"] = "lightblue"
    for t in target_nodes:
        U.nodes[t]["color"] = "red"

    return U


def to_directed_flow_graph(G):
    F = nx.get_edge_attributes(G, "flow")
    num_negative_flows = np.sum([f < 0 for f in F.values()])
    nodes = G.nodes(data=True)
    edges = G.edges(data=True)
    while num_negative_flows > 0:
        D = nx.DiGraph()
        for n, d in nodes:
            D.add_node(n, **d)
        for i, j, d in edges:
            fval = F[(i, j)]
            if fval >= 0:
                D.add_edge(i, j, **d)
            elif fval < 0:
                D.add_edge(j, i, **d)
                # D[j][i]["flow"] = -f

        D.source_nodes = G.source_nodes
        D.target_nodes = G.target_nodes
        D.total_load = G.total_load

        P = source_sink_dict(D)
        nx.set_node_attributes(D, P, "P")

        F = eq.linear_flow(D)
        nx.set_edge_attributes(D, F, "flow")
        num_negative_flows = np.sum([f < 0 for f in F.values()])
        edges = D.edges(data=True)
        # print("foo")

    return D


# %%
if __name__ == "__main__":
    sources = ["a", "b"]
    targets = ["c", "h"]
    G = random_graph(sources, targets, 1000)
    G = to_directed_flow_graph(G)
    pl.graphPlotCC(G, cc="flow")
# %%
