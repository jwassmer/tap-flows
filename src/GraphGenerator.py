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

    Ptot = dict(zip(G.nodes, np.zeros(G.number_of_nodes())))
    for (s, d), has_path in od_has_path.items():
        if has_path:
            for k, v in Ptot.items():
                if k == s:
                    Ptot[k] += total_load / num_od_pairs
                elif k == d:
                    Ptot[k] += -total_load / num_od_pairs

    return Ptot


def random_graph(
    source_nodes,
    target_nodes,
    total_load,
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

    U = nx.relabel_nodes(U, lambda x: chr(x + 97))
    U.source_nodes = source_nodes
    U.target_nodes = target_nodes
    U.total_load = total_load

    if alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, U.number_of_edges())
    if beta == "random":
        np.random.seed(seed)
        beta = np.random.randint(1, 10, U.number_of_edges())

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(U.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(U.number_of_edges())

    tt_func = {
        edge: (lambda alpha, beta: lambda n: alpha * n + beta)(alpha[e], beta[e])
        for e, edge in enumerate(U.edges)
    }
    # test = {edge: beta[e] for e, edge in enumerate(U.edges)}
    nx.set_edge_attributes(U, tt_func, "tt_function")

    if sum(alpha) > 0:
        weights = {
            edge: 1 / alpha[e] if alpha[e] != 0 else 0 for e, edge in enumerate(U.edges)
        }
    else:
        weights = {edge: 1 for edge in U.edges}
    nx.set_edge_attributes(U, weights, "weight")

    # G = to_directed(U)
    pos = nx.spring_layout(U, seed=seed)
    nx.set_node_attributes(U, pos, "pos")

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
    G = random_graph(sources, targets, 1000, beta="random", alpha="random")
    G = to_directed_flow_graph(G)
    pl.graphPlotCC(G, cc="flow")
    # %%

    f = nx.get_edge_attributes(G, "tt_function")
    betas = np.array([f[e](0) for e in G.edges()])
    alphas = np.array([f[e](1) - f[e](0) for e in G.edges()])
    print(betas, alphas)
# %%
