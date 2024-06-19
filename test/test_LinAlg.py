# %%
from src import LinAlg as la
from src import ConvexOptimization as co
import networkx as nx
import numpy as np
import numpy as np
import networkx as nx
import pytest


def positive_flow_graph(G):
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, **d)
    for i, j, d in G.edges(data=True):
        flow = d["flow"]
        if flow >= 0:
            D.add_edge(i, j, **d)
        else:
            D.add_edge(j, i, **d)
            D[j][i]["flow"] = -flow
    D.source_nodes = G.source_nodes
    D.target_nodes = G.target_nodes
    D.total_load = G.total_load
    return D


def generate_graph(source_node, target_node, total_load, p=0.3, seed=42, num_nodes=10):
    G = nx.erdos_renyi_graph(num_nodes, p, seed=seed, directed=False)
    G.source_nodes = source_node
    G.target_nodes = target_node
    G.total_load = total_load
    weights = dict(zip(G.edges, np.random.rand(G.number_of_edges())))
    nx.set_edge_attributes(G, weights, "weight")
    P = np.zeros(G.number_of_nodes())
    P[source_node] = total_load / len(source_node)
    P[target_node] = -total_load / len(target_node)
    Pdict = dict(zip(G.nodes, P))
    nx.set_node_attributes(G, Pdict, "P")
    F = co.convex_optimization_linflow(G, P, np.array(list(weights.values())))
    nx.set_edge_attributes(G, dict(zip(G.edges, F)), "flow")

    return positive_flow_graph(G)


def calculate_matrices(G):
    Pvec = np.array(list(nx.get_node_attributes(G, "P").values()))
    F = nx.get_edge_attributes(G, "flow")
    weights = list(nx.get_edge_attributes(G, "weight").values())
    K = np.diag(weights)
    L = nx.laplacian_matrix(G, weight="weight").toarray()
    I = -nx.incidence_matrix(G, oriented=True, weight="weight").toarray()
    psi = np.linalg.pinv(L) @ Pvec
    return Pvec, F, K, L, I, psi


def validate_edge_incidence(psi, I, F):
    return any(np.abs(I.T @ psi - np.array(list(F.values()))) <= 1e-5)


def validate_path_link_incidence(path_link_matrix, pathflow, F):
    return any(
        np.abs(
            path_link_matrix @ np.array(list(pathflow.values()))
            - np.array(list(F.values()))
        )
        < 1e-7
    )


def validate_od_path_incidence(od_path_matrix, pathflow, total_load):
    return (
        np.abs(sum(od_path_matrix @ np.array(list(pathflow.values()))) - total_load)
        < 1e-7
    )


@pytest.fixture
def setup_graph():
    source_node = [0, 2]
    target_node = [1, 3]
    total_load = 1000
    G = generate_graph(source_node, target_node, total_load)
    path_link_matrix = la.path_link_incidence_matrix(G)
    od_path_matrix = la.od_path_incidence_matrix(G)
    cycle_link_matrix = la.cycle_link_incidence_matrix(G)
    Pvec, F, K, L, I, psi = calculate_matrices(G)
    pathflow = la.path_flows(G)
    return {
        "source_node": source_node,
        "target_node": target_node,
        "total_load": total_load,
        "G": G,
        "path_link_matrix": path_link_matrix,
        "od_path_matrix": od_path_matrix,
        "cycle_link_matrix": cycle_link_matrix,
        "Pvec": Pvec,
        "F": F,
        "K": K,
        "L": L,
        "I": I,
        "psi": psi,
        "pathflow": pathflow,
    }


def test_edge_incidence(setup_graph):
    psi = setup_graph["psi"]
    I = setup_graph["I"]
    F = setup_graph["F"]
    assert validate_edge_incidence(psi, I, F), "Edge incidence validation failed"


def test_path_link_incidence(setup_graph):
    path_link_matrix = setup_graph["path_link_matrix"]
    pathflow = setup_graph["pathflow"]
    F = setup_graph["F"]
    assert validate_path_link_incidence(
        path_link_matrix, pathflow, F
    ), "Path link incidence validation failed"


def test_od_path_incidence(setup_graph):
    od_path_matrix = setup_graph["od_path_matrix"]
    pathflow = setup_graph["pathflow"]
    total_load = setup_graph["total_load"]
    assert validate_od_path_incidence(
        od_path_matrix, pathflow, total_load
    ), "OD path incidence validation failed"


# %%

if __name__ == "__main__":
    source_node = [0, 2]
    target_node = [1, 3]
    total_load = 1000
    G = generate_graph(source_node, target_node, total_load)
    path_link_matrix = la.path_link_incidence_matrix(G)
    od_path_matrix = la.od_path_incidence_matrix(G)
    cycle_link_matrix = la.cycle_link_incidence_matrix(G)
    Pvec, F, K, L, I, psi = calculate_matrices(G)
    pathflow = la.path_flows(G)
# %%
