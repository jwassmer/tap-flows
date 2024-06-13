# %%
import networkx as nx
from itertools import chain
import numpy as np
from scipy.optimize import linprog
import itertools


# %%


def all_od_paths(G):
    sources = G.source_nodes
    targets = G.target_nodes
    all_paths = []
    for s in sources:
        for t in targets:
            od_path = [tuple(path) for path in nx.all_simple_paths(G, s, t)]
            all_paths.extend(od_path)
    return all_paths


def path_flows(G):
    edges_with_flows = nx.get_edge_attributes(G, "flow")

    paths = all_od_paths(G)
    flows = np.array(list(edges_with_flows.values()))

    # Set up the linear programming problem
    c = np.zeros(
        len(paths)
    )  # Objective function coefficients (we want any feasible solution)
    A_eq = path_link_incidence_matrix(G)

    # Solve the linear programming problem
    res = linprog(c, A_eq=A_eq, b_eq=flows, bounds=(0, None), method="highs")

    # path_flows = {tuple(paths[i]): res.x[i] for i in range(len(paths))}
    if res.success:
        path_flows = res.x
        # Output the path flows
        path_flow_dict = {tuple(path): path_flows[i] for i, path in enumerate(paths)}
    else:
        path_flow_dict = None
        print("No solution found.")
    return path_flow_dict


def is_subsequence(A, B):
    return any(A == B[i : len(A) + i] for i in range(len(B) - len(A) + 1))


def path_link_incidence_matrix(G):
    links = G.edges
    all_paths = all_od_paths(G)

    path_link_matrix = np.zeros((len(links), len(all_paths)))
    for i, path in enumerate(all_paths):
        for j, link in enumerate(links):
            if is_subsequence(link, path):
                path_link_matrix[j, i] = 1

    return path_link_matrix


def od_path_incidence_matrix(G):
    sources = G.source_nodes
    targets = G.target_nodes

    all_paths = all_od_paths(G)
    od_pairs = list(itertools.product(sources, targets))

    od_path_incidence_matrix = np.zeros((len(od_pairs), len(all_paths)))

    for i, path in enumerate(all_paths):
        for j, od in enumerate(od_pairs):
            if path[0] == od[0] and path[-1] == od[1]:
                od_path_incidence_matrix[j, i] = 1
    return od_path_incidence_matrix


def cycle_link_incidence_matrix(G):
    if nx.is_directed(G):
        U = G.to_undirected()
    else:
        U = G
    cycle_basis = nx.minimum_cycle_basis(U)
    links = G.edges()

    cycle_edge_incidence_matrix = np.zeros((len(cycle_basis), len(links)))
    for i, cycle in enumerate(cycle_basis):
        cycle = tuple(cycle + [cycle[0]])
        for j, link in enumerate(links):
            if is_subsequence(link, cycle):
                cycle_edge_incidence_matrix[i, j] = 1
            elif is_subsequence(link[::-1], cycle):
                cycle_edge_incidence_matrix[i, j] = -1

    return cycle_edge_incidence_matrix


# %%

if __name__ == "__main__":
    from src import Equilibirium as eq
    from src import GraphGenerator as gg
    from src import Plotting as pl

    source_node, target_node = ["a", "b"], ["c", "j"]
    total_load = 1000

    U = gg.random_graph(
        source_node, target_node, total_load, seed=42, num_nodes=10, num_edges=15
    )
    G = gg.to_directed_flow_graph(U)
    pl.graphPlotCC(G)

    # %%

    path_link_matrix = path_link_incidence_matrix(G)
    od_path_matrix = od_path_incidence_matrix(G)
    cycle_link_matrix = cycle_link_incidence_matrix(G)
    # %%

    P = nx.get_node_attributes(G, "P")
    Pvec = np.array(list(nx.get_node_attributes(G, "P").values()))

    F = nx.get_edge_attributes(G, "flow")
    weights = list(nx.get_edge_attributes(U, "weight").values())
    K = np.diag(weights)
    L = nx.laplacian_matrix(U, weight="weight").toarray()
    I = -nx.incidence_matrix(U, oriented=True, weight="weight").toarray()

    psi = np.linalg.pinv(L) @ Pvec

    # %%
    if any(I.T @ psi - list(F.values()) <= 1e-5):
        print(r"Edge incidence @ psi yield flows", "\u2713")
    else:
        print("ERROR")

    pathflow = path_flows(G)

    if any(path_link_matrix @ list(pathflow.values()) - list(F.values()) < 1e-7):
        print("Path link incidence matrix @ pathflows yield flows", "\u2713")
    else:
        print("ERROR")

    if ((od_path_matrix @ list(pathflow.values()))[0] - total_load) < 1e-7:
        print("OD Path incidence matrix @ pathflows yield total_load", "\u2713")
    else:
        print("ERROR")

    # %%
