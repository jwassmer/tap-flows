# %%
import numpy as np
import cvxpy as cp
import networkx as nx
import itertools

from src import GraphGenerator as gg
from src import LinAlg as la
from src import Plotting as pl
from src import Equilibirium as eq
import time


def dict_add(d1, d2):
    if d1.keys() != d2.keys():
        raise ValueError("Keys of the dictionaries are not the same")
    return {k: d1[k] + d2[k] for k in d1.keys()}


def dict_subtract(d1, d2):
    if d1.keys() != d2.keys():
        raise ValueError("Keys of the dictionaries are not the same")
    return {k: d1[k] - d2[k] for k in d1.keys()}


def convex_optimization_linflow(G):
    # Get the number of edges and nodes
    num_edges = G.number_of_edges()

    # Create the edge incidence matrix E
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    # Define the weight vector K_e (from adjacency matrix weights)
    K = np.array([G[u][v]["weight"] for u, v in G.edges()])

    P = list(nx.get_node_attributes(G, "P").values())

    # Define the flow variable f_e
    f = cp.Variable(num_edges)

    # Objective function: minimize sum of (f_e^2 / (2 * K_e))
    objective = cp.Minimize(cp.sum(cp.multiply(f**2, 1 / (2 * K))))

    # Constraints: E @ f = p
    constraints = [E @ f == P, f >= np.zeros(num_edges)]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Get the results
    flow_values = f.value

    flow_values, problem.value
    return flow_values


def convex_optimization_TAP(G):
    total_load = G.total_load
    sources = G.source_nodes
    targets = G.target_nodes

    od_pairs = list(itertools.product(sources, targets))
    all_paths = la.all_od_paths(G)
    num_paths = len(all_paths)

    ddict = dict(zip(od_pairs, np.zeros(len(od_pairs))))
    for path in all_paths:
        for k, v in ddict.items():
            if (path[0], path[-1]) == k:
                ddict[k] = 1.0
                break
    d = np.array(list(ddict.values()))
    d *= total_load / sum(d)

    num_edges = nx.number_of_edges(G)

    delta = la.path_link_incidence_matrix(G)
    gamma = la.od_path_incidence_matrix(G)

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    betas = np.array([tt_funcs[e](0) for e in G.edges()])
    alphas = np.array([tt_funcs[e](1) - tt_funcs[e](0) for e in G.edges()])

    f = cp.Variable(num_edges)
    h = cp.Variable(num_paths)

    objective = cp.Minimize(cp.sum(alphas @ f**2 / 2 + betas @ f))
    constraints = [
        delta @ h == f,
        gamma @ h == d,
        f >= np.zeros(num_edges),
    ]
    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Get the results
    flow_values = f.value

    flow_values, problem.value
    return flow_values


# %%

if __name__ == "__main__":
    source, target = ["a", "b", "e", "k", "g"], ["j", "c", "d", "f"]
    total_load = 1000
    # Example graph creation
    U = gg.random_graph(source, target, total_load, seed=42, num_nodes=20, num_edges=30)
    G = gg.to_directed_flow_graph(U)

    # test_cv(G)
    pl.graphPlotCC(G, cc="flow")

    # %%

    # big graph timings
    source, target = ["a"], ["b"]
    U = gg.random_graph(
        source, target, total_load, seed=42, num_nodes=200, num_edges=300
    )
    G = gg.to_directed_flow_graph(U)
    print(G)

    start_time = time.time()
    eq.linear_flow(G)
    la_lin_flow_time = time.time() - start_time

    print("L^-1: ", la_lin_flow_time, "s")

    start_time = time.time()
    cv_lin_flow = convex_optimization_linflow(G)
    cv_lin_flow_time = time.time() - start_time
    print("CVO of KCL: ", cv_lin_flow_time, "s")

    start_time = time.time()
    cv_tap_flow = convex_optimization_TAP(G)
    cv_tap_flow_time = time.time() - start_time
    print("CVO of TAP: ", cv_tap_flow_time, "s")

    print(
        "CVO of KCL is factor {} faster than CVO of TAP".format(
            cv_tap_flow_time / cv_lin_flow_time
        )
    )

    print(
        "L^-1 is factor {} faster than CVO of KCL".format(
            cv_lin_flow_time / la_lin_flow_time
        )
    )
    # np.abs(cv_tap_flow - cv_lin_flow) < 1e-5
    # %%
    pathflows = list(la.path_flows(G).values())
    delta = la.path_link_incidence_matrix(G)
    gamma = la.od_path_incidence_matrix(G)

    # %%

    D = gamma @ pathflows
    od_pairs = list(itertools.product(G.source_nodes, G.target_nodes))

    dict(zip(od_pairs, D))

    # %%
