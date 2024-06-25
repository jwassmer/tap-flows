# %%
from src import ConvexOptimization as co
from src import GraphGenerator as gg
from src import Plotting as pl
from src import Equilibirium as eq

import networkx as nx
import numpy as np
import cvxpy as cp
import time
from scipy.optimize import linprog


def source_sink_vector(G):
    sources = G.source_nodes
    targets = G.target_nodes
    total_load = G.total_load

    P = np.zeros(G.number_of_nodes())
    P[sources] = total_load / len(sources)
    P[targets] = -total_load / len(targets)

    return P


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

    # U = nx.relabel_nodes(U, lambda x: chr(x + 97))
    U.source_nodes = source_nodes
    U.target_nodes = target_nodes
    U.total_load = total_load

    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, U.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(U.number_of_edges())

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(U.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(U.number_of_edges())

    tt_func = {
        edge: (lambda alpha, beta: lambda n: alpha * n + beta)(alpha[e], beta[e])
        for e, edge in enumerate(U.edges)
    }
    nx.set_edge_attributes(U, tt_func, "tt_function")

    P = source_sink_vector(U)
    P = dict(zip(U.nodes, P))
    nx.set_node_attributes(U, P, "P")

    pos = nx.spring_layout(U, seed=seed)
    nx.set_node_attributes(U, pos, "pos")

    nx.set_edge_attributes(U, "black", "color")
    nx.set_node_attributes(U, "lightgrey", "color")

    for s in source_nodes:
        U.nodes[s]["color"] = "lightblue"
    for t in target_nodes:
        U.nodes[t]["color"] = "red"

    return U


def to_full_directed(G):
    D = nx.DiGraph()
    for n, d in G.nodes(data=True):
        D.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        D.add_edge(u, v, **d)
        D.add_edge(v, u, **d)

    D.total_load = G.total_load
    D.source_nodes = G.source_nodes
    D.target_nodes = G.target_nodes
    return D


def convex_optimization_edge_betweenness_centrality(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    # Create the origin-destination matrix
    A_od = -np.ones((num_nodes, num_nodes))
    np.fill_diagonal(A_od, num_nodes - 1)

    tt_func = nx.get_edge_attributes(G, "tt_function")

    beta_e = np.array([tt_func[e](0) for e in G.edges()])

    # Define the variable F
    F = cp.Variable((num_edges, num_nodes))

    # Define the objective function
    objective = cp.Minimize(cp.sum(beta_e @ F))

    # Define the constraints
    constraints = [E @ F == A_od, F >= np.zeros((num_edges, num_nodes))]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve()

    ebc_linprog = np.sum(F.value, axis=1)
    linprog_time = time.time() - start_time
    print("Time:", linprog_time, "s")
    return ebc_linprog


def linear_program_edge_betweenness_centrality(G):
    start_time = time.time()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    E = -nx.incidence_matrix(G, oriented=True)  # .toarray()

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    betas = np.array([tt_funcs[e](0) for e in G.edges()])
    bounds = np.array([(0, None) for _ in range(num_edges)])

    linprog_ebc = np.zeros(num_edges)
    for source in G.nodes():
        targets = np.delete(np.array(list(G.nodes)), source)

        P = np.zeros(num_nodes)
        P[source] = total_load
        P[targets] = -total_load / (num_nodes - 1)

        # kcl_flow = co.convex_optimization_kcl_tap(G, solver=cp.SCS)
        result = linprog(betas, A_eq=E, b_eq=P, bounds=bounds, method="highs")
        kcl_flow = result.x
        # kcl_flow = co.linear_program(G)
        linprog_ebc += kcl_flow
    linprog_time = time.time() - start_time

    print("Time:", linprog_time, "s")
    return linprog_ebc


# %%
num_nodes = 20
num_edges = int(num_nodes * 1.5)
nodes = np.arange(num_nodes)
source = [0]
targets = np.delete(nodes, source[0])
total_load = len(targets)
# Example graph creation
U = random_graph(
    source,
    targets,
    total_load,
    seed=42,
    num_nodes=num_nodes,
    num_edges=num_edges,
    alpha=0,
    beta="random",
)
G = to_full_directed(U)


# pl.graphPlotCC(G, cc=kcl_flow)
# print([round(f, 2) for f in kcl_flow])


# %%
tt_func = nx.get_edge_attributes(G, "tt_function")
beta = np.array([tt_func[e](0) for e in G.edges()])
nx.set_edge_attributes(G, {n: beta[i] for i, n in enumerate(G.edges)}, "beta")

start_time = time.time()
ebc = nx.edge_betweenness_centrality(G, normalized=False, weight="beta")
sp_time = time.time() - start_time
print(sp_time)
# %%

linprog_ebc = linear_program_edge_betweenness_centrality(G)

delta = np.abs(list(ebc.values()) - linprog_ebc)
print(max(delta))
nc = ["lightgrey" for _ in range(G.number_of_nodes())]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
pl.graphPlotCC(G, cc=linprog_ebc, nc=nc, ax=ax)

fig.savefig("figs/ebc.png", dpi=300)

# %%


ebc_linprog = convex_optimization_edge_betweenness_centrality(G)

ebc = nx.edge_betweenness_centrality(G, normalized=False, weight="beta")

delta = np.abs(list(ebc.values()) - ebc_linprog)
print(max(delta))
# %%
num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
A_od = -np.ones((num_nodes, num_nodes))
np.fill_diagonal(A_od, len(nodes) - 1)

E = -nx.incidence_matrix(G, oriented=True).toarray()
start_time = time.time()
F = np.linalg.pinv(E) @ A_od
print(time.time() - start_time)
F
# %%
beta_e = np.array([tt_func[e](0) for e in G.edges()])

Einv = np.linalg.pinv(E)
# %%
P = np.array(list(nx.get_node_attributes(G, "P").values()))
f = Einv @ P
f

# %%
weights = dict(zip(U.edges, np.ones(U.number_of_edges())))
nx.set_edge_attributes(U, weights, "weight")

F = eq.linear_flow(U)
nx.set_edge_attributes(U, F, "flow")
G = gg.to_directed_flow_graph(U)

# %%
weights = dict(zip(G.edges, np.ones(G.number_of_edges())))
nx.set_edge_attributes(G, weights, "weight")
linflow = eq.linear_flow(G)
# %%
np.array(list(linflow.values())).sum()

# %%
E = -nx.incidence_matrix(G, oriented=True, weight="weight").toarray()
P = np.array(list(nx.get_node_attributes(G, "P").values()))
f = np.linalg.pinv(E) @ P


f - list(linflow.values())
# %%
