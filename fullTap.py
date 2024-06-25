# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx

from src import ConvexOptimization as co
from src import Plotting as pl
from src import Equilibirium as eq


def social_cost(G, kwd="flow"):
    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    return sum([G.edges[e][kwd] * tt_funcs[e](G.edges[e][kwd]) for e in G.edges()])


def convex_optimization_fulltap(G, A_od):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    tt_func = nx.get_edge_attributes(G, "tt_function")
    maxx = np.array(list(nx.get_edge_attributes(G, "xmax").values()))

    beta_e = np.array([tt_func[e](0) for e in G.edges()])
    alpha_e = np.array([tt_func[e](1) - tt_func[e](0) for e in G.edges()])

    # Define the variable F
    fe = cp.Variable((num_nodes, num_edges))

    # Define the objective function
    objective = cp.Minimize(
        cp.sum(1 / 2 * fe**2 @ alpha_e + fe @ beta_e)
    )  # + cp.sum(fe))
    # objective = cp.Minimize(cp.sum(cp.multiply(cp.sum(fe, axis=1), alpha_e) + beta_e))

    # Define the constraints
    if len(maxx) == 0:
        # print(E.toarray())
        constraints = [fe >= np.zeros((num_nodes, num_edges))]
        for n in G.nodes():
            # print(n, A_od[:, n])
            constraints.append(E @ fe[n, :] == A_od[:, n])
    else:
        constraints = [
            E @ fe.T == A_od,
            fe >= np.zeros((num_nodes, num_edges)),
            cp.sum(fe, axis=0) <= maxx,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve()

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    print("Time:", linprog_time, "s")
    return np.round(fe.value.T, 2)
    return np.sum(fe.value, axis=1)


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

    G = U.to_directed()

    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(G.number_of_edges())

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())

    tt_func = {
        edge: (lambda alpha, beta: lambda n: alpha * n + beta)(alpha[e], beta[e])
        for e, edge in enumerate(G.edges)
    }
    nx.set_edge_attributes(G, tt_func, "tt_function")

    pos = nx.spring_layout(G, seed=seed)
    nx.set_node_attributes(G, pos, "pos")

    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


def ODmatrix(G):
    num_nodes = G.number_of_nodes()
    A = -np.ones((num_nodes, num_nodes))
    np.fill_diagonal(A, (num_nodes - 1))
    return A


# %%
num_nodes = 3
num_edges = 2  # int(num_nodes * 1.5)
nodes = np.arange(num_nodes)
source = [0]
targets = np.delete(nodes, source[0])
total_load = 1000
# Example graph creation
G = random_graph(
    seed=42,
    num_nodes=num_nodes,
    num_edges=num_edges,
    alpha=1,
    beta=3,
)
rem_edges = [(1, 0), (0, 2)]
G.remove_edges_from(rem_edges)
G.add_edge(2, 1, tt_function=lambda x: 1 * x + 6)
E = -nx.incidence_matrix(G, oriented=True)
# Einv = np.linalg.pinv(E.toarray())
A = ODmatrix(G)


for n in [1, 0]:
    A = np.zeros((num_nodes, num_nodes))
    A[2, 0] += total_load
    A[1, 0] += -total_load
    A[2, n] += total_load
    A[0, n] += -total_load
    print(A)

    F_nm = convex_optimization_fulltap(G, A)
    F = F_nm.sum(axis=1)
    nx.set_edge_attributes(G, dict(zip(G.edges, F)), "flow")

    pl.graphPlotCC(G, cc=F, edge_labels=dict(zip(G.edges, F)))
    # print("F_nm=", F_nm)
    E @ F_nm

# %%

# G.add_edge(1, 2, tt_function=lambda x: 1 * x + 6)
G.source_nodes = [2]
G.target_nodes = [0, 1]
G.total_load = 2000

f = co.convex_optimization_TAP(G)
pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)))
# %%
