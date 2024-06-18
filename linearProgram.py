# %%
import networkx as nx
import numpy as np
import time
from scipy.optimize import linprog
from scipy.linalg import null_space

import cvxpy as cp

from src import Equilibirium as eq
from src import ConvexOptimization as co
from src import Plotting as pl


def social_cost(G, kwd="flow"):
    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    return sum([G.edges[e][kwd] * tt_funcs[e](G.edges[e][kwd]) for e in G.edges()])


def convex_optimization(G, A_od):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    tt_func = nx.get_edge_attributes(G, "tt_function")
    maxx = np.array(list(nx.get_edge_attributes(G, "xmax").values()))

    beta_e = np.array([tt_func[e](0) for e in G.edges()])
    alpha_e = np.array([tt_func[e](1) - tt_func[e](0) for e in G.edges()])

    # Define the variable F
    fe = cp.Variable((num_edges, num_nodes))

    # Define the objective function
    objective = cp.Minimize(cp.sum(1 / 2 * alpha_e @ fe**2 + beta_e @ fe))
    # objective = cp.Minimize(cp.sum(cp.multiply(cp.sum(fe, axis=1), alpha_e) + beta_e))

    # Define the constraints
    if len(maxx) == 0:
        # maxx = np.array([np.inf for e in G.edges()])
        constraints = [E @ fe == A_od]  # , fe >= np.zeros((num_edges, num_nodes))]
    else:
        constraints = [
            E @ fe == A_od,
            fe >= np.zeros((num_edges, num_nodes)),
            cp.sum(fe, axis=1) <= maxx,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve()

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    print("Time:", linprog_time, "s")
    # return fe.value
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
num_nodes = 20
num_edges = int(num_nodes * 1.5)
nodes = np.arange(num_nodes)
source = [0]
targets = np.delete(nodes, source[0])
total_load = len(targets)
# Example graph creation
G = random_graph(
    seed=42,
    num_nodes=num_nodes,
    num_edges=num_edges,
    alpha=3,
    beta=3,
)

# A = ODmatrix(G)
A = np.zeros((num_nodes, num_nodes))
# A[:, 0] = -np.ones(num_nodes)
A[0, 0] = 100
A[-1, 0] = -100

# xmax = np.array([10 for e in G.edges()])
# nx.set_edge_attributes(G, dict(zip(G.edges, xmax)), "xmax")
F = convex_optimization(G, A)
nx.set_edge_attributes(G, dict(zip(G.edges, F)), "flow")

pl.graphPlotCC(G, cc=np.abs(F))
f = dict(zip(G.edges, F))
f

# %%
tt_f = nx.get_edge_attributes(G, "tt_function")
alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
beta = np.array([tt_f[e](0) for e in G.edges()])
P = A[:, 0]

E = -nx.incidence_matrix(G, oriented=True)

kappa = 1 / alpha
nx.set_edge_attributes(G, dict(zip(G.edges, kappa)), "kappa")

L = nx.laplacian_matrix(G, weight="kappa").toarray()

lamb = np.linalg.pinv(L + L.T) @ P


f_alg = (E.T @ lamb - num_nodes * beta) / alpha
f_alg - F
# %%

E @ (f_alg - min(f_alg))
# %%

# %%
