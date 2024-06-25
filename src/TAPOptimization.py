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


def social_cost(G, F):
    tt_f = nx.get_edge_attributes(G, "tt_function")
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    beta = np.array([tt_f[e](0) for e in G.edges()])

    return np.sum(alpha @ F**2 + beta @ F)


def potential_energy(G, F):
    tt_f = nx.get_edge_attributes(G, "tt_function")
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    beta = np.array([tt_f[e](0) for e in G.edges()])

    return np.sum(1 / 2 * alpha @ F**2 + beta @ F)


def social_optimum(G, A_od):
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
    objective = cp.Minimize(cp.sum(alpha_e @ fe**2 + beta_e @ fe))

    # Define the constraints
    if len(maxx) == 0:
        constraints = [E @ fe == A_od]
    else:
        constraints = [
            E @ fe == A_od,
            cp.sum(fe, axis=1) <= maxx,
        ]

    constraints.append(fe >= np.zeros((num_edges, num_nodes)))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve(verbose=False)
    # print("Optimal value:", problem.value)

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    print("Time:", linprog_time, "s")
    F = np.sum(fe.value, axis=1)
    return F


def user_equilibrium(G, A_od, positive_constraint=True):
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
    # objective = cp.Minimize(cp.sum(list(map(lambda f: f(0), func_list))))

    # Define the constraints
    if len(maxx) == 0:
        # maxx = np.array([np.inf for e in G.edges()])
        constraints = [E @ fe == A_od]
    else:
        constraints = [
            E @ fe == A_od,
            # fe >= np.zeros((num_edges, num_nodes)),
            cp.sum(fe, axis=1) <= maxx,
        ]

    if positive_constraint:
        constraints.append(fe >= np.zeros((num_edges, num_nodes)))

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve(verbose=False)
    # print("Optimal value:", problem.value)

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    print("Time:", linprog_time, "s")
    F = np.sum(fe.value, axis=1)
    print("Social cost:", social_cost(G, F))
    return F


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

    if isinstance(alpha, str) and alpha == "random_symmetric":
        # doesnt work atm
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, U.number_of_edges())
    if isinstance(beta, str) and beta == "random_symmetric":
        # doesnt work atm
        np.random.seed(seed)
        beta = 100 * np.random.rand(U.number_of_edges())

    G = U.to_directed()

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(G.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(G.number_of_edges())
    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, G.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(G.number_of_edges())

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


def linearTAP(G, P):
    """
    Calculate the linear TAP solution for a given graph G and OD matrix P.
    """
    start_time = time.time()
    num_nodes = G.number_of_nodes()
    tt_f = nx.get_edge_attributes(G, "tt_function")
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    beta = np.array([tt_f[e](0) for e in G.edges()])

    E = -nx.incidence_matrix(G, oriented=True)

    kappa = 1 / alpha
    nx.set_edge_attributes(G, dict(zip(G.edges, kappa)), "kappa")
    L = nx.laplacian_matrix(G, weight="kappa").toarray()

    alpha_dict = dict(zip(G.edges, alpha))
    beta_dict = dict(zip(G.edges, beta))
    Gamma = np.zeros((num_nodes, num_nodes))
    if G.is_directed():
        superL = L + L.T
        np.fill_diagonal(superL, 0)
        np.fill_diagonal(superL, -np.sum(superL, axis=0))
        for e in G.edges:
            i, j = e
            a = alpha_dict[e]
            b = beta_dict[e]
            gamma = b / a
            Gamma[i, j] += gamma
            Gamma[j, i] -= gamma
        lamb = np.linalg.pinv(superL) @ (P + Gamma @ (num_nodes * np.ones(num_nodes)))

    else:
        for e in G.edges:
            i, j = e
            a = alpha_dict[e]
            b = beta_dict[e]
            gamma = b / a
            Gamma[i, j] += gamma
            Gamma[j, i] -= gamma
        lamb = np.linalg.pinv(L) @ (P + Gamma @ (num_nodes * np.ones(num_nodes)))

    f_alg = ((E.T @ lamb) - (num_nodes * beta)) / alpha
    print("Time:", time.time() - start_time, "s")
    return f_alg, lamb


def ODmatrix(G):
    num_nodes = G.number_of_nodes()
    A = -np.ones((num_nodes, num_nodes))
    np.fill_diagonal(A, (num_nodes - 1))
    return A


# %%
if __name__ == "__main__":
    num_nodes = 20
    num_edges = int(num_nodes * 1.2)

    # Example graph creation
    G = random_graph(
        seed=42,
        num_nodes=num_nodes,
        num_edges=num_edges,
        alpha="random",
        beta="random",
    )
    # edges = list(G.edges())
    # edges_to_remove = np.random.choice(range(len(edges)), 20, replace=False)
    # G.remove_edges_from([edges[i] for i in edges_to_remove])

    A = ODmatrix(G)
    B = np.zeros((num_nodes, num_nodes))
    B[:, 0] = A[:, 0]

    # xmax = np.array([10 for e in G.edges()])
    # nx.set_edge_attributes(G, dict(zip(G.edges, xmax)), "xmax")
    F = user_equilibrium(G, B, positive_constraint=False)
    nx.set_edge_attributes(G, dict(zip(G.edges, F)), "flow")
    pl.graphPlotCC(G, cc=F)

    posF = user_equilibrium(G, B, positive_constraint=True)
    nx.set_edge_attributes(G, dict(zip(G.edges, posF)), "flow")
    pl.graphPlotCC(G, cc=posF, norm="LogNorm")
    # f = dict(zip(G.edges, F))
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    print(np.isclose(E @ F, B[:, 0]))

    f = linearTAP(G, B[:, 0])
    print(np.abs(f - F) < 1e-5)

    # %%
    g = G  # .to_undirected()

    P = np.zeros(g.number_of_nodes())
    P[16], P[15] = 100, -100
    E = -nx.incidence_matrix(g, oriented=True)

    f = linearTAP(g, P)
    pl.graphPlotCC(g, cc=f)
    E @ f
    np.isclose(E @ f, P)

    # %%
