# %%
import networkx as nx
import numpy as np
import time
from scipy.optimize import linprog
from scipy.linalg import null_space
import matplotlib.pyplot as plt

import cvxpy as cp

from src import Plotting as pl


def social_cost(G, F):
    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    return np.sum(alpha_arr @ F**2 + beta_arr @ F)


def potential_energy(G, F):
    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    return np.sum(1 / 2 * alpha_arr @ F**2 + beta_arr @ F)


def social_optimum(G, P, positive_constraint=True, **kwargs):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    maxx = np.array(list(nx.get_edge_attributes(G, "xmax").values()))

    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    # Define the variable F
    if positive_constraint:
        fe = cp.Variable(num_edges, nonneg=True)
    else:
        fe = cp.Variable(num_edges)

    # Define the objective function
    objective = cp.Minimize(cp.sum(alpha_arr @ fe**2 + beta_arr @ fe))

    # Define the constraints
    if len(maxx) == 0:
        constraints = [E @ fe == P]
    else:
        constraints = [
            E @ fe == P,
            cp.sum(fe, axis=1) <= maxx,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve(
        verbose=False, solver=cp.OSQP, eps_rel=1e-7
    )  # print("Optimal value:", problem.value)

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    print_time = kwargs.get("print_time", False)
    if print_time:
        print("Time:", linprog_time, "s")
        print("Social cost:", social_cost(G, fe.value))
        print("Potential energy:", potential_energy(G, fe.value))
        print("Minimum:", problem.value)
    return fe.value


def user_equilibrium(G, P, positive_constraint=True, **kwargs):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    maxx = np.array(list(nx.get_edge_attributes(G, "xmax").values()))

    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    # Define the variable F
    if positive_constraint:
        fe = cp.Variable(num_edges, nonneg=True)
    else:
        fe = cp.Variable(num_edges)

    # Define the objective function
    objective = cp.Minimize(1 / 2 * alpha_arr @ fe**2 + beta_arr @ fe)
    # objective = cp.Minimize(cp.sum(list(map(lambda f: f(0), func_list))))

    # Define the constraints
    if len(maxx) == 0:
        # maxx = np.array([np.inf for e in G.edges()])
        constraints = [E @ fe == P]
    else:
        constraints = [
            E @ fe == P,
            # fe >= np.zeros((num_edges, num_nodes)),
            cp.sum(fe, axis=1) <= maxx,
        ]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve(verbose=False, solver=cp.OSQP, eps_rel=1e-7)
    # print("Optimal value:", problem.value)

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    F = fe.value

    print_time = kwargs.get("print_time", False)
    if print_time:
        print("Time:", linprog_time, "s")
        print("Social cost:", social_cost(G, F))
        print("Potential energy:", potential_energy(G, F))
        print("Minimum:", problem.value)
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
    # nx.set_edge_attributes(G, tt_func, "tt_function")
    nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
    nx.set_edge_attributes(G, dict(zip(G.edges, beta)), "beta")

    pos = nx.spring_layout(G, seed=seed)
    nx.set_node_attributes(G, pos, "pos")

    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


def linearTAP(G, P, social_optimum=False, **kwargs):
    """
    Calculate the linear TAP solution for a given graph G and OD matrix P.
    """
    start_time = time.time()
    if social_optimum:
        Q = 1 / 2
    else:
        Q = 1
    P = np.array(P)

    num_nodes = G.number_of_nodes()

    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    E = -nx.incidence_matrix(G, oriented=True)

    kappa = 1 / alpha_arr
    nx.set_edge_attributes(G, dict(zip(G.edges, kappa)), "kappa")
    L = nx.laplacian_matrix(G, weight="kappa").toarray()

    gamma = beta_arr / alpha_arr
    nx.set_edge_attributes(G, dict(zip(G.edges, gamma)), "gamma")
    A = nx.adjacency_matrix(G, weight="gamma")

    if G.is_directed():
        Gamma = A - A.T
    else:
        upper_triangle = np.triu(A.toarray())
        Gamma = upper_triangle - upper_triangle.T

    if G.is_directed():
        L = E @ np.diag(kappa) @ E.T
        lamb = np.linalg.pinv(L) @ (1 / Q * P + Gamma @ np.ones(num_nodes))
    else:
        lamb = np.linalg.pinv(L) @ (1 / Q * P + Gamma @ np.ones(num_nodes))

    f_alg = Q * (E.T @ lamb) / alpha_arr - Q * beta_arr / alpha_arr
    # f_alg += beta * (num_nodes - 1) / alpha
    print_time = kwargs.get("print_time", False)
    if print_time:
        print("Time:", time.time() - start_time, "s")
    return f_alg, lamb


def ODmatrix(G):
    num_nodes = G.number_of_nodes()
    A = -np.ones((num_nodes, num_nodes))
    np.fill_diagonal(A, (num_nodes - 1))
    return A


def kappa_matrix(G):
    alpha_d = nx.get_edge_attributes(G, "alpha")
    alpha_arr = np.array(list(alpha_d.values()))

    kappa = 1 / alpha_arr
    return np.diag(kappa)


def gamma_matrix(G):
    alpha_dict = nx.get_edge_attributes(G, "alpha")
    beta_dict = nx.get_edge_attributes(G, "beta")

    Gamma = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for e in G.edges:
        i, j = e
        a = alpha_dict[e]
        b = beta_dict[e]
        gamma = b / a
        Gamma[i, j] += gamma
        # Gamma[j, i] -= gamma
    return Gamma


def gamma_vector(G):

    alpha_dict = nx.get_edge_attributes(G, "alpha")
    beta_dict = nx.get_edge_attributes(G, "beta")

    gamma = np.zeros(G.number_of_edges())
    for k, e in enumerate(G.edges):
        a = alpha_dict[e]
        b = beta_dict[e]
        gamma[k] += b / a
    return gamma


# %%
if __name__ == "__main__":
    num_nodes = 5
    num_edges = int(num_nodes * 1.2)
    alpha = 1  # "random"
    beta = 1  # "random"

    # Example graph creation
    G = random_graph(
        seed=42,
        num_nodes=num_nodes,
        num_edges=num_edges,
        alpha=alpha,
        beta=beta,
    )
    G = G.to_undirected()
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    P = np.zeros(G.number_of_nodes())
    load = 100
    P[0], P[1:] = load, -load / (num_nodes - 1)

    F = user_equilibrium(G, P, positive_constraint=False)
    nx.set_edge_attributes(G, dict(zip(G.edges, F)), "flow")

    print(np.isclose(E @ F, P))

    f, lamb = linearTAP(G, P)
    # f += beta * (num_nodes - 1) / alpha

    print(np.abs(f - F) < 1e-5)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cbar = False

    # nc = dict(zip(G.nodes, nc))
    for i, ax in enumerate(axs):
        if i == 0:
            pl.graphPlotCC(G, cc=f, edge_labels=dict(zip(G.edges, f)), ax=ax, cbar=cbar)
        elif i == 1:
            pl.graphPlotCC(G, cc=F, edge_labels=dict(zip(G.edges, F)), ax=ax, cbar=cbar)
            cbar = True

        # fpos = tap.user_equilibrium(G, A, positive_constraint=True)

    # %%

    P = np.zeros(G.number_of_nodes())
    P[0], P[-1] = 100, -100

    f, lamb = linearTAP(G, P)
    fue = user_equilibrium(G, P, positive_constraint=False)
    # pl.graphPlotCC(g, cc=f)
    np.isclose(E @ f, P)
    print(np.abs(f - fue) < 1e-5)

    # %%

# %%

# %%
