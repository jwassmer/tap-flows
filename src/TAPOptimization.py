# %%
import networkx as nx
import numpy as np
import time
from scipy.optimize import linprog
from scipy.linalg import null_space
import matplotlib.pyplot as plt

import cvxpy as cp

from src import Plotting as pl
from src._graph_utils import (
    edge_attribute_array,
    random_directed_cost_graph,
    validate_node_balance,
)


def social_cost(G, F):
    alpha_arr = edge_attribute_array(G, "alpha")
    beta_arr = edge_attribute_array(G, "beta")

    flow = np.asarray(F, dtype=float).reshape(-1)
    return np.sum(alpha_arr @ flow**2 + beta_arr @ flow)


def potential_energy(G, F):
    alpha_arr = edge_attribute_array(G, "alpha")
    beta_arr = edge_attribute_array(G, "beta")
    flow = np.asarray(F, dtype=float).reshape(-1)

    return np.sum(1 / 2 * alpha_arr @ flow**2 + beta_arr @ flow)


def optimize_tap(
    G,
    demands,
    with_capacity=False,
    social_optimum=False,
    positive_constraint=True,
    return_lagrange_multiplier=False,
    **kwargs
):
    """
    Solves the multicommodity flow problem using CVXPY for a given graph,
    demands, and linear cost function parameters alpha and beta.

    Parameters:
        G: nx.DiGraph - the graph
        demands: list - the demands for each commodity
    """
    start_time = time.time()
    A = -nx.incidence_matrix(G, oriented=True)  # .toarray()

    alpha = kwargs.pop("alpha", None)
    beta = kwargs.pop("beta", None)
    alpha = edge_attribute_array(G, "alpha", alpha)
    beta = edge_attribute_array(G, "beta", beta)
    demand_vec = validate_node_balance(G, demands)

    # Number of edges
    num_edges = G.number_of_edges()

    # Variables for the flow on each edge for each commodity
    if positive_constraint:
        flows = cp.Variable(num_edges, nonneg=True)
    elif not positive_constraint:
        flows = cp.Variable(num_edges)

    demand_constraint = A @ flows == demand_vec
    constraints = [demand_constraint]

    if with_capacity:
        capacity = edge_attribute_array(G, "capacity")
        if len(capacity) > 0:
            constraints.append(flows <= capacity)

    if social_optimum:
        Q = 1
    elif not social_optimum:
        Q = 1 / 2

    # Objective function
    objective = cp.Minimize(
        cp.sum(cp.multiply(Q * alpha, flows**2) + cp.multiply(beta, flows))
    )

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    # Extracting specific kwargs if provided, otherwise setting default values
    solver = kwargs.pop("solver", cp.OSQP)
    eps_rel = kwargs.pop("eps_rel", 1e-6)
    prob.solve(solver=solver, eps_rel=eps_rel, **kwargs)

    # Extract the flows for each commodity
    # flows_value = [f.value for f in flows]
    conv_time = time.time() - start_time

    lagrange_multipliers = demand_constraint.dual_value

    print_time = kwargs.get("print_time", False)
    if print_time:
        print("Time:", conv_time, "s")

    if return_lagrange_multiplier:
        return np.asarray(flows.value).reshape(-1), lagrange_multipliers
    else:
        return np.asarray(flows.value).reshape(-1)


def social_optimum(
    G, P, positive_constraint=True, return_lagrange_multiplier=False, **kwargs
):
    F = optimize_tap(
        G,
        P,
        positive_constraint=positive_constraint,
        social_optimum=True,
        return_lagrange_multiplier=return_lagrange_multiplier,
        **kwargs
    )
    return F


def user_equilibrium(
    G, P, positive_constraint=True, return_lagrange_multiplier=False, **kwargs
):
    F = optimize_tap(
        G,
        P,
        positive_constraint=positive_constraint,
        social_optimum=False,
        return_lagrange_multiplier=return_lagrange_multiplier,
        **kwargs
    )
    return F


def random_graph(
    num_nodes=10,
    num_edges=15,
    seed=42,
    alpha=1,
    beta=0,
):
    return random_directed_cost_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        alpha=alpha,
        beta=beta,
    )


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

    alpha_arr = kwargs.pop("alpha", None)
    beta_arr = kwargs.pop("beta", None)
    alpha_arr = edge_attribute_array(G, "alpha", alpha_arr)
    beta_arr = edge_attribute_array(G, "beta", beta_arr)
    P = validate_node_balance(G, P)

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


def od_matrix(G):
    """Preferred snake_case alias for :func:`ODmatrix`."""
    return ODmatrix(G)


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


def linear_tap(G, P, social_optimum=False, **kwargs):
    """Preferred snake_case alias for :func:`linearTAP`."""
    return linearTAP(G, P, social_optimum=social_optimum, **kwargs)


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
