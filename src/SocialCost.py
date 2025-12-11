# %%

from src import Graphs as gr
from src import Plotting as pl

import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression


def braessGraph():
    G = nx.DiGraph()

    a, b, c, d = 0, 1, 2, 3

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0.5)}),
            (b, {"pos": (0.5, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 0.5)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"alpha": 1 / 100, "beta": 10}),
            (b, d, {"alpha": 1 / 1000, "beta": 25}),
            (a, c, {"alpha": 1 / 1000, "beta": 25}),
            (c, d, {"alpha": 1 / 100, "beta": 10}),
            (b, c, {"alpha": 1 / 1000, "beta": 1}),
        ]
    )

    # G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    return G


def _social_cost_from_vecs(G, alpha, beta, P):
    E = -nx.incidence_matrix(G, oriented=True)
    num_nodes = E.shape[0]

    L = E @ np.diag(1 / alpha) @ E.T
    nx.set_edge_attributes(G, dict(zip(G.edges, beta / alpha)), "gamma")
    A = nx.adjacency_matrix(G, weight="gamma")
    Gamma = A - A.T

    lamb_ue = np.linalg.pinv(L) @ (P + Gamma @ np.ones(num_nodes))

    delta_lamb = E.T @ lamb_ue

    sc = (delta_lamb**2 - delta_lamb * beta) / alpha  # / load

    return np.sum(sc)


def total_social_cost(G, f, **kwargs):
    alpha = kwargs.pop("alpha", None)
    beta = kwargs.pop("beta", None)
    if alpha is None:
        alpha_d = nx.get_edge_attributes(G, "alpha")
        alpha = np.array(list(alpha_d.values()))
    if beta is None:
        beta_d = nx.get_edge_attributes(G, "beta")
        beta = np.array(list(beta_d.values()))
    sc_vec = social_cost_vec(G, f, alpha=alpha, beta=beta)
    return np.sum(sc_vec)


def social_cost_vec(G, f, alpha=None, beta=None):
    if alpha is None:
        alpha_d = nx.get_edge_attributes(G, "alpha")
        alpha = np.array(list(alpha_d.values()))
    if beta is None:
        beta_d = nx.get_edge_attributes(G, "beta")
        beta = np.array(list(beta_d.values()))
    f = np.array(f)
    return alpha * f**2 + beta * f


def all_braess_edges(G, P):
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.array(P)

    alpha_d = nx.get_edge_attributes(G, "alpha")
    alpha_arr = np.array(list(alpha_d.values()))

    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)

    is_braessian = {}

    for edge in G.edges:
        slope = derivative_socia_cost_ab(G, Linv, P, edge, alpha_arr)
        is_braessian[edge] = slope

    return is_braessian


def slope_social_cost(G, P, edge):
    # a, b = edge
    # edge_idx = list(G.edges).index(edge)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.array(P)

    alpha_d = nx.get_edge_attributes(G, "alpha")
    alpha_arr = np.array(list(alpha_d.values()))

    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)

    return derivative_socia_cost_ab(G, Linv, P, edge, alpha_arr)


def all_derivatives_slope_social_cost(G, P):
    # a, b = edge
    # edge_idx = list(G.edges).index(edge)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.array(P)

    alpha_d = nx.get_edge_attributes(G, "alpha")
    alpha_arr = np.array(list(alpha_d.values()))

    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)

    dsc = (E.T @ Linv @ P) / alpha_arr

    dsc_dict = dict(zip(G.edges, dsc))

    return dsc_dict


def derivative_socia_cost_ab(G, Linv, P, edge, alpha_arr):
    a, b = edge[0], edge[1]

    a_idx = list(G.nodes).index(a)
    b_idx = list(G.nodes).index(b)

    edge_idx = list(G.edges()).index(edge)

    P = np.array(P)

    slope = (Linv[a_idx, :] - Linv[b_idx, :]) @ P / alpha_arr[edge_idx]
    return slope


def linreg_slope_sc(G, P, edge):
    alpha_d = nx.get_edge_attributes(G, "alpha")
    beta_d = nx.get_edge_attributes(G, "beta")
    alpha_arr = np.array(list(alpha_d.values()))
    beta_arr = np.array(list(beta_d.values()))

    beta_e = np.linspace(-1e1, 1e1, 5)

    edge_idx = list(G.edges).index(edge)

    sc_beta = []
    for i, bet in enumerate(beta_e):
        beta_arr[edge_idx] = bet
        s = _social_cost_from_vecs(G, alpha_arr, beta_arr, P)
        sc_beta.append(s)

    # Reshape the data for linear regression
    X = np.array(beta_e).reshape(-1, 1)
    y = np.array(sc_beta)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    # y_intercept = model.intercept_
    m = model.coef_[0]
    return m


def all_social_cost_derivatives(G, P, alpha_arr=None):
    if alpha_arr is None:
        alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))

    E = -nx.incidence_matrix(G, oriented=True)
    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)
    slopes = {}

    for e in G.edges:
        s = derivative_socia_cost_ab(G, Linv, P, e, alpha_arr)
        slopes[e] = s
    return slopes


# %%

if __name__ == "__main__":
    from src import TAPOptimization as tap
    from src import Plotting as pl

    import matplotlib.pyplot as plt

    G = gr.random_planar_graph(10, seed=1)

    # G = braessGraph()

    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.zeros(G.number_of_nodes())
    load = 1
    source = 0
    P[source] = load
    targets = [-1]
    P[targets] = -load / len(targets)

    f_ue, lamb_ue = tap.user_equilibrium(
        G, P, positive_constraint=True, return_lagrange_multiplier=True
    )
    f_, lamb_ = tap.linearTAP(G, P)
    # print(tap.social_cost(G, f_) / load)

    slopes = all_derivatives_slope_social_cost(G, P)
    print(slopes)

    # %%

    # edge = min(slopes, key=slopes.get)

    beta_e = np.linspace(-1e1, 1e1, 5)

    for edge, slope in slopes.items():

        m = linreg_slope_sc(G, P, edge)

        print(np.isclose(m, slope))

    # %%

    E = -nx.incidence_matrix(G, oriented=True).toarray()
    alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    L = E @ np.diag(1 / alpha) @ E.T

    E @ (beta / alpha)

    Linv = np.linalg.pinv(L)

    # %%
    dscdbeta = np.round(E.T @ Linv @ P, 3)

    # %%


# %%
