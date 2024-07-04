# %%

import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression


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


def total_social_cost(G, f):
    sc_vec = social_cost_vec(G, f)
    return np.sum(sc_vec)


def social_cost_vec(G, f):
    f = np.array(f)
    tt_f = nx.get_edge_attributes(G, "tt_function")
    beta = np.array([tt_f[e](0) for e in G.edges()])
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    return alpha * f**2 + beta * f


def social_cost_vec(G, f):
    f = np.array(f)
    tt_f = nx.get_edge_attributes(G, "tt_function")
    beta = np.array([tt_f[e](0) for e in G.edges()])
    alpha = np.array([tt_f[e](1) - tt_f[e](0) for e in G.edges()])
    return alpha * f**2 + beta * f


def slope_social_cost(G, P, edge):
    # a, b = edge
    # edge_idx = list(G.edges).index(edge)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.array(P)

    alpha_arr = np.array(
        [G.edges[e]["tt_function"](1) - G.edges[e]["tt_function"](0) for e in G.edges()]
    )

    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)

    return slope_socia_cost_ab(G, Linv, P, edge, alpha_arr)


def slope_socia_cost_ab(G, Linv, P, edge, alpha_arr):
    a, b = edge
    edge_idx = list(G.edges).index(edge)
    P = np.array(P)

    slope = (Linv[a, :] - Linv[b, :]) @ P / alpha_arr[edge_idx]
    return slope


def linreg_slope_sc(G, P, edge):
    tt_fs = nx.get_edge_attributes(G, "tt_function")
    alpha_arr = np.array([tt_fs[e](1) - tt_fs[e](0) for e in G.edges()])
    beta_arr = np.array([tt_fs[e](0) for e in G.edges()])

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


def all_social_cost_slopes(G, P, alpha_arr):
    E = -nx.incidence_matrix(G, oriented=True)
    L = E @ np.diag(1 / alpha_arr) @ E.T
    Linv = np.linalg.pinv(L)
    slopes = {}
    for e in G.edges():
        s = slope_socia_cost_ab(G, Linv, P, e, alpha_arr)
        slopes[e] = s
    return slopes


# %%

if __name__ == "__main__":
    from src import TAPOptimization as tap
    from src import Plotting as pl

    import matplotlib.pyplot as plt

    G = tap.random_graph(
        seed=42,
        num_edges=25,
        num_nodes=20,
        alpha=1,
        beta=10,
    )

    # G = braessGraph()

    E = -nx.incidence_matrix(G, oriented=True).toarray()
    P = np.zeros(G.number_of_nodes())
    load = 500
    source = 15
    P[source] = load
    targets = [16]  # np.delete(np.arange(G.number_of_nodes()), source)
    P[targets] = -load / len(targets)

    f_ue = tap.user_equilibrium(G, P, positive_constraint=True)
    f_, lamb_ue = tap.linearTAP(G, P)
    print(tap.social_cost(G, f_) / load)

    # pl.graphPlotCC(G, cc=f_ue)  # , edge_labels=dict(zip(G.edges, f_ue)))

    slopes = {}
    for e in G.edges():
        s = slope_social_cost(G, P, e)
        slopes[e] = s
    slopes

    # %%

    # edge = min(slopes, key=slopes.get)

    tt_fs = nx.get_edge_attributes(G, "tt_function")
    alpha_arr = np.array([tt_fs[e](1) - tt_fs[e](0) for e in G.edges()])
    beta_arr = np.array([tt_fs[e](0) for e in G.edges()])

    beta_e = np.linspace(-1e1, 1e1, 5)

    for edge, slope in slopes.items():

        m = linreg_slope_sc(G, P, edge)

        print(np.isclose(m, slope))

# %%
