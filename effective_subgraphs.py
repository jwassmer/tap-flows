# %%
from src import Graphs as gr
from src import TAPOptimization as tap
from src import Plotting as pl
from src import SIBC as sibc

import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl


def effective_beta(G, f):
    if isinstance(f, (int, float)):
        f = f * np.ones(len(G.edges))

    if isinstance(f, list):
        f = np.array(f)

    if isinstance(f, np.ndarray):
        f_dict = dict(zip(G.edges, f))

    beta = nx.get_edge_attributes(G, "beta")
    alpha = nx.get_edge_attributes(G, "alpha")

    eff_beta = {}

    for e in G.edges:
        eff_beta[e] = alpha[e] * f_dict[e] + beta[e]

    return eff_beta


def effective_subgraph(G, depth=5, weight="beta"):
    A0 = nx.adjacency_matrix(G, weight=weight).toarray()
    A = A0.copy()
    A[A == 0] = np.inf
    A[np.arange(0, len(G)), np.arange(0, len(G))] = 0

    res_mat = A.copy()
    for _ in range(depth):
        res_mat_new = res_mat.copy()
        for i in range(len(G)):
            for j in range(len(G)):
                res_mat_new[i, j] = np.min(res_mat[i, :] + res_mat[:, j])
        res_mat = res_mat_new

    conenction_matrix_flow = A0.copy()
    for i in range(len(G)):
        for j in range(len(G)):
            if res_mat[0, i] > res_mat[0, j]:
                conenction_matrix_flow[i, j] = 0

    Gsub = nx.from_numpy_array(
        conenction_matrix_flow, create_using=nx.DiGraph, edge_attr=weight
    )

    pos = nx.get_node_attributes(G, "pos")
    nx.set_node_attributes(Gsub, pos, "pos")
    alpha = nx.get_edge_attributes(G, "alpha")
    beta = nx.get_edge_attributes(G, "beta")
    nx.set_edge_attributes(Gsub, alpha, "alpha")
    nx.set_edge_attributes(Gsub, beta, "beta")
    return Gsub


def effective_flow_solver(G, P, f0=None, depth=3):
    n = len(G)
    A_alpha = nx.adjacency_matrix(G, weight="alpha").toarray()
    A_alpha_connectivity = A_alpha.copy()
    A_alpha_connectivity[A_alpha_connectivity == 0] = np.inf
    A_alpha_connectivity[np.arange(0, n), np.arange(0, n)] = 0

    A_beta = nx.adjacency_matrix(G, weight="beta").toarray()
    A_beta_connectivity = A_beta.copy()
    A_beta_connectivity[A_beta_connectivity == 0] = np.inf
    A_beta_connectivity[np.arange(0, n), np.arange(0, n)] = 0

    if f0 is None:
        f0 = np.zeros(n)

    if isinstance(f0, (int, float)):
        f0 = f0

    connectivity_mat = f0 * A_alpha_connectivity + A_beta_connectivity

    res_mat = connectivity_mat.copy()
    for _ in range(depth):
        res_mat_new = res_mat.copy()
        for i in range(n):
            for j in range(n):
                res_mat_new[i, j] = np.min(res_mat[i, :] + res_mat[:, j])
        res_mat = res_mat_new

    con_flow = connectivity_mat.copy()
    beta_flow = A_beta_connectivity.copy()
    alpha_flow = A_alpha_connectivity.copy()
    for i in range(n):
        for j in range(n):
            if res_mat[0, i] > res_mat[0, j]:
                con_flow[i, j] = np.inf
                beta_flow[i, j] = np.inf
                alpha_flow[i, j] = np.inf

    L2 = -np.divide(1, alpha_flow) - np.divide(1, alpha_flow.T)
    L2[np.arange(n), np.arange(n)] = 0
    L2[np.arange(n), np.arange(n)] = -L2.sum(-1)

    gamma = beta_flow / alpha_flow
    gamma[np.isnan(gamma)] = 0
    gamma = gamma - gamma.T

    lambdas = np.linalg.lstsq(L2, gamma.sum(-1) + P)[0]

    f = 1 / alpha_flow * (lambdas[:, None] - lambdas[None, :]) - beta_flow / alpha_flow
    f[np.isnan(f)] = 0

    # subgraph = nx.from_numpy_array(f, create_using=nx.DiGraph, edge_attr="eff_flow")
    # f_dict = nx.get_edge_attributes(subgraph, "eff_flow")

    # for e in G.edges:
    #    if e not in f_dict:
    #        f_dict[e] = 0

    return lambdas[:, None], f


def iterative_solver(G, P):
    Gs = effective_subgraph(G, depth=5, weight="beta")
    # print(Gs.edges[(1, 5)])
    for _ in range(20):
        f0 = tap.user_equilibrium(
            Gs, P, positive_constraint=False, solver=cp.SCS, eps_rel=1e-8
        )
        print(min(f0))
        f0_pos = np.where(f0 < 0, 0, f0)
        eff_beta = effective_beta(Gs, f0_pos)
        nx.set_edge_attributes(Gs, eff_beta, "beta_eff")
        Gs = effective_subgraph(Gs, depth=5, weight="beta_eff")

    return Gs, f0


def potential_condition(G, lamb):
    beta = nx.get_edge_attributes(G, "beta")
    lamb_dict = dict(zip(G.nodes, lamb))
    f_bool = {}
    for e in G.edges:
        n, m = e
        diff = lamb_dict[n] - (lamb_dict[m] + beta[e])
        f_bool[e] = lamb_dict[n] >= lamb_dict[m] + beta[e]

    if all(f_bool.values()):
        return True
    else:
        return False


def compare_flows(f0, f1):
    if isinstance(f0, np.ndarray):
        return TypeError("f0 must be a dictionary")
    if isinstance(f1, np.ndarray):
        return TypeError("f1 must be a dictionary")

    diff = {}
    longer_dict = f0 if len(f0) > len(f1) else f1
    for e in longer_dict:
        if e in f0 and e in f1:
            diff[e] = f0[e] - f1[e]
        elif e in f0:
            diff[e] = f0[e]
        elif e in f1:
            diff[e] = f1[e]

    return diff


def flow_subgraph(G, feff):
    if isinstance(feff, np.ndarray):
        feff = dict(zip(G.edges, feff))

    Gs = nx.DiGraph()
    Gs.add_nodes_from(G.nodes)
    eps = 0

    for e, f in feff.items():
        if f > eps:
            Gs.add_edge(*e, alpha=G.edges[e]["alpha"], beta=G.edges[e]["beta"])
        # if f < -eps:
        #    l = e[::-1]
        #    Gs.add_edge(*l, alpha=G.edges[l]["alpha"], beta=G.edges[l]["beta"])

    pos = nx.get_node_attributes(G, "pos")
    nx.set_node_attributes(Gs, pos, "pos")

    return Gs


# %%

G = gr.triangularLattice(4, beta="random", alpha="random")
P = np.zeros(G.number_of_nodes())
P[0] = 1000
P[1:] = -P[0] / (len(P) - 1)
# P[-1] = -100

# %%
fue = tap.user_equilibrium(G, P, positive_constraint=True)
cmap = plt.cm.cividis
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=max(fue))
pl.graphPlot(G, ec=fue, cmap=cmap, norm=norm)

# %%

T = 1e7
Gs = G.copy()
alpha_vec = np.array(list(nx.get_edge_attributes(Gs, "alpha").values()))
fx = tap.linearTAP(Gs, P, alpha=T * alpha_vec)[0]
while min(fx) < 0:
    T /= 10
    Gs = flow_subgraph(Gs, dict(zip(Gs.edges, fx)))
    alpha_vec = np.array(list(nx.get_edge_attributes(Gs, "alpha").values()))
    fx = tap.linearTAP(Gs, P, alpha=T * alpha_vec)[0]
    print(min(fx))

# fx = tap.linearTAP(Gs, P)[0]
pl.graphPlot(Gs, ec=fx)

# %%

fuedict = dict(zip(G.edges, fue))
fxdict = dict(zip(Gs.edges, fx))
deltaf = compare_flows(fuedict, fxdict)

cmap = plt.cm.coolwarm
norm = mpl.colors.TwoSlopeNorm(
    vmin=min(deltaf.values()), vmax=max(deltaf.values()), vcenter=0
)
pl.graphPlot(G, ec=deltaf, cmap=cmap, norm=norm)
# %%
