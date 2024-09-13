# %%
from src import Graphs as gr
from src import TAPOptimization as tap
from src import Plotting as pl

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


def flow_subgraph(G, feff):
    Gs = nx.DiGraph()
    Gs.add_nodes_from(G.nodes)
    eps = 1e-4

    for e, f in feff.items():
        if f > eps:
            Gs.add_edge(*e, alpha=G.edges[e]["alpha"], beta=G.edges[e]["beta"])
        if f < -eps:
            l = e[::-1]
            Gs.add_edge(*l, alpha=G.edges[l]["alpha"], beta=G.edges[l]["beta"])

    pos = nx.get_node_attributes(G, "pos")
    nx.set_node_attributes(Gs, pos, "pos")

    return Gs


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


def reverse_flow_matrix_sign(f):
    i, j = np.where(f < 0)
    f[i, j] = 0
    f[j, i] = 0
    return f


def reverse_flow_dict(f_dict):
    for e in f_dict:
        if f_dict[e] < 0:
            f_dict[e] = 0
            f_dict[(e[1], e[0])] = 0
    return f_dict


# %%
braess_beta = np.array(
    [
        [0, 0.01, 50, np.inf],
        [0.01, 0, np.inf, 50],
        [50, 10, 0, 0.01],
        [np.inf, 50, 0.01, 0],
    ]
)
braess_alpha = np.array(
    [[0, 10, 1, np.inf], [10, 0, np.inf, 1], [1, 1, 0, 10], [np.inf, 1, 10, 0]]
)
# P = np.array([-6, 0, 0, 6])
braess_conectivity = 1 * braess_alpha + braess_beta

braess_conectivity_plot = braess_conectivity.copy()
braess_conectivity_plot[braess_conectivity_plot == np.inf] = 0
n = braess_conectivity.shape[0]

# G = gr.triangularLattice(2, beta="random", alpha=1e-1)
G = nx.from_numpy_array(braess_conectivity_plot, create_using=nx.DiGraph)
pos = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 0)}
nx.set_node_attributes(G, pos, "pos")
for e in G.edges:
    i, j = e
    G.edges[i, j]["alpha"] = float(braess_alpha[i, j])
    G.edges[i, j]["beta"] = float(braess_beta[i, j])

# %%

G = gr.triangularLattice(2, beta="random", alpha=1e-1)
# beta = nx.get_edge_attributes(G, "beta")
# alpha = nx.get_edge_attributes(G, "alpha")
# pos = nx.get_node_attributes(G, "pos")

# G = gr.squareLattice(2, beta="random", alpha="random")
# G = gr.random_graph(10, beta="random", alpha="random")

# G.edges[(31, 26)]["alpha"] = 1e-8
P = np.zeros(G.number_of_nodes())
source = 0
sinks = np.delete(np.arange(G.number_of_nodes()), source)
P[source] = 6
P[sinks] = -np.sum(P) / len(sinks)

f0 = tap.linearTAP(G, P)
fp = tap.user_equilibrium(G, P, positive_constraint=True)

cmap = plt.cm.cividis
cmap.set_under("lightgrey")
cmap.set_bad("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=max(fp))
pl.graphPlot(G, ec=fp, show_labels=True, cmap=cmap, norm=norm)


# %%
# eff_beta = effective_beta(G, 0)
# nx.set_edge_attributes(G, eff_beta, "beta_eff")
Geff = effective_subgraph(G, depth=5, weight="beta")
feff0 = tap.linearTAP(Geff, P)[0]
cmap = plt.cm.coolwarm
norm = mpl.colors.Normalize(vmin=-max(feff0), vmax=max(feff0))
pl.graphPlot(Geff, ec=feff0, show_labels=True, cmap=cmap, norm=norm)

# %%
f_eff_dict = dict(zip(G.edges, fp))
Gs = flow_subgraph(G, f_eff_dict)
feff, lamb_eff = tap.linearTAP(Gs, P)
pl.graphPlot(Gs, ec=feff, show_labels=True)


# %%
