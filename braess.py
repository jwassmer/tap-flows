# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import Plotting as pl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LinearRegression

from src import Plotting as pl

# %%

G = gr.random_graph(21, num_edges=5, beta="random", alpha="random", directed=True)

P = -10 * np.ones(G.number_of_nodes())
origin = 0
P[origin] = np.abs(np.sum(np.delete(P, origin)))
# print(P)
f, lambda_ = tap.user_equilibrium(G, P, return_lagrange_multiplier=True)

# pl.graphPlot(G, ec=f)
cost = sc.total_social_cost(G, f)
cost

U = gr.potential_subgraph(G, f)
Eu = -nx.incidence_matrix(U, oriented=True).toarray()
alpha_u = np.array(list(nx.get_edge_attributes(U, "alpha").values()))
beta_u = np.array(list(nx.get_edge_attributes(U, "beta").values()))
# pl.graphPlot(U)
# %%
slopes = sc.all_braess_edges(U, P)
for e, s in slopes.items():
    if s < 0:
        print("Edge", e, s)

# %%
E = -nx.incidence_matrix(G, oriented=True).toarray()
alpha_arr = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
L = E @ np.diag(1 / alpha_arr) @ E.T
Linv = np.linalg.pinv(L)


edge = list(G.edges)[0]


a, b = edge[0], edge[1]

a_idx = list(G.nodes).index(a)
b_idx = list(G.nodes).index(b)

edge_idx = list(G.edges).index(edge)


(Linv[a_idx, :] - Linv[b_idx, :]) @ P / alpha_arr[edge_idx]


# %%
#### single origin
def try_all_edges(G, P):
    f = tap.user_equilibrium(G, P)
    cost = sc.total_social_cost(G, f)
    eps = 1e-2
    edges = list(G.edges)
    # print(edges)
    for i, edge in enumerate(edges):
        beta_d = nx.get_edge_attributes(G, "beta")
        U = G.copy()
        # U.remove_edge(*edge)
        beta_d[edge] *= 1.5
        beta_vec = np.array(list(beta_d.values()))

        f = tap.user_equilibrium(G, P, beta=beta_vec)
        try:
            c = sc.total_social_cost(G, f, beta=beta_vec)
            delta_c = c - cost
            # print(delta_c)
            if delta_c < -eps:
                # print(P)
                print(f"{edge}: Delta={delta_c}")
            # print(f"{edge}: Delta={delta_c}")
        except:
            pass
            # print(f"{edge}: Delta=inf")


# for o in range(G.number_of_nodes()):
o = origin
P = -10 * np.ones(G.number_of_nodes())
P[o] = np.abs(np.sum(np.delete(P, o)))
print(f"Origin: {o}")
try_all_edges(G, P)


# %%
#### multi commodity
def try_all_edges(G, demands):
    f = mc.solve_multicommodity_tap(G, demands)
    cost = sc.total_social_cost(G, f)
    eps = 1e-1
    edges = list(G.edges)
    for i, edge in enumerate(edges):
        U = G.copy()
        U.remove_edge(*edge)

        f = mc.solve_multicommodity_tap(U, demands)
        try:
            c = sc.total_social_cost(U, f)
            delta_c = c - cost
            if delta_c < -eps:
                # print(demands)
                print(f"{edge}: Delta={delta_c}")
            # print(f"{edge}: Delta={delta_c}")
        except:
            pass
            # print(f"{edge}: Delta=inf")


for o in range(G.number_of_nodes()):
    n = 1
    od_matrix = -n * np.ones((G.number_of_nodes(), G.number_of_nodes()))
    np.fill_diagonal(od_matrix, (G.number_of_nodes() - 1) * n)

    od_matrix[o, :] *= 10
    # print(f"Origin: {o}, Destination: {d}")
    print(f"Origin: {o}")
    try_all_edges(G, od_matrix)

# try_all_edges(G, P)


# demands = [od_matrix[:, n] for n in G.nodes]

# try_all_edges(G, demands)
# %%
# demands = [od_matrix[:, n] for n in G.nodes]
mc.solve_multicommodity_tap(G, od_matrix)
# edge = list(G.edges)[0]
beta_vals = np.linspace(0, 25, 25)

costs_edge = {}
for i, edge in enumerate(G.edges):
    costs = []
    for b in beta_vals:
        beta_vec = np.array(list(nx.get_edge_attributes(G, "beta").values()))
        beta_vec[i] = b

        f = mc.solve_multicommodity_tap(G, od_matrix, beta=beta_vec, pos_flows=False)
        # f = tap.user_equilibrium(
        #    G, od_matrix[0, :], beta=beta_vec, positive_constraint=False
        # )
        cost = sc.total_social_cost(G, f, beta=beta_vec)
        costs.append(cost)

    costs_edge[edge] = costs

# %%
for edge, c in costs_edge.items():
    plt.plot(beta_vals, c, label=edge, linewidth=5)
plt.grid()
plt.ylabel("social cost", fontsize=24)
plt.xlabel("beta", fontsize=24)
# plt.legend()
# %%
