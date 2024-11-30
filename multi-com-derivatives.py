# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import SocialCost as sc
from src import Plotting as pl
from src import TAPOptimization as tap

from sklearn.linear_model import LinearRegression

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# %%
G = gr.random_graph(10, num_edges=5, beta="random", alpha="random", directed=True)

od_matrix = -1 * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, (G.number_of_nodes() - 1) * 1)

od_matrix[0, :] *= 10

w = 2

od_matrix = [od_matrix[i, :] for i in range(w)]
od_matrix

# %%


def derivative_socia_cost_ab(G, Linv, P, edge, alpha_arr):
    a, b = edge[0], edge[1]

    a_idx = list(G.nodes).index(a)
    b_idx = list(G.nodes).index(b)

    edge_idx = list(G.edges).index(edge)
    P = np.array(P)

    slope = (Linv[a_idx, :] - Linv[b_idx, :]) @ P / alpha_arr[edge_idx]
    return slope


# %%

f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_matrix, pos_flows=True, return_fw=True
)
edge = list(G.edges)[0]
a, b = edge
a_idx = list(G.nodes).index(a)
b_idx = list(G.nodes).index(b)
edge_idx = list(G.edges).index(edge)


w = 0

U = gr.potential_subgraph(G, f_mat[w])
tap.linearTAP(U, od_matrix[w])


# %%

for w in range(len(od_matrix)):
    P = od_matrix[w]
    U = gr.potential_subgraph(G, f_mat[w])
    alpha_arr = np.array(list(nx.get_edge_attributes(U, "alpha").values()))
    L = gr.directed_laplacian(U)
    Linv = np.linalg.pinv(L)
    m = derivative_socia_cost_ab(G, Linv, P, edge, alpha_arr)
    print(m)


# %%
def linreg_slope_sc(G, od_matrix, edge):
    beta_d = nx.get_edge_attributes(G, "beta")
    beta_arr = np.array(list(beta_d.values()))

    beta_vals = np.linspace(-1e1, 1e1, 5)

    edge_idx = list(G.edges).index(edge)

    sc_beta = []
    for i, beta_e in enumerate(beta_vals):
        beta_arr[edge_idx] = beta_e
        f = mc.solve_multicommodity_tap(G, od_matrix, beta=beta_arr, pos_flows=True)
        s = sc.total_social_cost(G, f, beta=beta_arr)
        sc_beta.append(s)

    # Reshape the data for linear regression
    X = np.array(beta_vals).reshape(-1, 1)
    y = np.array(sc_beta)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    # y_intercept = model.intercept_
    m = model.coef_[0]
    return X, y, m


x, y, m = linreg_slope_sc(G, od_matrix, edge)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="m = %.2f" % m)
ax.grid()
ax.legend()
# %%


# %%
