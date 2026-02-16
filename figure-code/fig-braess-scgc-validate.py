# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

import osmnx as ox
import cvxpy as cp
from matplotlib.gridspec import GridSpec


from src import Plotting as pl
from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc
from src import SocialCost as sc
from src import TAPOptimization as tap
from src import Graphs as gr
from src import osmGraphs as og


def middle_value(arr):
    n = len(arr)
    if n % 2 == 1:  # Odd length
        return arr[n // 2]
    else:  # Even length
        return (arr[n // 2 - 1] + arr[n // 2]) / 2


# %%
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 1)]
alpha = np.array([0.1, 1, 1, 0.1, 0.1])
beta = np.array([5, 1, 1, 5, 1])
G = nx.DiGraph()
G.add_edges_from(edges)

nx.set_edge_attributes(G, dict(zip(G.edges, alpha)), "alpha")
nx.set_edge_attributes(G, dict(zip(G.edges, beta)), "beta")

pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 1), 3: (1.5, 1)}
nx.set_node_attributes(G, pos, "pos")

relabel_dict = {0: "A", 1: "B", 2: "C", 3: "D"}
G = nx.relabel_nodes(G, relabel_dict)


# %%
# G = gr.random_graph(50, 7, alpha=1, beta=1, seed=1)
source = 0

num = 9.5
P = -np.ones(G.number_of_nodes()) * num / (len(G.nodes) - 1)
P[source] = num


f_mat, lambda_mat = mc.solve_multicommodity_tap(G, [P], return_fw=True)
F = np.sum(f_mat, 0)

# %%

derivatives = mcsc.derivative_social_cost(G, f_mat, [P], eps=1e-3)

derivative_list = [derivatives.get(e, 0) for e in G.edges]
derivatives = dict(zip(G.edges, derivative_list))


# %%

fig = plt.figure(figsize=(14, 6))
labels = [
    r"\textbf{a}",
    r"\textbf{b}",
    r"\textbf{c}",
    r"\textbf{d}",
    r"\textbf{e}",
    r"\textbf{f}",
]

gs = GridSpec(4, 4, figure=fig, width_ratios=[2, 1, 1, 1])

ax1 = fig.add_subplot(gs[:, 0])

ax1.text(0.25, 0.95, labels[0], transform=ax1.transAxes, fontsize=22, fontweight="bold")

cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(
    vmin=min(derivative_list), vmax=max(derivative_list), vcenter=0
)

pl.graphPlot(
    G,
    ax=ax1,
    ec=derivative_list,
    edge_labels=derivatives,
    cmap=cmap,
    norm=norm,
    edgewith=6,
    cbar=False,
    title="",
)
cbar = plt.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    orientation="horizontal",
    ax=ax1,
    shrink=1 / 2,
    pad=-0.05,
    aspect=20,
)

cbar.ax.set_xlabel(r"SCGC $\frac{\partial}{\partial\beta_e}sc(\mathbf{\beta_e})$")

for i, e in enumerate(G.edges):
    if i < 4:  # First four edges in rows 0 and 3
        row = (i % 2) * 2  # Row 0 or 3 for symmetry
        col = int(np.floor(1 + i / 2))
        ax = fig.add_subplot(gs[row : row + 2, col])
    else:  # Last subplot between rows 1 and 2 in the last column
        ax = fig.add_subplot(gs[1:3, 3])  # Spans rows 1 and 2, last column

    ax.text(
        0.01,
        1.02,
        labels[i + 1],
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
    )

    slope_list, beta_list, cost = mcsc.numerical_derivative(
        G, [P], e, var_percentage=0.25
    )

    ax.plot(beta_list, cost, linewidth=2, marker=".", color="grey")

    beta_middle = beta_list[10:15]
    cost_middle = cost[10:15]
    slope = middle_value(slope_list)

    c = cmap(norm(slope))

    ax.plot(
        beta_middle,
        cost_middle,
        color=c,
        marker=".",
        markersize=10,
        linewidth=2.5,
        label=rf"$ \frac{{\partial}}{{\partial\beta_e}}sc(\mathbf{{\beta_e}})$ = {slope:.2f}",
    )

    ax.legend(loc="upper right")
    e_label = (
        str(e)
        .replace("(", "")
        .replace(")", "")
        .replace(",", r"$\rightarrow$")
        .replace("'", "")
    )
    ax.title.set_text(f"Edge {e_label}")

    ax.grid()

#

fig.text(
    2 / 3 + 0.04, 0.05, r"Free flow travel time $\beta_e$", ha="center", va="center"
)
fig.text(
    1 / 3 + 0.02, 0.5, r"Social cost", ha="center", va="center", rotation="vertical"
)


fig.savefig("figs/classic-braess-social-cost.pdf", dpi=300, bbox_inches="tight")


# %%
def sub_incidence_matrix(G, F, eps=1e-2):
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    neg_flow_indices = np.where(F <= eps)[0]
    E[:, neg_flow_indices] = 0
    return E


# E = -nx.incidence_matrix(U, oriented=True).toarray()
E = sub_incidence_matrix(G, F)
L = E @ np.diag(1 / alpha) @ E.T
Linv = np.linalg.pinv(L)

dsc = (E.T @ Linv @ P) / alpha
dsc

# %%
delta = (E.T @ Linv @ E @ (beta / alpha)) / alpha - beta / alpha
delta
# %%

G = gr.random_planar_graph(5, seed=42)
nodes, edges = ox.graph_to_gdfs(G)

selected_nodes = og.select_evenly_distributed_nodes(nodes, 3)
demands = np.array(og.demand_list(nodes, commodity=selected_nodes, gamma=1))


f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, demands, verbose=True, return_fw=True, solver=cp.OSQP
)
F = np.sum(f_mat, 0)
f_mat1, lambda_mat1 = mc.solve_multicommodity_tap(
    G, demands, verbose=True, return_fw=True, solver=cp.OSQP
)
F1 = np.sum(f_mat1, 0)


# %%
mcsc.test_system(G, f_mat1, lambda_mat1, demands)

# %%


def inverse_coupling_matrix(G, f_mat, eps=1e-1):
    alpha_ = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    num_layers = len(f_mat)
    binary_f_mat = np.where(f_mat > eps, 1, 0)

    flow_count = np.sum(binary_f_mat, axis=0)

    alpha_count = flow_count * alpha_
    layered_alpha_count = np.array([alpha_count for _ in range(num_layers)]).flatten()

    layered_alpha_count = layered_alpha_count * binary_f_mat.flatten()
    layered_alpha_count = layered_alpha_count[layered_alpha_count > 0]
    kappa = np.diag(1 / layered_alpha_count)
    return kappa


Kinv = inverse_coupling_matrix(G, f_mat, eps=1e-2)
K = mcsc.generate_coupling_matrix(G, f_mat, eps=1e-2)

mu = 1e-5
K_tilde = K - mu * np.eye(K.shape[0])
K_tilde_inv = np.linalg.inv(K_tilde)


np.allclose(K @ Kinv, K @ K_tilde_inv)

# %%
eps = 1e-2

num_layers = len(f_mat)

f_vec = np.hstack(f_mat)
p_vec = demands.flatten()
beta_arr = np.array(list(nx.get_edge_attributes(G, "beta").values()))
beta_vec = np.array([beta_arr for _ in range(num_layers)]).flatten()
lambda_vec = np.hstack(np.diff(lambda_mat))

EE = mcsc.layered_edge_incidence_matrix(G, f_mat, eps=eps)
K_inv = inverse_coupling_matrix(G, f_mat, eps=eps)

LL = EE @ K_inv @ EE.T

D = np.linalg.pinv(-LL)

C = -D @ EE @ K_inv

rhs = -C @ beta_vec[f_vec >= eps] + D @ p_vec

lhs = lambda_vec

np.diff(lhs) - np.diff(rhs)

# %%
