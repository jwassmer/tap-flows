# %%
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

import networkx as nx
from src import Graphs as gr
from src import SIBC as sibc
from src import TAPOptimization as tap
from src import Plotting as pl
from src import multiCommodityTAP as mc

# %%

G = gr.random_graph(3, beta="random", alpha="random", directed=True)
print(G)

od_matrix = -1 * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, G.number_of_nodes() - 1)

demand_list = [od_matrix[:, n] for n in G.nodes]

E = -nx.incidence_matrix(G, oriented=True).toarray()

f_o_w = []
for demand in demand_list:
    f0 = sibc._single_source_interaction_betweenness_centrality(
        G, P=demand, weight="beta"
    )
    f_o_w.append(f0)


fp_multi = mc.solve_multicommodity_tap(G, demand_list)

fp_false_multi = np.zeros_like(fp_multi)
for w in range(len(f_o_w)):
    fp = tap.user_equilibrium(G, demand_list[w], positive_constraint=True)
    fp_false_multi += fp

C, cycles = gr.cycle_edge_incidence_matrix(G)

alpha_d = nx.get_edge_attributes(G, "alpha")
beta_d = nx.get_edge_attributes(G, "beta")
alpha_vec = np.array(list(alpha_d.values()))
beta_vec = np.array(list(beta_d.values()))


# %%
# Define parameters
B = C.T @ np.diag(alpha_vec / 2) @ C


a_w = []
for f0 in f_o_w:
    a = (alpha_vec * f0 + beta_vec) @ C
    a_w.append(a)

a_matrix = np.array(a_w).T


cmap = mpl.colormaps.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=max(f0))


# %%import numpy as np
import numpy as np
import time


def primal_dual_method(a_w, B, C, f_o_w, sigma=1e-7, tol=1e-7, max_iter=100):
    start_time = time.time()
    # Determine dimensions
    n = B.shape[0]  # Number of variables
    m = C.shape[0]  # Number of constraints
    w_len = len(a_w)  # Number of different values of w

    # Initialize variables
    ell = np.ones(n)  # Initial guess for primal variables
    mu = np.ones((m, w_len))  # Initial guess for dual variables for each w

    # Function to compute the residuals for all w
    def compute_residuals(ell, mu):
        r_d = (
            2 * B @ ell
            + sum(a_w[w] for w in range(w_len))
            - sum(C.T @ mu[:, w] for w in range(w_len))
        )
        r_p = np.array(
            [f_o_w[w] + C @ ell for w in range(w_len)]
        )  # Primal residual for each w
        r_c = np.array(
            [mu[:, w] * r_p[w] - sigma for w in range(w_len)]
        )  # Complementary slackness residual for each w
        return r_d, r_p.flatten(), r_c.flatten()

    # Primal-dual interior point method
    for k in range(max_iter):
        # Compute residuals
        r_d, r_p, r_c = compute_residuals(ell, mu)

        # Check for convergence
        if np.linalg.norm(r_d) < tol and np.linalg.norm(r_c) < tol:
            print("Converged after", k + 1, "iterations")
            break

        # Construct the Newton system
        diag_mu = np.diag(mu.flatten())  # Correct diagonal matrix for all w
        diag_r_p = np.diag(
            r_p.flatten()
        )  # Correctly create diagonal matrix from the flattened r_p vector
        C_block = np.vstack([C for _ in range(w_len)])  # Correctly stack C for all w
        KKT_matrix = np.block(
            [[2 * B, -C_block.T], [C_block, diag_mu @ np.eye(m * w_len)]]
        )

        # Construct the right-hand side
        rhs = -np.concatenate((r_d, r_c))

        # Solve the Newton system
        delta = np.linalg.solve(KKT_matrix, rhs)
        delta_ell = delta[:n]
        delta_mu = delta[n:].reshape(m, w_len)

        # Line search to ensure feasibility
        alpha = 1.0
        while any(
            np.any(f_o_w[w] + C @ (ell + alpha * delta_ell) < 0)
            or np.any(mu[:, w] + alpha * delta_mu[:, w] < 0)
            for w in range(w_len)
        ):
            alpha *= 0.5

        # Update primal and dual variables
        ell += alpha * delta_ell
        mu += alpha * delta_mu

    print("Time:", time.time() - start_time, "s")

    if k + 1 == max_iter:
        print("Did not converge after", k + 1, "iterations")
    return ell, mu


ell, mu = primal_dual_method(a_w, B, C, f_o_w, sigma=1e-7, tol=1e-7, max_iter=50)

ell, mu


# %%
fff = np.sum(f_o_w, axis=0) + C @ ell


# %%


inequaltiy_condition = np.all(fff >= -1e-10)
kcl_condition = np.all(np.abs(E @ fff) <= 1e-7)

# diff_fp = np.max(np.abs(fff - fp))

print("Inequality condition:", inequaltiy_condition)
print("KCL condition:", kcl_condition)
# print("Same as FP:", diff_fp)


# %%
fff
# %%
