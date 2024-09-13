# %%
import numpy as np

import networkx as nx
from src import Graphs as gr
from src import SIBC as sibc
from src import TAPOptimization as tap

G = gr.triangularLattice(6, beta="random", alpha="random", directed=True)
print(G)
P = np.zeros(G.number_of_nodes())
P[0] = 100
P[np.delete(np.arange(G.number_of_nodes()), 0)] = -sum(P) / (G.number_of_nodes() - 1)


E = -nx.incidence_matrix(G, oriented=True).toarray()

f0 = sibc._single_source_interaction_betweenness_centrality(G, P=P, weight="beta")
fp = tap.user_equilibrium(G, P, positive_constraint=True)

C, cycles = gr.cycle_edge_incidence_matrix(G)

alpha_d = nx.get_edge_attributes(G, "alpha")
beta_d = nx.get_edge_attributes(G, "beta")
alpha = np.array(list(alpha_d.values()))
beta = np.array(list(beta_d.values()))

# Define parameters
B = C.T @ np.diag(alpha / 2) @ C

a = (alpha * f0 + beta) @ C


# %%


def primal_dual_method(a, B, C, f_o, sigma=1e-7, tol=1e-7, max_iter=100):
    # Initialize variables
    n = B.shape[0]  # Number of variables
    m = C.shape[0]  # Number of constraints
    ell = np.ones(n)  # Initial guess for primal variables
    mu = np.ones(m)  # Initial guess for dual variables

    # Function to compute the residuals
    def compute_residuals(ell, mu):
        r_d = 2 * B @ ell + a - C.T @ mu  # Dual residual
        r_p = f_o + C @ ell  # Primal residual
        r_c = mu * r_p - sigma  # Complementary slackness residual
        return r_d, r_p, r_c

    # Primal-dual interior point method
    for k in range(max_iter):
        # Compute residuals
        r_d, r_p, r_c = compute_residuals(ell, mu)

        # Check for convergence
        if np.linalg.norm(r_d) < tol and np.linalg.norm(r_c) < tol:
            print("Converged after", k + 1, "iterations")
            break

        # Construct the Newton system
        diag_mu = np.diag(mu)
        diag_r_p = np.diag(r_p)
        KKT_matrix = np.block([[2 * B, -C.T], [diag_mu @ C, diag_r_p]])
        # KKT_inv = np.linalg.pinv(KKT_matrix)

        rhs = -np.concatenate((r_d, r_c))

        # Solve the Newton system
        delta = np.linalg.solve(KKT_matrix, rhs)
        delta_ell = delta[:n]
        delta_mu = delta[n:]

        # Line search to ensure feasibility
        alpha = 1.0
        while np.any(f_o + C @ (ell + alpha * delta_ell) < 0) or np.any(
            mu + alpha * delta_mu < 0
        ):
            alpha *= 0.5

        # Update primal and dual variables
        ell += alpha * delta_ell
        mu += alpha * delta_mu

    if k + 1 == max_iter:
        print("Did not converge after", k + 1, "iterations")
    return ell, mu


ell, mu = primal_dual_method(a, B, C, f0, sigma=1e-7, tol=1e-7, max_iter=500)

# %%

fff = f0 + C @ ell

inequaltiy_condition = np.all(fff >= -1e-10)
kcl_condition = np.all(np.abs(E @ fff - P) <= 1e-7)

diff_fp = np.max(np.abs(fff - fp))

print("Inequality condition:", inequaltiy_condition)
print("KCL condition:", kcl_condition)
print("Same as FP:", diff_fp)


# %%
def primal_feasibility():
    return np.all(f0 + C @ ell > -1e-6)


def dual_feasibility():
    return np.all(mu >= -1e-6)


def complementary_slackness():
    return np.all(mu * (f0 + C @ ell) < 1e-6)


def stationarity():
    return np.all(2 * B @ ell + a - C.T @ mu < 1e-6)


print("Primal feasibility:", primal_feasibility())
print("Dual feasibility:", dual_feasibility())
print("Complementary slackness:", complementary_slackness())
print("Stationarity:", stationarity())

# %%
sigma = 1e-7
tol = 1e-7
max_iter = 200
# Initialize variables
f_o = f0
n = B.shape[0]  # Number of variables
m = C.shape[0]  # Number of constraints
ell = np.ones(n)  # Initial guess for primal variables
mu = np.ones(m)  # Initial guess for dual variables


# Function to compute the residuals
def compute_residuals(ell, mu):
    r_d = 2 * B @ ell + a - C.T @ mu  # Dual residual
    r_p = f_o + C @ ell  # Primal residual
    r_c = mu * r_p - sigma  # Complementary slackness residual
    return r_d, r_p, r_c


AA = 2 * B
AAinv = np.linalg.inv(AA)
BB = -C.T
# BBinv = np.linalg.inv(BB)
# Primal-dual interior point method
for k in range(max_iter):
    # Compute residuals
    r_d, r_p, r_c = compute_residuals(ell, mu)

    # Check for convergence
    if np.linalg.norm(r_d) < tol and np.linalg.norm(r_c) < tol:
        break

    # Construct the Newton system
    diag_mu = np.diag(mu)
    diag_r_p = np.diag(r_p)

    CC, DD = diag_mu @ C, diag_r_p
    DDinv = np.diag(1 / r_p)

    aa = np.linalg.inv(AA - BB @ DDinv @ CC)
    bb = -aa @ BB @ DDinv
    cc = -DDinv @ CC @ aa
    dd = DDinv + DDinv @ CC @ aa @ BB @ DDinv

    KKT_inv = np.block([[aa, bb], [cc, dd]])

    # KKT_matrix = np.block([[AA, BB], [CC, DD]])
    # KKT_inv = np.linalg.inv(KKT_matrix)

    rhs = -np.concatenate((r_d, r_c))

    # Solve the Newton system
    delta = KKT_inv @ rhs
    delta_ell = delta[:n]
    delta_mu = delta[n:]

    # Line search to ensure feasibility
    alpha = 1.0
    while np.any(f_o + C @ (ell + alpha * delta_ell) < 0) or np.any(
        mu + alpha * delta_mu < 0
    ):
        alpha *= 0.5

    # Update primal and dual variables
    ell += alpha * delta_ell
    mu += alpha * delta_mu

if k + 1 == max_iter:
    print("Did not converge after", k + 1, "iterations")
else:
    print("Converged after", k + 1, "iterations")


# ell, mu = primal_dual_method(a, B, C, f0, sigma=1e-7, tol=1e-7, max_iter=500)

# %%
