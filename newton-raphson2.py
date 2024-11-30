# %%

import numpy as np
import networkx as nx
import scipy.sparse as sp
import time

from src import Graphs as gr
from src import multiCommodityTAP as mc

# %%

G = gr.random_graph(200, beta="random", alpha="random", directed=True)
print(G)

od_matrix = -1 * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, G.number_of_nodes() - 1)

p_n_w = [od_matrix[:, n] for n in G.nodes]

F = mc.solve_multicommodity_tap(G, p_n_w, verbose=True, pos_flows=True)
F

E = -nx.incidence_matrix(G, oriented=True).toarray()


# %%


def fast_kkt_mat_inverse(alpha, E, p_n_w):
    # Compute the inverse of Hessian_f
    Hessian_f_inv = np.diag(1 / alpha)

    # A_tilde = A @ np.sqrt(Hessian_f_inv)
    A = np.vstack([E for _ in p_n_w])
    E_tilde = E @ np.sqrt(Hessian_f_inv)
    E_tilde_inv = np.linalg.pinv(E_tilde) / len(p_n_w)

    A_tilde_inv = np.hstack([E_tilde_inv for _ in p_n_w])
    # A_tilde_inv = np.linalg.pinv(A_tilde)

    # S_inv0 = A_inv.T @ Hessian_f_inv @ A_inv
    S_inv = A_tilde_inv.T @ A_tilde_inv

    # Compute the block inverse
    upper_left = Hessian_f_inv - Hessian_f_inv @ A.T @ S_inv @ A @ Hessian_f_inv
    upper_right = Hessian_f_inv @ A.T @ S_inv
    lower_left = -S_inv @ A @ Hessian_f_inv
    lower_right = S_inv

    # Form the inverse matrix
    KKT_matrix_inv = np.block([[upper_left, upper_right], [lower_left, lower_right]])
    return KKT_matrix_inv


def newton_raphson(G, demands, tolerance=1e-6, max_iterations=100):
    """
    Solve the multicommodity traffic assignment problem using Newton-Raphson method.
    ATM this does not respect inequality constraints.
    """
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    p = np.hstack(demands)  # Flattening p for all scenarios
    alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta = np.array(list(nx.get_edge_attributes(G, "beta").values()))
    num_edges = len(alpha)
    num_constraints = E.shape[0] * len(demands)  # Total constraints

    # Initialize variables
    f = np.zeros(num_edges)  # Initial guess for flows
    lambda_ = np.zeros(num_constraints)  # Initial guess for Lagrange multipliers

    # Block matrix for Newton's method (KKT system)
    # Hessian_f = np.diag(alpha)  # Hessian of the objective
    A = np.vstack([E for _ in demands])  # Extended E matrix for each scenario
    # KKT_matrix = np.block(
    #    [[Hessian_f, -A.T], [A, np.zeros((num_constraints, num_constraints))]]
    # )

    KKT_inv = fast_kkt_mat_inverse(alpha, E, demands)

    # Newton-Raphson iteration
    for iteration in range(max_iterations):
        # Gradient of the Lagrangian (with respect to f and lambda)
        grad_f = alpha * f + beta - A.T @ lambda_
        grad_lambda = A @ f - p

        # KKT conditions combined gradient vector
        grad = np.concatenate([grad_f, grad_lambda])

        # Update step: solve KKT system
        # step = np.linalg.solve(KKT_matrix, grad) ## KKt_matrix is singular
        # step = np.linalg.lstsq(KKT_matrix, grad, rcond=None)[0]
        step = KKT_inv @ grad
        delta_f = step[:num_edges]
        delta_lambda = step[num_edges:]

        # Update variables
        f -= delta_f
        lambda_ -= delta_lambda

        # Check for convergence
        if (
            np.linalg.norm(delta_f) < tolerance
            and np.linalg.norm(delta_lambda) < tolerance
        ):
            print(f"Converged after {iteration + 1} iterations.")
            return f

    print(f"Didn't converge after {max_iterations}.")
    return f


f = newton_raphson(G, demands=p_n_w)

# Check if the result matches the known optimal solution
f, np.allclose(f, F, atol=1e-6)


# %%

alpha = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
E = -nx.incidence_matrix(G, oriented=True).toarray()

num_constraints = E.shape[0] * len(p_n_w)  # Total constraints

Hessian_f = np.diag(alpha)  # Hessian of the objective
A = np.vstack([E for _ in p_n_w])  # Extended E matrix for each scenario


KKT_matrix = np.block(
    [[Hessian_f, -A.T], [A, np.zeros((num_constraints, num_constraints))]]
)

start_time = time.time()
fast_kkt_mat_inverse(alpha, E, p_n_w)
print("Fast inverse:", time.time() - start_time)

# %%
kkt_inv = np.linalg.pinv(KKT_matrix)
np.isclose(kkt_inv, fast_kkt_mat_inverse(alpha, E, p_n_w)).all()
# %%
# Compute the inverse of Hessian_f
Hessian_f_inv = np.diag(1 / alpha)

# A_tilde = A @ np.sqrt(Hessian_f_inv)
A = np.vstack([E for _ in p_n_w])
E_tilde = E @ np.sqrt(Hessian_f_inv)
E_tilde_inv = np.linalg.pinv(E_tilde) / len(p_n_w)

# %%
A_tilde_inv = np.hstack([E_tilde_inv for _ in p_n_w])
# A_tilde_inv = np.linalg.pinv(A_tilde)
# %%
# S_inv0 = A_inv.T @ Hessian_f_inv @ A_inv
S_inv = A_tilde_inv.T @ A_tilde_inv

# %%

# Compute the block inverse
upper_left = Hessian_f_inv - Hessian_f_inv @ A.T @ S_inv @ A @ Hessian_f_inv
upper_right = Hessian_f_inv @ A.T @ S_inv
lower_left = -S_inv @ A @ Hessian_f_inv
lower_right = S_inv

# Form the inverse matrix
KKT_matrix_inv = np.block([[upper_left, upper_right], [lower_left, lower_right]])
# %%
