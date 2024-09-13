# %%
import networkx as nx
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Graphs as gr
from src import Plotting as pl
from src import SIBC as sibc
from src import TAPOptimization as tap


# %%
G = gr.squareLattice(3, beta="random", alpha="random", directed=True)
P = np.zeros(G.number_of_nodes())
P[0] = 1
P[np.delete(np.arange(G.number_of_nodes()), 0)] = -sum(P) / (G.number_of_nodes() - 1)


E = -nx.incidence_matrix(G, oriented=True).toarray()

f0 = sibc._single_source_interaction_betweenness_centrality(G, P=P, weight="beta")

start_time = time.time()
f = tap.user_equilibrium(G, P, positive_constraint=True)
print("Elapsed time:", time.time() - start_time)

cycle_f = f - f0
vmin, vmax = np.min(cycle_f), np.max(cycle_f)

cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
pl.graphPlot(G, ec=cycle_f, cmap=cmap, norm=norm, show_labels=True)

# %%
C, cycles = gr.cycle_edge_incidence_matrix(G)
Cmat = C
# %%

np.all(E @ C == 0)


# %%

start_time = time.time()
alpha_d = nx.get_edge_attributes(G, "alpha")
beta_d = nx.get_edge_attributes(G, "beta")
alpha = np.array(list(alpha_d.values()))
beta = np.array(list(beta_d.values()))

Bmat = C.T @ np.diag(alpha / 2) @ C

avec = (alpha * f0 + beta) @ C

n = len(avec)

l = cp.Variable(n)

# Define the objective function
objective = cp.Minimize(cp.quad_form(l, Bmat) + avec @ l)

# Define the constraints
constraints = [f0 + C @ l >= 0]


problem = cp.Problem(objective, constraints)
problem.solve()

# Output the results
print("Optimal value:", problem.value)
print("Optimal l:", l.value)

f1 = f0 + C @ l.value
print("Elapsed time:", time.time() - start_time)

Cinv = np.linalg.pinv(C.T)

lambda_vals = Cinv @ (2 * Bmat @ l.value + avec)

# %%
print(np.all(np.abs(f1 - f) < 1e-7))
print(max(np.abs(np.abs(f1 - f))))


# %%import numpy as np


# %%
B = Bmat
f_o = f0
C = Cmat
a = avec

max_iterations = 1000
tolerance = 1e-7

# Hessian matrix
H = np.block([[2 * B, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])

# Compute the Newton step
H_inv = np.linalg.pinv(H)  # Use pseudo-inverse in case H is singular

lambda_val = np.random.rand(C.shape[0])
l = np.random.rand(C.shape[1])


for iteration in range(max_iterations):
    # Gradient components
    grad_l = 2 * B @ l + a + C.T @ lambda_val
    grad_lambda = f_o + C @ l

    # Combine the gradient components into a single vector
    g = np.concatenate((grad_l, grad_lambda))

    delta = H_inv @ g

    # Update l and lambda
    l_new = l - delta[: l.shape[0]]
    lambda_new = lambda_val - delta[l.shape[0]]

    # if np.any(C @ l + f_o < 0):
    # l_new = np.maximum(0, l_new - (C.T @ np.maximum(0, -(C @ l_new + f_o))))

    # Check for convergence
    if (
        np.linalg.norm(l_new - l) < tolerance
        and np.linalg.norm(lambda_new - lambda_val) < tolerance
    ):
        l, lambda_val = l_new, lambda_new
        print(f"Converged after {iteration} iterations")
        break

    # Update l and lambda for the next iteration
    l, lambda_val = l_new, lambda_new


print(l)
print(lambda_val)
# %%
f1 = f0 + C @ l
# %%
E @ f1 - P
# %%
