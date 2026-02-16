# %%
import numpy as np
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib as mpl
import time
from scipy.sparse import csc_matrix


from src import Plotting as pl
from src import Graphs as gr
from src import SIBC as sibc
from src import TAPOptimization as tap

# %%
G = gr.random_graph(100, beta="random", alpha="random", directed=True)
print(G)
P = np.zeros(G.number_of_nodes())
P[0] = 100
P[np.delete(np.arange(G.number_of_nodes()), 0)] = -sum(P) / (G.number_of_nodes() - 1)


E = -nx.incidence_matrix(G, oriented=True).toarray()

f0 = sibc._single_source_interaction_betweenness_centrality(G, P=P, weight="beta")

f = tap.user_equilibrium(G, P, positive_constraint=True)


# %%

C, cycles = gr.cycle_edge_incidence_matrix(G)

alpha_d = nx.get_edge_attributes(G, "alpha")
beta_d = nx.get_edge_attributes(G, "beta")
alpha = np.array(list(alpha_d.values()))
beta = np.array(list(beta_d.values()))

# Define parameters
B = C.T @ np.diag(alpha / 2) @ C

a = (alpha * f0 + beta) @ C

# %%
start_time = time.time()

n = len(a)

l = cp.Variable(n)

# Define the objective function
objective = cp.Minimize(cp.quad_form(l, B) + a @ l)

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

lambda_vals = Cinv @ (2 * B @ l.value + a)
lambda_vals

np.allclose(f1, f)

# %%


# Initialize variables
l = np.zeros(B.shape[0])  # Initial guess for l, matching the size of B
lambda_ = np.ones(
    C.shape[0]
)  # Initial guess for lambda, matching the number of constraints
tolerance = 1e-7
max_iterations = 100

# Newton-Raphson method
for iteration in range(max_iterations):
    # Compute the gradients
    gradient_l = 2 * B @ l + a + C.T @ lambda_
    gradient_lambda = f0 + C @ l

    # Construct the KKT matrix
    J11 = 2 * B
    J12 = C.T
    J21 = np.diag(lambda_) @ C
    J22 = np.diag(f0 + C @ l)

    # Form the full Jacobian matrix
    J = np.block([[J11, J12], [J21, J22]])

    # Construct F(x)
    F = np.concatenate([gradient_l, lambda_ * gradient_lambda])

    try:
        delta_x = -np.linalg.inv(J) @ F  # Use pseudo-inverse to handle singularity
    except np.linalg.LinAlgError:
        print(
            "Numerical issues encountered; unable to proceed with Newton-Raphson update."
        )
        break

    # Update the variables
    delta_l = delta_x[: l.size]  # First part is for l
    delta_lambda = delta_x[l.size :]  # Remaining part is for lambda

    l += delta_l
    lambda_ += delta_lambda

    # Check convergence
    if np.linalg.norm(delta_x) < tolerance:
        print("Converged in {} iterations.".format(iteration + 1))
        break
else:
    print("Reached maximum iterations without convergence.")


fff = f0 + C @ l

inequaltiy_condition = np.all(fff >= -1e-10)
kcl_condition = np.all(np.abs(E @ fff - P) <= 1e-7)

print("Inequality condition:", inequaltiy_condition)
print("KCL condition:", kcl_condition)


# %%
# KKT conditions


def primal_feasibility():
    return np.all(f0 + C @ l > -1e-10)


def dual_feasibility():
    return np.all(lambda_ >= -1e-10)


def complementary_slackness():
    return np.all(lambda_ * (f0 + C @ l) < 1e-10)


def stationarity():
    return np.all(2 * B @ l + a + C.T @ lambda_ < 1e-10)


print("Primal feasibility:", primal_feasibility())
print("Dual feasibility:", dual_feasibility())
print("Complementary slackness:", complementary_slackness())
print("Stationarity:", stationarity())
# %%
