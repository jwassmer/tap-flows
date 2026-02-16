# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import ConvexOptimization as co
from src import Plotting as pl
from src import Equilibirium as eq
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc


def ODmatrix(G, load=100):
    num_nodes = G.number_of_nodes()
    A = -load * np.ones((num_nodes, num_nodes)) / (num_nodes - 1)
    np.fill_diagonal(A, load)
    return A


# %%
total_load = 100
# Example graph creation
G = gr.squareLattice(
    10,
    alpha="random",
    beta="random",
)
nodes = list(G.nodes)
A = ODmatrix(G)
demands = [A[:, n] for n in nodes]

fue, fs = mc.solve_multicommodity_tap(G, demands, verbose=False, pos_flows=True)

# %%

fso, fs = mc.solve_multicommodity_tap(G, demands, social_optimum=True)

diff = fue - fso
cmap = plt.cm.coolwarm
norm = mpl.colors.TwoSlopeNorm(vmin=diff.min(), vmax=diff.max(), vcenter=0)

pl.graphPlot(G, ec=fue - fso, cmap=cmap, norm=norm)


# %%
edge = (14, 11)
betas = np.linspace(0, 10, 25)

sc_so_list = []
sc_ue_list = []

for beta in betas:
    G.edges[edge]["beta"] = beta
    fso, _ = mc.solve_multicommodity_tap(G, demands, social_optimum=True)
    fue, _ = mc.solve_multicommodity_tap(G, demands)
    sc_so = sc.total_social_cost(G, fso)
    sc_ue = sc.total_social_cost(G, fue)

    sc_so_list.append(sc_so)
    sc_ue_list.append(sc_ue)


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(betas, sc_so_list, label="Social Optimum")
ax.plot(betas, sc_ue_list, label="User Equilibrium")
ax.legend()
ax.grid()
# %%


total_load = 100
# Example graph creation
G = gr.squareLattice(
    200,
    alpha="random",
    beta="random",
)

print(G)

P = np.zeros(G.number_of_nodes())
P[0] = total_load
P[-1] = -total_load

# %%


def TAP(G, P_w, positive_constraint=True):
    alpha_e = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta_e = np.array(list(nx.get_edge_attributes(G, "beta").values()))

    m = G.number_of_edges()
    w = len(P_w)

    E = -nx.incidence_matrix(G, oriented=True).toarray()

    # Define the variables
    if positive_constraint:
        f_e_w = cp.Variable((m, w), nonneg=True)
    else:
        f_e_w = cp.Variable((m, w))
    # f_e_w = cp.Variable((m, w), nonneg=True)  # Flow on edge e for each demand w

    # Objective function (quadratic cost)
    objective = cp.Minimize(
        0.5 * cp.sum(cp.multiply(alpha_e, cp.square(cp.sum(f_e_w, axis=1))))
        + cp.sum(cp.multiply(beta_e, cp.sum(f_e_w, axis=1)))
    )

    # Flow conservation constraints
    constraints = []
    for k in range(w):
        constraints += [E @ f_e_w[:, k] == P_w[k]]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True, verbose=True, solver=cp.OSQP)

    # Extract the optimal flow values
    f_e_w_optimal = f_e_w.value

    print("Time:", prob.solver_stats.solve_time)
    return f_e_w_optimal


G = gr.random_graph(50, beta="random", alpha="random", directed=True)
od_matrix = ODmatrix(G, load=1000)
demands = [od_matrix[:, n] for n in G.nodes]

fw = TAP(G, demands, positive_constraint=True)
fe = np.sum(fw, axis=1)


# %%

# %%
import numpy as np
import osqp
import networkx as nx
from scipy import sparse
import numpy as np
import osqp
import networkx as nx
from scipy import sparse


def TAP_osqp(G, P_w, positive_constraint=True):
    # Extract graph attributes for the edges
    alpha_e = np.array(list(nx.get_edge_attributes(G, "alpha").values()))
    beta_e = np.array(list(nx.get_edge_attributes(G, "beta").values()))

    m = G.number_of_edges()  # number of edges
    w = len(P_w)  # number of demands

    # Incidence matrix of the graph
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    # Define the number of variables (m edges x w demands)
    n_vars = m * w

    # Objective function
    # Quadratic term: we need to square the sum of flows for each edge
    # This means we need to sum the flow across demands for each edge and then apply the quadratic alpha_e cost.
    P_block = sparse.kron(sparse.eye(w), np.diag(alpha_e))
    P = P_block.T @ P_block  # Equivalent to summing squared terms
    P = sparse.triu(P)  # OSQP expects a sparse upper triangular matrix

    # Linear term: summing flows across demands for each edge and multiplying by beta_e
    q = np.tile(beta_e, w)

    # Flow conservation constraints for each demand
    A_eq_blocks = [E for _ in range(w)]
    A_eq = sparse.block_diag(A_eq_blocks)

    # Right-hand side of the equality constraints (demands)
    b_eq = np.hstack(P_w)

    # Inequality constraints (positive flow if needed)
    if positive_constraint:
        A_pos = sparse.eye(n_vars)
        l_pos = np.zeros(n_vars)  # Flows must be >= 0
        u_pos = np.inf * np.ones(n_vars)  # No upper bound
    else:
        A_pos = sparse.csr_matrix((0, n_vars))  # No inequality constraints
        l_pos = np.array([])
        u_pos = np.array([])

    # Combine equality and inequality constraints
    A = sparse.vstack([A_eq, A_pos])
    l = np.hstack([b_eq, l_pos])
    u = np.hstack([b_eq, u_pos])

    # Setup OSQP problem
    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=True, warm_start=True)
    result = solver.solve()

    # Reshape the result to match the (m, w) shape for flow on each edge for each demand
    f_e_w_optimal = result.x.reshape(m, w)

    print("Time:", result.info.solve_time)
    return f_e_w_optimal


fw1 = TAP_osqp(G, demands, positive_constraint=True)

f_e1 = np.sum(fw1, axis=1)

# %%
