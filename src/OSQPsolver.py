# %%
import osqp
import networkx as nx
import numpy as np
import time
import cvxpy as cp

from src import TAPOptimization as tap

G = tap.random_graph(seed=42, num_nodes=10, num_edges=12, alpha="random", beta="random")

# %%


def user_equilibrium(G, P, positive_constraint=True):
    ###############
    ###############
    # TODO: Implement using the  OSQP solver
    ################
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    E = -nx.incidence_matrix(G, oriented=True)

    tt_func = nx.get_edge_attributes(G, "tt_function")

    beta_e = np.array([tt_func[e](0) for e in G.edges()])
    alpha_e = np.array([tt_func[e](1) - tt_func[e](0) for e in G.edges()])

    # Define the variable F
    if positive_constraint:
        fe = cp.Variable(num_edges, nonneg=True)
    else:
        fe = cp.Variable(num_edges)

    # Define the objective function
    objective = cp.Minimize(1 / 2 * alpha_e @ fe**2 + beta_e @ fe)
    # objective = cp.Minimize(cp.sum(list(map(lambda f: f(0), func_list))))

    # Define the constraints
    constraints = [E @ fe == P]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    start_time = time.time()
    # Solve the problem
    problem.solve(verbose=False, solver=cp.OSQP, eps_rel=1e-7)

    # print("Optimal value:", problem.value)

    # ebc_linprog = np.sum(fe.value, axis=1)
    linprog_time = time.time() - start_time
    F = fe.value
    print("Time:", linprog_time, "s")
    print("Social cost:", tap.social_cost(G, F))
    print("Potential energy:", tap.potential_energy(G, F))
    print("Minimum:", problem.value)
    return F
