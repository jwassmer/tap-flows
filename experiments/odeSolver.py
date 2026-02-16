# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# %%
def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    for e, f in lambda_funcs.items():
        G.edges[e]["weight"] = 1 / f(1) - f(0)
    return G


def build_graph(start_node, end_node, total_load):
    G = nx.DiGraph()

    a, b, c, d, e, f = 0, 1, 2, 3, 4, 5

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0)}),
            (b, {"pos": (0.0, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 1 / 3)}),
            (e, {"pos": (1, 1)}),
            (f, {"pos": (1.8, 1 / 3)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: n / 10}),
            (b, e, {"tt_function": lambda n: n / 2}),
            (e, f, {"tt_function": lambda n: n / 3}),
            (b, d, {"tt_function": lambda n: n / 1}),
            (a, c, {"tt_function": lambda n: n / 5}),
            (c, d, {"tt_function": lambda n: n / 4}),
            (d, f, {"tt_function": lambda n: n / 10}),
            (c, d, {"tt_function": lambda n: n / 20}),
            (e, d, {"tt_function": lambda n: n / 2}),
        ]
    )

    nodes_dict = dict(zip(G.nodes, [i for i in range(G.number_of_nodes())]))
    P = np.zeros(G.number_of_nodes())

    # if type(start_node) == int:
    #    start_node = [start_node]
    # if type(end_node) == int:
    #    end_node = [end_node]

    start_node_int = nodes_dict[start_node]
    end_node_int = nodes_dict[end_node]

    # oval = 1 / len(start_node)
    # dval = -1 / len(end_node)
    P[start_node_int], P[end_node_int] = total_load, -total_load

    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    # for s in start_node:
    G.nodes[start_node]["color"] = "lightblue"
    # for e in end_node:
    G.nodes[end_node]["color"] = "red"

    planar = nx.check_planarity(G)
    if planar[0]:
        print("The graph is planar")
    else:
        print("The graph is NOT planar")

    return G


# %%


def flow(G, psi):
    F = {}
    for i, j, K in G.edges(data="weight"):
        F[(i, j)] = -K * (psi[j] - psi[i])

    return F


def diffusion(t, theta, omega, A):
    dtheta_dt = omega - np.sum(A * np.subtract.outer(theta, theta), axis=1)
    return dtheta_dt


def diffusion_loop(t, theta, omega, A):
    dtheta_dt = np.zeros_like(theta)
    for i in range(len(theta)):
        for j in range(len(theta)):
            dtheta_dt[i] += A[i, j] * (theta[i] - theta[j])
    dtheta_dt = omega - dtheta_dt
    return dtheta_dt


def kuramoto(t, theta, omega, A):
    dtheta_dt = omega - np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
    return dtheta_dt


G = build_graph(0, 5, 1_000)


theta_0 = [0, 0, 0, 0, 0, 0]  # Initial phases


A = nx.adjacency_matrix(
    G.to_undirected(), weight="weight"
).toarray()  # Adjacency matrix


t_end = 10
t_span = (0, t_end)  # Time span for integration
t_eval = np.linspace(0, t_end, 1000)  # Times at which to store the computed solutions

P = np.array(list(nx.get_node_attributes(G, "P").values()))  # Power injections
solution = solve_ivp(
    diffusion, t_span, theta_0, args=(P, A), t_eval=t_eval, method="RK45"
)

# %%


F = flow(G, solution.y.T[-1])
F = {k: v for k, v in F.items()}
Fvec = np.array(list(F.values()))
Flabels = {k: f"{v:.2f}" for k, v in F.items()}


pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos)
# nx.draw_networkx_edge_labels(G, pos, edge_labels=w)
nx.draw_networkx_edge_labels(G, pos, edge_labels=Flabels)
# nx.draw_networkx_labels(G, pos, labels=p)
nx.draw_networkx_labels(G, pos)

# %%
