# %%
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import itertools


# %%
def travel_time(edge, n):
    return edge["tt_function"](n)


def assign_travel_times(G):
    for edge in G.edges():
        G.edges[edge]["travel_time"] = travel_time(G.edges[edge], G.edges[edge]["load"])
    return G


def travel_time_path(G, path):
    edge_path = ((path[i], path[i + 1]) for i in range(len(path) - 1))
    return np.sum(
        [travel_time(G.edges[edge], G.edges[edge]["load"]) for edge in edge_path]
    )


def potential_energy(edge, kwd="load"):
    load = edge[kwd]
    return np.sum([travel_time(edge, i) for i in range(int(load))])


def total_potential_energy(G, kwd="load"):
    return np.sum([potential_energy(edge, kwd) for edge in G.edges.values()])


def social_cost(edge, kwd="load"):
    load = edge[kwd]
    return load * travel_time(edge, load)


def total_social_cost(G, kwd="load"):
    return np.sum([social_cost(edge, kwd) for edge in G.edges.values()])


def assign_random_loads(G, total_load):
    # Generate random numbers
    random_weights = np.random.random(len(G.edges))

    # Normalize these weights so their sum equals total_load
    scaled_weights = np.floor(
        random_weights / random_weights.sum() * total_load
    ).astype(int)

    # Correct for any rounding error
    remainder = total_load - scaled_weights.sum()
    scaled_weights[:remainder] += 1

    for edge, w in zip(G.edges(), scaled_weights):
        G.edges[edge]["load"] = int(w)
    return G


def assign_initial_loads(G, start_nodes, end_nodes, total_load):
    # loads = nx.get_edge_attributes(G, "load")
    # if len(loads) == 0:
    nx.set_edge_attributes(G, 0, "load")

    G = assign_travel_times(G)

    if type(start_nodes) == int:
        start_nodes = [start_nodes]
    if type(end_nodes) == int:
        end_nodes = [end_nodes]

    for o in start_nodes:
        o_load = int(round(total_load / (len(start_nodes) * len(end_nodes))))
        for d in end_nodes:
            paths = nx.shortest_simple_paths(G, o, d, weight="travel_time")

            path = list(paths)[-1]

            for i, j in zip(path[:-1], path[1:]):
                G.edges[(i, j)]["load"] += o_load
    return G


def path_time(path):
    path_time = 0
    for i, j in zip(path[:-1], path[1:]):
        load = G.edges[(i, j)]["load"]
        time = G.edges[(i, j)]["tt_function"](load)
        path_time += time
    return path_time


def user_equilibrium(G, start_nodes, end_nodes):
    if type(start_nodes) == int:
        start_nodes = [start_nodes]
    if type(end_nodes) == int:
        end_nodes = [end_nodes]

    T_total = np.array([])
    S_total = np.array([])

    for o in start_nodes:
        for d in end_nodes:
            sc_arr = [total_social_cost(G)]
            T_arr = [total_potential_energy(G)]

            while len(T_arr) < 2 or T_arr[-1] < T_arr[-2]:

                paths = nx.shortest_simple_paths(G, o, d, weight="travel_time")

                lpaths = list(paths)

                min_path = lpaths[0]
                max_path = lpaths[-1]

                # check if max_path has load>0 on all edges
                prod = np.prod(
                    [
                        G.edges[(i, j)]["load"]
                        for i, j in zip(max_path[:-1], max_path[1:])
                    ]
                )
                n = 0
                while prod == 0:
                    n += 1
                    max_path = lpaths[-1 - n]
                    prod = np.prod(
                        [
                            G.edges[(i, j)]["load"]
                            for i, j in zip(max_path[:-1], max_path[1:])
                        ]
                    )

                # update loads
                for i, j in zip(max_path[:-1], max_path[1:]):
                    G.edges[(i, j)]["load"] -= 1
                for i, j in zip(min_path[:-1], min_path[1:]):
                    G.edges[(i, j)]["load"] += 1

                G = assign_travel_times(G)
                sc_arr.append(total_social_cost(G))
                T_arr.append(total_potential_energy(G))

            # there may be multiple optimal configurations
            if len(lpaths) > 1:
                if travel_time_path(G, max_path) == travel_time_path(G, lpaths[-2]):
                    max_path = lpaths[-2]

            # revert last step
            for i, j in zip(max_path[:-1], max_path[1:]):
                G.edges[(i, j)]["load"] += 1
            for i, j in zip(min_path[:-1], min_path[1:]):
                G.edges[(i, j)]["load"] -= 1

            G = assign_travel_times(G)

            T_total = np.append(T_total, T_arr[:-1])
            S_total = np.append(S_total, sc_arr[:-1])

    return G, T_total, S_total


def to_undirected(graph, weight="weight"):
    U = nx.Graph()
    U.add_nodes_from(graph.nodes)
    for i, j, data in graph.edges(data=True):
        w = data[weight]
        if U.has_edge(i, j):
            current_weight = U.edges[i, j][weight]
            if w > current_weight:
                U.edges[i, j][weight] = w
        else:
            U.add_edge(i, j, **data)
    return U


def linear_flow(G, weight="weight", P=None):
    if P is None:
        Pdict = nx.get_node_attributes(G, "P")
        P = np.array(list(Pdict.values()))
    # I = nx.incidence_matrix(G, oriented=True, weight="weight")
    # return np.linalg.pinv(I.toarray()) @ P_vec
    return linear_flow_solver(G, P, weight=weight)


def phases(G):
    P = nx.get_node_attributes(G, "P")
    P_vec = np.array(list(P.values()))
    return phase_solver(G, P_vec)


def linear_flow_solver(G, P, weight="weight"):
    if nx.is_directed(G):
        gamma = 1 / 2
        # L = nx.laplacian_matrix(to_undirected(G, weight=weight), weight=weight)
    else:
        gamma = 1
    L = nx.laplacian_matrix(G, weight=weight)

    # L = nx.laplacian_matrix(to_undirected(G), weight="weight")
    psi = np.linalg.pinv(L.toarray()) @ P
    nx.set_node_attributes(G, dict(zip(G.nodes(), psi)), "psi")

    incidence = -nx.incidence_matrix(G, oriented=True, weight=weight)

    return dict(zip(G.edges(), gamma * incidence.T @ psi))


def phase_solver(G, P):
    L = nx.laplacian_matrix(to_undirected(G), weight="weight")
    psi = np.linalg.pinv(L.toarray()) @ P
    return psi


# %%

if __name__ == "__main__":

    G = nx.DiGraph()
    G.add_nodes_from(
        [
            (0, {"pos": (0, 0.5)}),
            (1, {"pos": (0.5, 0)}),
            (2, {"pos": (0.5, 1)}),
            (3, {"pos": (1, 0.5)}),
        ]
    )

    G.add_edges_from(
        [
            (0, 1, {"tt_function": lambda n: 10}),
            (0, 2, {"tt_function": lambda n: n / 100}),
            (1, 3, {"tt_function": lambda n: n / 100}),
            (2, 3, {"tt_function": lambda n: 10}),
            # (2, 1, {"tt_function": lambda n: 0}),
        ]
    )

    start_nodes = [0]
    end_nodes = [3]
    total_load = 1000

    G = assign_initial_loads(G, start_nodes, end_nodes, total_load)

    G, T, S = user_equilibrium(G, start_nodes, end_nodes)
    print("Potential energy: ", T[-1])
    print("Social cost: ", S[-1])

    fig = plt.figure()

    pos = nx.get_node_attributes(G, "pos")
    nx.draw(G, pos)
    labels = nx.get_edge_attributes(G, "load")
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # %%
    fig, ax = plt.subplots()
    plt.plot(T, label="Potential energy")
    plt.plot(S, label="Social cost")
    plt.grid()
    plt.legend()


# %%
