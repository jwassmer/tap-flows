# %%
from src import Equilibirium as eq
import networkx as nx
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

from src import Plotting as pl

pl.mpl_params(fontsize=22)


# %%
def linear_function(alpha, beta, x):
    return alpha + beta * x


def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    for e, f in lambda_funcs.items():
        G.edges[e]["weight"] = 1 / f(1) - f(0)
    return G


def graphPlot(graph, start_node=None, end_node=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    weights = nx.get_edge_attributes(graph, "weight")
    pos = nx.get_node_attributes(graph, "pos")
    loads = nx.get_edge_attributes(graph, "load")

    edge_colors = nx.get_edge_attributes(graph, "color")
    node_colors = nx.get_node_attributes(graph, "color")

    edge_labels = {e: "" for e in graph.edges()}

    if len(loads) > 0:
        loads = {k: round(v, 3) for k, v in loads.items()}
        for e, l in loads.items():
            eidx = e[0] + e[1]
            edge_labels[e] += rf"$x_{{{eidx}}}$={l}" + "\n"

    F = nx.get_edge_attributes(graph, "flow")
    if len(F) > 0:
        F = {k: round(v, 3) for k, v in F.items()}
        for e, v in F.items():
            edge_labels[e] += f"F={v}"

    if len(weights) > 0:
        weights = {k: round(v, 3) for k, v in weights.items()}
        for e, w in weights.items():
            edge_labels[e] += f"\nW={w}"

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors.values())
    nx.draw_networkx_labels(graph, pos)

    for u, v in graph.edges():
        if (v, u) in graph.edges():
            # Draw with curvature if bidirectional
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                connectionstyle="arc3,rad=0.1",
                edge_color=edge_colors[(u, v)],
            )
            sublabels = {(u, v): edge_labels[(u, v)]}
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=sublabels,
                connectionstyle="arc3,rad=0.2",
            )

        else:
            # Draw straight lines if not bidirectional
            sublabels = {(u, v): edge_labels[(u, v)]}
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                connectionstyle="arc3,rad=0",
                edge_color=edge_colors[(u, v)],
            )
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=sublabels,
                connectionstyle="arc3,rad=0",
            )

    if start_node is not None and end_node is not None:
        weights = nx.get_edge_attributes(graph, "weight")
        reciprocal_weights = {k: np.divide(1, v) for k, v in weights.items()}
        nx.set_edge_attributes(graph, reciprocal_weights, "reciprocal_weight")
        shortest_path = nx.shortest_path(
            graph, start_node, end_node, weight="reciprocal_weight"
        )
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=path_edges,
            edge_color="green",
            width=0.5,
            style="dashed",
        )

    return fig


def BraessGraph(start_node, end_node):
    G2 = nx.DiGraph()
    G2.add_nodes_from(
        [
            (0, {"pos": (0, 0.5)}),
            (1, {"pos": (0.5, 0)}),
            (2, {"pos": (0.5, 1)}),
            (3, {"pos": (1, 0.5)}),
            (4, {"pos": (0.2, 0.1)}),
            # (5, {"pos": (0.25, 0.1)}),
        ]
    )

    G2.add_edges_from(
        [
            (0, 4, {"tt_function": lambda n: n}),
            (4, 1, {"tt_function": lambda n: n}),
            (0, 2, {"tt_function": lambda n: n}),
            # (5, 1, {"tt_function": lambda n: n / 2}),
            (1, 3, {"tt_function": lambda n: n}),
            (2, 3, {"tt_function": lambda n: n}),
            (2, 1, {"tt_function": lambda n: n}),
        ]
    )
    G2 = updateEdgeWeights(G2)
    nx.set_edge_attributes(G2, "black", "color")
    nx.set_node_attributes(G2, "lightblue", "color")
    G2 = addGraphAttributes(G2, start_node, end_node)

    return G2


def build_graph(start_node, end_node):
    G1 = nx.DiGraph()

    G1.add_nodes_from(
        [
            (0, {"pos": (0, 0)}),
            (1, {"pos": (0.0, 1)}),
            (2, {"pos": (0.5, 0.5)}),
            (3, {"pos": (1, 2 / 3)}),
            (4, {"pos": (0.5, 0)}),
            (6, {"pos": (1, 0)}),
            (5, {"pos": (1.5, 1 / 3)}),
        ]
    )

    G1.add_edges_from(
        [
            # (1, 5, {"tt_function": lambda n: n}),
            (1, 3, {"tt_function": lambda n: n}),
            (0, 1, {"tt_function": lambda n: n}),
            (2, 3, {"tt_function": lambda n: n}),
            (1, 4, {"tt_function": lambda n: n}),
            # (4, 2, {"tt_function": lambda n: n}),
            (3, 5, {"tt_function": lambda n: n}),
            (4, 6, {"tt_function": lambda n: n}),
            (6, 5, {"tt_function": lambda n: n}),
            (0, 2, {"tt_function": lambda n: n}),
            # double edges
            # (3, 4, {"tt_function": lambda n: n / 4}),
            # (1, 0, {"tt_function": lambda n: n / 2}),
            # (3, 2, {"tt_function": lambda n: n}),
            # (2, 1, {"tt_function": lambda n: 2*n}),
        ]
    )

    # for e in G1.edges():
    #    G1.edges[e]["tt_function"] = lambda n: n

    P = np.zeros(G1.number_of_nodes())
    if type(start_node) == int:
        start_node = [start_node]
    if type(end_node) == int:
        end_node = [end_node]
    start_node = np.array(start_node)
    end_node = np.array(end_node)

    oval = 1 / len(start_node)
    dval = -1 / len(end_node)
    P[start_node], P[end_node] = oval, dval

    nx.set_node_attributes(G1, dict(zip(G1.nodes, P)), "P")

    G1 = updateEdgeWeights(G1)
    nx.set_edge_attributes(G1, "black", "color")
    nx.set_node_attributes(G1, "lightgrey", "color")

    for s in start_node:
        G1.nodes[s]["color"] = "blue"
    for e in end_node:
        G1.nodes[e]["color"] = "red"
    # G1.nodes[start_node]["color"] = "blue"
    # G1.nodes[end_node]["color"] = "red"

    return G1


def addGraphAttributes(G, start_node, end_node):
    P = np.zeros(G.number_of_nodes())
    if type(start_node) == int:
        start_node = [start_node]
    if type(end_node) == int:
        end_node = [end_node]
    start_node = np.array(start_node)
    end_node = np.array(end_node)

    oval = 1 / len(start_node)
    dval = -1 / len(end_node)
    P[start_node], P[end_node] = oval, dval

    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    for s in start_node:
        G.nodes[s]["color"] = "blue"
    for e in end_node:
        G.nodes[e]["color"] = "red"

    pos = nx.get_edge_attributes(G, "pos")
    if len(pos) == 0:
        pos = nx.spring_layout(G, seed=42)
        nx.set_node_attributes(G, pos, "pos")

    tt_funcs = nx.get_edge_attributes(G, "tt_function")
    if len(tt_funcs) == 0:
        for e in G.edges():
            G.edges[e]["tt_function"] = lambda n: n

    return G


def randomGraph(n, p, start_node=0, end_node=1, seed=42):
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)

    largest = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest)
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes)})

    for e in G.edges():
        G.edges[e]["tt_function"] = lambda n: n

    P = np.zeros(G.number_of_nodes())
    P[start_node], P[end_node] = 1, -1

    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    G.nodes[start_node]["color"] = "blue"
    G.nodes[end_node]["color"] = "red"

    reciprocal_weights = {
        k: np.divide(1, v) for k, v in nx.get_edge_attributes(G, "weight").items()
    }
    nx.set_edge_attributes(G, reciprocal_weights, "reciprocal_weight")
    pos = nx.spring_layout(G, weight="reciprocal_weight", seed=seed)
    nx.set_node_attributes(G, pos, "pos")
    return G


def tuneEdgeWeights(G, edge, alpha, beta, start_node, end_node, total_load):
    edg_idx = list(G.edges).index(edge)

    G.edges[edge]["color"] = "blue"
    G.edges[edge]["tt_function"] = lambda n: linear_function(alpha, beta, n)

    G = eq.assign_initial_loads(G, start_node, end_node, total_load)
    G, _, _ = eq.user_equilibrium(G, start_node, end_node)

    nash_equilibrium = nx.get_edge_attributes(G, "load")

    P = np.zeros(G.number_of_nodes())
    P[start_node], P[end_node] = 1, -1

    gamma_update = beta ** (-1)
    condition = False
    n = 0
    while not condition:
        gamma = gamma_update
        G.edges[edge]["weight"] = gamma
        F = eq.linear_flow_solver(G, P)
        negative_flows = {e: f for e, f in F.items() if f < -1e-3}

        if len(negative_flows) > 0:
            print(f"Negative flow after {n} iterations. Returning np.nan")
            nx.set_edge_attributes(G, F, "flow")
            return G, np.nan

        compare_arr = np.array(
            [f * total_load - nash_equilibrium[e] for e, f in F.items()]
        )

        condition = np.all(np.abs(compare_arr) < np.sqrt(2))

        if compare_arr[edg_idx] < 0:
            gamma_update = gamma * 1.111
        else:
            gamma_update = gamma * 0.999

        n += 1
        if n > 10_000:
            print("Did not converge in 10000 iterations.")
            nx.set_edge_attributes(G, F, "flow")

            return G, np.nan

    print("Converged in ", n, " iterations.")

    nx.set_edge_attributes(G, F, "flow")
    return G, gamma


def reducedFlowGraph(G):
    F = eq.linear_flow(G)

    # itereate negative flows
    epsilon = -1e-3 / nx.number_of_edges(G)

    negative_flows = {e: f for e, f in F.items() if f < epsilon}
    condition = len(negative_flows) > 0
    C = G.copy()
    # R = G.copy()
    while condition:
        min_key = min(negative_flows, key=negative_flows.get)

        # print("Negative flow detected. Setting edge weight to zero: ", min_key)
        C.edges[min_key]["color"] = "red"
        C.edges[min_key]["weight"] = 0
        F = eq.linear_flow(C)
        negative_flows = {e: f for e, f in F.items() if f < epsilon}
        condition = len(negative_flows) > 0

    # C = eq.assign_initial_loads(C, start_node, end_node, total_load)
    # C, _, _ = eq.user_equilibrium(C, start_node, end_node)
    nx.set_edge_attributes(C, F, "flow")

    return C


def dualFlowGraph(G):
    epsilon = -1e-3 / nx.number_of_edges(G)

    F = eq.linear_flow(G)

    negative_flows = {e: f for e, f in F.items() if f < epsilon}
    condition = len(negative_flows) > 0

    L, R = G.copy(), G.copy()
    C = G.copy()

    edge_list = []
    while condition:
        min_edge = min(negative_flows, key=negative_flows.get)

        C.edges[min_edge]["weight"] = 0

        F = eq.linear_flow(C)
        nx.set_edge_attributes(C, F, "flow")
        negative_flows = {e: f for e, f in F.items() if f < epsilon}
        condition = len(negative_flows) > 0

        edge_list.append(min_edge)

    for u, v in edge_list:
        if (u, v) in G.edges and (v, u) in G.edges:
            L.remove_edge(v, u)
            R.remove_edge(u, v)

    return R, L


def build_graph(start_node, end_node):
    G = nx.DiGraph()

    a, b, c, d, e, f = "a", "b", "c", "d", "e", "f"

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0)}),
            (b, {"pos": (0.0, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 1 / 3)}),
            (e, {"pos": (1, 1)}),
            (f, {"pos": (1.5, 1 / 3)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: n}),
            (b, e, {"tt_function": lambda n: n}),
            (e, f, {"tt_function": lambda n: n}),
            (b, d, {"tt_function": lambda n: n}),
            (a, c, {"tt_function": lambda n: n}),
            (c, d, {"tt_function": lambda n: n}),
            (d, f, {"tt_function": lambda n: n}),
            (c, d, {"tt_function": lambda n: n}),
            # (b, c, {"tt_function": lambda n: n}),
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
    P[start_node_int], P[end_node_int] = 1, -1

    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    G = updateEdgeWeights(G)
    nx.set_edge_attributes(G, "black", "color")
    nx.set_node_attributes(G, "lightgrey", "color")

    # for s in start_node:
    G.nodes[start_node]["color"] = "blue"
    # for e in end_node:
    G.nodes[end_node]["color"] = "red"

    planar = nx.check_planarity(G)
    if planar[0]:
        print("The graph is planar")
    else:
        print("The graph is NOT planar")

    return G


# %%
total_load = 1_000
start_node, end_node = "a", "f"
G = build_graph(start_node, end_node)
G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)
# G = reducedFlowGraph(G)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")

f = graphPlot(G)


# %%

loadflow = {e: total_load * G.edges[e]["flow"] for e in G.edges()}
nx.set_edge_attributes(G, loadflow, "loadflow")


print("Total potential energy, linflow: ", eq.total_potential_energy(G, "loadflow"))
print("Total potential energy, wardrop: ", eq.total_potential_energy(G, "load"))
eq.total_potential_energy(G, "loadflow") <= eq.total_potential_energy(G, "load")

# %%

L = nx.get_edge_attributes(G, "load")

s = 0
for e, v in F.items():
    s += total_load * v * (total_load * v + 1) / 2

s

# %%
