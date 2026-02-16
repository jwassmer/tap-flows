# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
from src import Plotting as pl

pl.mpl_params(fontsize=32)


# %%
def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    nash_flow = nx.get_edge_attributes(G, "load")
    for e, f in lambda_funcs.items():
        beta = f(0)
        alpha = f(1) - beta
        x = nash_flow[e]  # / 1000
        if x != 0:
            K = x / (2 * beta + alpha * x)
        else:
            K = 0
        G.edges[e]["weight"] = K
    return G


def linear_function(alpha, beta, x):
    return alpha + beta * x


def graphPlot(graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pos = nx.get_node_attributes(graph, "pos")
    loads = nx.get_edge_attributes(graph, "load")
    flows = nx.get_edge_attributes(graph, "flow")
    tt_func = nx.get_edge_attributes(graph, "tt_function")
    weights = nx.get_edge_attributes(graph, "weight")

    edge_colors = nx.get_edge_attributes(graph, "color")
    node_colors = nx.get_node_attributes(graph, "color")

    edge_labels = {e: "" for e in graph.edges()}

    if len(tt_func) > 0:
        for e, f in tt_func.items():
            beta = int(tt_func[e](0))
            # if beta > 0:
            alpha = round(tt_func[e](1) - beta, 2)
            Kij = round(weights[e], 3)
            edge_labels[e] += (
                rf"$\alpha={alpha}$, $\beta={beta}$" + "\n" + rf"$K={Kij}$" + "\n"
            )

    if len(loads) > 0:
        loads = {k: round(v, 3) for k, v in loads.items()}
        flows = {k: round(v, 3) for k, v in flows.items()}
        for e, l in loads.items():
            edge_labels[e] += rf"$x$={l}" + "\n" + rf"F={flows[e]}"

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors.values())
    nx.draw_networkx_labels(graph, pos, ax=ax)

    for u, v in graph.edges():
        if (v, u) in graph.edges():
            # Draw with curvature if bidirectional
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                connectionstyle="arc3,rad=0.1",
                edge_color=edge_colors[(u, v)],
            )
            sublabels = {(u, v): edge_labels[(u, v)]}
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                ax=ax,
                edge_labels=sublabels,
                connectionstyle="arc3,rad=0.2",
                font_size=12,
            )

        else:
            # Draw straight lines if not bidirectional
            sublabels = {(u, v): edge_labels[(u, v)]}
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                connectionstyle="arc3,rad=0",
                edge_color=edge_colors[(u, v)],
            )
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                ax=ax,
                edge_labels=sublabels,
                connectionstyle="arc3,rad=0",
                font_size=12,
            )

    ax.axis("off")

    return ax


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
            (f, {"pos": (1.8, 1 / 3)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: n / 10 + 100}),
            (b, e, {"tt_function": lambda n: n / 10 + 100}),
            (e, f, {"tt_function": lambda n: n / 10 + 100}),
            # (b, d, {"tt_function": lambda n: n}),
            (a, c, {"tt_function": lambda n: n / 10 + 100}),
            (c, d, {"tt_function": lambda n: n / 10 + 100}),
            (d, f, {"tt_function": lambda n: n / 10 + 100}),
            (c, d, {"tt_function": lambda n: n / 10 + 100}),
            (b, d, {"tt_function": lambda n: n / 10 + 10}),
            (d, e, {"tt_function": lambda n: n / 10 + 10}),
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

    # G = updateEdgeWeights(G)
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


start_node, end_node = "a", "f"
total_load = 10_000
G = build_graph(start_node, end_node)

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)

G = updateEdgeWeights(G)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
graphPlot(G, ax=ax)

# %%

loadflow = {e: total_load * G.edges[e]["flow"] for e in G.edges()}
nx.set_edge_attributes(G, loadflow, "loadflow")
print("Total potential energy, linflow: ", eq.total_potential_energy(G, "loadflow"))
print("Total potential energy, nash TAP: ", eq.total_potential_energy(G, "load"))
eq.total_potential_energy(G, "loadflow") <= eq.total_potential_energy(G, "load")


# %%
