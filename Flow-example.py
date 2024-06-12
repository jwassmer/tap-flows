# %%
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src import Equilibirium as eq
from src import Plotting as pl
import itertools

pl.mpl_params(fontsize=32)


# %%
def updateEdgeWeights(G):
    lambda_funcs = nx.get_edge_attributes(G, "tt_function")
    for e, f in lambda_funcs.items():
        G.edges[e]["weight"] = 1 / f(1) - f(0)
    return G


def linear_function(alpha, beta, x):
    return alpha + beta * x


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


def graphPlot(graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    tt_func = nx.get_edge_attributes(graph, "tt_function")
    weights = nx.get_edge_attributes(graph, "weight")
    pos = nx.get_node_attributes(graph, "pos")
    loads = nx.get_edge_attributes(graph, "load")
    flows = nx.get_edge_attributes(graph, "flow")

    edge_colors = nx.get_edge_attributes(graph, "color")
    node_colors = nx.get_node_attributes(graph, "color")

    edge_labels = {e: "" for e in graph.edges()}

    if len(tt_func) > 0:
        for e, f in tt_func.items():
            beta = int(tt_func[e](0))
            alpha = round(tt_func[e](1) - beta, 2)
            Kij = int(weights[e])
            edge_labels[e] += rf"$\alpha={alpha}$, $K={Kij}$" + "\n"

    if len(loads) > 0 and len(flows) > 0:
        loads = {k: round(v, 3) for k, v in loads.items()}
        flows = {k: round(v, 3) for k, v in flows.items()}
        for e, l in loads.items():
            edge_labels[e] += rf"$x$={l}" + "\n" + rf"F={flows[e]}"
    elif len(flows) > 0:
        flows = {k: round(v, 3) for k, v in flows.items()}
        for e, f in flows.items():
            edge_labels[e] += rf"F={f}"

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


def build_graph(start_node, end_node, total_load):
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
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

start_node, end_node = "a", "f"
total_load = 1_000
G = build_graph(start_node, end_node, total_load)

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")

graphPlot(G, ax=ax)

print("Total potential energy, linflow: ", eq.total_potential_energy(G, "flow"))
print("Total potential energy, nash TAP: ", eq.total_potential_energy(G, "load"))
eq.total_potential_energy(G, "flow") <= eq.total_potential_energy(G, "load")

# %%


fig, axs = plt.subplots(2, 1, figsize=(10, 8))
ax = axs[0]
ax.text(
    0.03,
    1.0,
    r"\textbf{(a)}",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    va="top",
)

start_node, end_node = "a", "f"
total_load = 1_000
G = build_graph(start_node, end_node, total_load)

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")

graphPlot(G, ax=ax)


ax = axs[1]
ax.text(
    0.03,
    1.0,
    r"\textbf{(b)}",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    va="top",
)

start_node, end_node = "a", "f"
total_load = 1_000
G = build_graph(start_node, end_node, total_load)
G.remove_edge("e", "d")
G.add_edge("d", "e", tt_function=lambda n: n / 2)
G = updateEdgeWeights(G)

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)
G = reducedFlowGraph(G)

F = eq.linear_flow(G)
nx.set_edge_attributes(G, F, "flow")

graphPlot(G, ax=ax)

# %%

fig.savefig("figs/Flow-example.pdf", bbox_inches="tight")

# %%
