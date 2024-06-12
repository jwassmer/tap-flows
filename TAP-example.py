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
    for e, f in lambda_funcs.items():
        G.edges[e]["weight"] = 1 / f(1) - f(0)
    return G


def linear_function(alpha, beta, x):
    return alpha + beta * x


def build_graph(start_node, end_node):
    G = nx.DiGraph()

    a, b, c, d, e, f = "a", "b", "c", "d", "e", "f"

    G.add_nodes_from(
        [
            (a, {"pos": (0, 0.5)}),
            (b, {"pos": (0.5, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 0.5)}),
        ]
    )

    G.add_edges_from(
        [
            (a, b, {"tt_function": lambda n: 1 / 10 * n + 10}),
            (b, d, {"tt_function": lambda n: 1 / 10 * n + 10}),
            (a, c, {"tt_function": lambda n: 1 / 10 * n + 10}),
            (c, d, {"tt_function": lambda n: 1 / 10 * n + 10}),
            # (b, c, {"tt_function": lambda n: n}),
        ]
    )

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


def graphPlot(graph, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    tt_func = nx.get_edge_attributes(graph, "tt_function")
    pos = nx.get_node_attributes(graph, "pos")
    loads = nx.get_edge_attributes(graph, "load")

    edge_colors = nx.get_edge_attributes(graph, "color")
    node_colors = nx.get_node_attributes(graph, "color")

    edge_labels = {e: "" for e in graph.edges()}

    if len(tt_func) > 0:
        for e, f in tt_func.items():
            beta = int(tt_func[e](0))
            alpha = round(tt_func[e](1) - beta, 2)
            eidx = e[0] + e[1]

            # edge_labels[e] += (
            #    rf"$t_{{{eidx}}}(x_{{{eidx}}})={alpha}x_{{{eidx}}}+{beta}$" + "\n"
            # )

    if len(loads) > 0:
        loads = {k: round(v, 3) for k, v in loads.items()}
        for e, l in loads.items():
            eidx = e[0] + e[1]
            edge_labels[e] += rf"$x_{{{eidx}}}$={l}"

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
                font_size=18,
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
                font_size=18,
            )

    SCtot = int(round(eq.total_social_cost(graph)))
    Utot = int(round(eq.total_potential_energy(graph)))
    ax.text(
        0.5,
        0.45,
        rf"$SC_{{tot}}$: {SCtot}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
    )
    ax.text(
        0.5,
        0.55,
        rf"$U_{{tot}}$: {Utot}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
    )
    ax.axis("off")

    return ax


# %%
start_node, end_node = "a", "d"
total_load = 1000
G = build_graph(start_node, end_node)


# %%
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
G, Utot, SCtot = eq.user_equilibrium(G, start_node, end_node)

G = eq.assign_initial_loads(G, start_node, end_node, total_load)
loads = nx.get_edge_attributes(G, "load")
loaddiffs = [0, 1, 99, 400]

fig, axs = plt.subplots(2, 2, figsize=(8, 6))

for i, ax in enumerate(axs.flatten()):
    loads[("a", "b")] += loaddiffs[i]
    loads[("b", "d")] += loaddiffs[i]
    loads[("a", "c")] -= loaddiffs[i]
    loads[("c", "d")] -= loaddiffs[i]
    nx.set_edge_attributes(G, loads, "load")
    graphPlot(G, ax=ax)
    ax.text(
        0.15,
        0.85,
        panel_labels[i],
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
    )

ax = fig.add_axes([1.0, 0.1, 0.33, 0.8])
ax.grid()
ax.scatter(range(len(Utot)), Utot, label=r"Potential energy $U_{tot}$")
ax.scatter(range(len(SCtot)), SCtot, label=r"Social cost $SC_{tot}$")
ax.legend(fontsize=14)
ax.set_xlabel("Iteration", fontsize=18)
# ax.set_ylabel("Value", fontsize=18)

ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.tick_right()
ax.yaxis.set_tick_params(labelsize=16)
ax.yaxis.set_label_position("right")

ax.text(
    0.05,
    1.03,
    panel_labels[-1],
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=18,
)
# %%
fig.savefig("figs/fig1.pdf", bbox_inches="tight")


# %%
