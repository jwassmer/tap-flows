# %%
from src import Plotting as pl
from src import multiCommodityTAP as mc
from src import osmGraphs as og
from src import Graphs as gr

import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx
import cvxpy as cv
from tqdm import tqdm

from matplotlib import patches

from scipy.spatial import Delaunay

# %%s


def demand_list(nodes, commodity, gamma=0.1):

    demands = []
    tot_pop = 0
    for idx, row in commodity.iterrows():
        pop_com = row["population"]
        source_node = row.name
        target_nodes = nodes.index.difference([source_node])

        # random_com_node = np.random.choice(com_nodes)
        # nodes.loc[source_nodes, "source_node"] = True

        P = dict(zip(nodes.index, np.zeros(len(nodes))))
        # for node in com_nodes:
        # for node in source_nodes:
        P[source_node] = pop_com * gamma  # / len(source_nodes)

        # P[random_com_node] = pop_com
        for node in target_nodes:
            P[node] = -pop_com * gamma / len(target_nodes)
        # print(sum(P.values()))

        demands.append(list(P.values()))
        # print(c, pop_com)
        tot_pop += pop_com
    print("Total Population:", tot_pop)
    return demands


def flow_variability(G, gamma=0.1, maxlen=None, steps=25):
    flow_mat = []
    if maxlen is None:
        maxlen = len(G)
    if maxlen > len(G):
        maxlen = len(G)

    n_size = np.round(np.logspace(np.log10(3), np.log10(maxlen), num=steps)).astype(int)
    n_size = np.unique(n_size)
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    for n in tqdm(n_size):
        selected_nodes = og.select_evenly_distributed_nodes(nodes, n)
        demands = demand_list(nodes, commodity=selected_nodes, gamma=gamma)
        try:
            F = mc.solve_multicommodity_tap(G, demands, verbose=False, solver=cv.MOSEK)
        except:
            print("MOSEK failed. Trying OSQP")
            F = mc.solve_multicommodity_tap(G, demands, verbose=False, solver=cv.OSQP)

        # n = len(G)
        edges["flow"] = F
        edges["flow_per_distance"] = edges["flow"] / (edges["length"] * edges["lanes"])
        flow_mat.append(F)

    flow_mat = np.array(flow_mat).T
    return np.array(n_size), flow_mat


# %%
query = "Göttingen, Germany"
G, boundary = og.osmGraph(
    query,
    return_boundary=True,
)

# G = gr.random_planar_graph(100)
nodes, edges = ox.graph_to_gdfs(G)

selected_nodes = og.select_evenly_distributed_nodes(nodes, 100).set_geometry("voronoi")

demands = demand_list(nodes, commodity=selected_nodes, gamma=0.05)


F = mc.solve_multicommodity_tap(G, demands, solver=cv.MOSEK, verbose=False)

# %%


query = "Nippes, Cologne, Germany"
G, boundary = og.osmGraph(
    query,
    return_boundary=True,
)

# G = gr.random_planar_graph(100)
nodes, edges = ox.graph_to_gdfs(G)


F_list = []
selected_nodes_list = []
for n in [3, 15, len(G)]:
    print(n)
    selected_nodes = og.select_evenly_distributed_nodes(nodes, n).set_geometry(
        "voronoi"
    )

    demands = demand_list(nodes, commodity=selected_nodes, gamma=0.1)

    F = mc.solve_multicommodity_tap(G, demands, solver=cv.MOSEK, verbose=False)
    F_list.append(F)
    selected_nodes_list.append(selected_nodes)

# %%

"""
sizes = np.array([50, 100])  # ], 150, 200, 250])

std_random_g_dict = {}
std_std_random_g_dict = {}
nodes_random_g_dict = {}
for size in sizes:
    std_list = []
    for seed in range(3):
        G = gr.random_planar_graph(size, seed=seed)
        print(G)

        sel_nodes, flow_mat = flow_variability(G, gamma=1)
        std = np.std(flow_mat.T - flow_mat[:, -1], 1)
        std_list.append(std)

    std_random_g_dict[size] = np.mean(std_list, 0)
    std_std_random_g_dict[size] = np.std(std_list, 0)
    nodes_random_g_dict[size] = sel_nodes
"""

# %%
query_list = [
    "Nippes, Cologne, Germany",
    "Pankow, Berlin, Germany",
    "Potsdam, Germany",
    # "Bonn, Germany",
    "Heidelberg, Germany",
    # "Osnabrück, Germany",
    # "Freiburg im Breisgau, Germany",
    "Koblenz, Germany",
    # "Münster, Germany",
    "Göttingen, Germany",
]

std_dict = {}
nodes_dict = {}
G_dict = {}
for query in query_list:
    print(query)
    G = og.osmGraph(query)
    G_dict[query] = G
    sel_nodes, flow_mat = flow_variability(G, gamma=0.04)
    std_dict[query] = np.std(
        flow_mat.T - flow_mat[:, -1], 1
    )  # np.std(np.diff(flow_mat).T, 1)
    nodes_dict[query] = sel_nodes

# %%


fig = plt.figure(figsize=(10, 6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, wspace=0, hspace=0, height_ratios=[1, 1])


axs = [fig.add_subplot(gs[1, j]) for j in range(0, 3)]
labels = [r"\textbf{b}", r"\textbf{c}", r"\textbf{d}"]
for i, ax in enumerate(axs):
    ax.text(
        0.1,
        0.95,
        labels[i],
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=18,
    )
    F = F_list[i]
    selected_nodes = selected_nodes_list[i].set_geometry("geometry")
    edges.plot(ax=ax, column=F, cmap="viridis", linewidth=1)
    if i != 2:
        selected_nodes.plot(
            ax=ax, color="red", marker="x", markersize=50, zorder=3, label="Sources"
        )
    # selected_nodes.plot(ax=ax, color="None", zorder=0, edgecolor="black", linewidth=2)
    # nodes.plot(ax=ax, color="black", markersize=1, zorder=3)
    ax.axis("off")
    if i == 0:
        ax.legend(fontsize=12, loc="upper right")


xyA = [7, 50.99]
xyB = [6.9, 50.98]

for i in range(2):
    arrow = patches.ConnectionPatch(
        xyA,
        xyB,
        coordsA=axs[i].transData,
        coordsB=axs[i + 1].transData,
        # Default shrink parameter is 0 so can be omitted
        color="black",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=20,  # controls arrow head size
        linewidth=2,
        connectionstyle="arc3,rad=0.3",
    )
    fig.patches.append(arrow)
    axs[i].text(xyA[0], xyA[1], "increase\nsources $|W|$", ha="center", va="bottom")


cbar = fig.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=np.array(F_list).max()), cmap="viridis"
    ),
    ax=axs,
    shrink=1 / 2,
    pad=0.01,
)
cbar.ax.set_ylabel(r"Flow $f_{e}$", labelpad=2.5, fontsize=12)


ax = fig.add_subplot(gs[0, :])


tab_cmap = mpl.cm.tab10
colors = [tab_cmap(i) for i in range(10)]
for i, (query, std) in enumerate(std_dict.items()):
    label_query = query.replace(", Germany", "")
    nnodes = len(G_dict[query])
    nedges = G_dict[query].number_of_edges()
    label_query = f"{label_query}\n $|V|={nnodes}, |E|={nedges}$"
    percentage_sources = nodes_dict[query] / len(G_dict[query])
    ax.plot(percentage_sources, std, label=label_query, marker=".", color=colors[i])


ax.text(
    0.01,
    1.05,
    r"\textbf{a}",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=18,
)


ax.grid()
ax.set_xlabel(r"Fraction of nodes as sources $|W|/|V|$")
ax.set_ylabel(rf"Flow disparity $\sigma \left(f_e^{{|V|}} - f_e^{{|W|}} \right)$")
ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=10,
    # bbox_to_anchor=(1, 0.5),
)

# ax.set_yscale("log")

# %%
fig.savefig("figs/flow_commodity_variability.pdf", bbox_inches="tight")
# %%
