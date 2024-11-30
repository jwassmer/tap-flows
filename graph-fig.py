# %%

from src import Plotting as pl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

from src import Graphs as gr
from src import TAPOptimization as tap
from src import SocialCost as sc

# %%

seed = 5
G = gr.random_graph(10, num_edges=19, alpha="random", beta="random", seed=seed)
edges = list(G.to_undirected().edges)
for e in edges:
    if np.random.rand() < 0.95:
        G.remove_edge(*e)

# %%
ax_labels = [r"\textbf{a}", r"\textbf{b}"]

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
pl.graphPlot(
    G.to_undirected(), ax=axs[0], ec=np.ones(len(G.edges)), cbar=False, title=""
)
pl.graphPlot(G, title="", ec=np.ones(len(G.edges)), cbar=False, ax=axs[1])

for i, ax in enumerate(axs):
    ax.text(
        0.1,
        0.95,
        ax_labels[i],
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
    )
fig.savefig("figs/two-graphs.pdf", bbox_inches="tight")

# %%

G = gr.random_graph(2, num_edges=1)
G.remove_edge(1, 0)
G = nx.MultiDiGraph(G)
G.add_edge(0, 1, key=3)
pos = {0: (0, 0), 1: (1, 0)}

cmap = plt.get_cmap("cividis")
norm = mpl.colors.Normalize(vmin=1, vmax=1)
flows = dict(zip(G.edges, np.ones(G.number_of_edges())))
edge_colors = {e: cmap(norm(flows[e])) for e in G.edges}

# %%


fig, axs = plt.subplots(1, 2, figsize=(8, 2))
ax = axs[0]
ax.axis("off")
G.add_node(2)
G.add_edge(0, 2, key=0)
pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.5)}
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgrey", node_size=0)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="white", width=0)
G.remove_node(2)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color="lightgrey")
nx.draw_networkx_labels(G, pos, ax=ax)
edges = G.edges
for u, v, k in edges:
    # Draw with curvature if bidirectional
    connection_style = f"arc3,rad={-0.2*k*1.5}"
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=[(u, v)],
        connectionstyle=connection_style,
        edge_color=edge_colors[(u, v, k)],
        width=2,
    )

ax = axs[1]
ax.axis("off")

H = gr.random_graph(3, num_edges=3)
H.remove_edge(1, 0)
H.remove_edge(2, 0)
H.remove_edge(1, 2)
pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 0.5)}

cmap = plt.get_cmap("cividis")
norm = mpl.colors.Normalize(vmin=1, vmax=1)
flows = dict(zip(H.edges, np.ones(H.number_of_edges())))
edge_colors2 = {e: cmap(norm(flows[e])) for e in H.edges}

nx.draw_networkx_nodes(H, pos, ax=ax, node_color="lightgrey")
nx.draw_networkx_labels(H, pos, ax=ax)
edges = H.edges
for u, v in edges:
    # Draw with curvature if bidirectional
    if (u, v) == (0, 1):
        k = 0
    else:
        k = 1
    connection_style = f"arc3,rad={-0.2*k}"
    nx.draw_networkx_edges(
        H,
        pos,
        ax=ax,
        edgelist=[(u, v)],
        connectionstyle=connection_style,
        edge_color=edge_colors2[(u, v)],
        width=2,
    )

for i, ax in enumerate(axs):
    ax.text(
        0.15,
        0.95,
        ax_labels[i],
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
    )

fig.savefig("figs/multi-graphs-2.pdf", bbox_inches="tight")
# %%


# %%
