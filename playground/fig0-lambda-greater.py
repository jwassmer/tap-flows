# %%
from src import Plotting as pl
from src import Graphs as gr
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import matplotlib as mpl

mpl.rcParams["text.latex.preamble"] = r"\usepackage{stmaryrd}"

# %%


# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_node("n")
G.add_node("m")

G.add_edge("n", "m", color="black", style="-", label=r"$\beta_{nm}$")
G.add_edge("m", "n", color="red", style="--", label=r"$\beta_{mn}$")

# Define positions
pos = {"n": (0, 0), "m": (1, 0)}

# Create figure
fig, ax = plt.subplots(figsize=(6, 3))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=600,
    node_color="lightgrey",
    edgecolors="black",
    ax=ax,
)
nx.draw_networkx_labels(
    G,
    pos,
    font_size=22,
    font_family="sans-serif",
    font_weight="bold",
    verticalalignment="center",
    ax=ax,
)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, "label")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=22,
    connectionstyle="arc3,rad=0.4",
    font_color="black",
    ax=ax,
    # label_pos=0.7,
)


nx.draw_networkx_edges(
    G,
    pos,
    connectionstyle="arc3,rad=0.27",
    ax=ax,
    arrows=True,
    edge_color="white",
    alpha=0,
)


for u, v, d in G.edges(data=True):
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v)],
        connectionstyle="arc3,rad=0.27",
        arrows=True,
        arrowsize=40,
        edge_color=d["color"],
        style=d["style"],
        arrowstyle="->",
        ax=ax,
    )


ax.text(0, -0.15, r"$\lambda_n$", fontsize=22, ha="center")
ax.text(1.02, -0.15, r"$\lambda_n+\beta_{nm}$", fontsize=22, ha="center")
ax.text(0.0, 0.15, r"$\lambda_n+\beta_{nm}+\beta_{mn}$", fontsize=22, ha="center")
ax.text(0.5, 0.3, r"$\lightning$", fontsize=150, ha="center", va="center")

ax.axis("off")

fig.savefig(
    "figs/fig0-lambda-greater.pdf",
    bbox_inches="tight",
    # pad_inches=0.1,
    dpi=300,
)


# %%


# %%
