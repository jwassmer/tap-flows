# %%
from src import Plotting as pl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

from src import Graphs as gr
from src import TAPOptimization as tap
from src import SocialCost as sc
from matplotlib.transforms import Affine2D


# %%


def braess_graph():
    G = nx.DiGraph()
    G.add_edge(0, 1, alpha=0, beta=5)
    G.add_edge(0, 2, alpha=0.1, beta=0)
    G.add_edge(2, 1, alpha=0, beta=1)
    G.add_edge(1, 3, alpha=0.1, beta=0)
    G.add_edge(2, 3, alpha=0, beta=5)

    pos = {0: (0, 0.5), 1: (1, 0), 2: (1, 1), 3: (2, 0.5)}
    nx.set_node_attributes(G, pos, "pos")
    return G


# %%


G = braess_graph()
# G.relabel_nodes({0: "A", 1: "B", 2: "C", 3: "D"})
G = nx.relabel_nodes(G, {0: "A", 1: "B", 2: "C", 3: "D"})
P = np.zeros(G.number_of_nodes())
load = 40
P[0] = load
P[-1] = -load


betas = np.linspace(0, 5, 100)

social_cost_list = []
for beta in betas:
    G.edges[("C", "B")]["beta"] = beta
    f = tap.user_equilibrium(G, P, positive_constraint=True)
    print(np.round(f, 2))
    social_cost_list.append(sc.total_social_cost(G, f))

G.remove_edge("C", "B")
f = tap.user_equilibrium(G, P, positive_constraint=True)
r_sc = sc.total_social_cost(G, f)
G.add_edge("C", "B", beta=5, alpha=0)


nodecolors = {"A": "red", "B": "grey", "C": "grey", "D": "lightblue"}

# %%

fig, axs = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 2])

axs[0].plot(betas, social_cost_list, color="black", linewidth=5)
axs[0].tick_params(axis="both", which="major", labelsize=18)
axs[0].set_xlabel(r"Free flow travel time $\beta_{CB}$", fontsize=22)
axs[0].set_ylabel(r"Total social cost sc$(f_e)$", fontsize=22)
axs[0].grid()
axs[0].scatter(
    betas[-1],
    r_sc,
    color="red",
    label="Deleted\n edge $(C, B)$",
    marker="x",
    zorder=3,
    s=250,
)
leg = axs[0].legend(fontsize=22, loc="upper right")
leg.set_bbox_to_anchor((1.3, 1.1))  # optional: reposition if needed


axs[0].text(
    0.05,
    1.03,
    r"\textbf{a}",
    ha="center",
    va="center",
    transform=axs[0].transAxes,
    fontsize=30,
)


nx.draw(
    G,
    nx.get_node_attributes(G, "pos"),
    node_size=1000,
    width=2,
    with_labels=True,
    node_color=nodecolors.values(),
    font_weight="bold",
    arrowsize=18,
    ax=axs[1],
    font_size=22,
)

edge_labels = {}
for e in G.edges:
    alpha = G.edges[e]["alpha"]
    beta = G.edges[e]["beta"]
    l = "".join(e)

    expr = ""

    # Handle alpha term
    if alpha != 0:
        if alpha == 1:
            expr += f"f_{{{l}}}"
        elif alpha == -1:
            expr += f"-f_{{{l}}}"
        else:
            expr += f"{alpha}f_{{{l}}}"

    # Handle beta term
    if beta != 0:
        sign = " + " if beta > 0 and expr else ""
        if beta < 0:
            sign = " - " if expr else "-"
        expr += f"{sign}{abs(beta) if abs(beta) != 1 or expr else ''}"

    # If both are zero
    if not expr:
        expr = "0"

    edge_labels[e] = rf"$t_{{{l}}}(f_{{{l}}})={expr}$"


edge_labels[("C", "B")] = r"$t_{CB}(f_{CB})=\beta_{CB}$"


nx.draw_networkx_edge_labels(
    G,
    pos=nx.get_node_attributes(G, "pos"),
    edge_labels=edge_labels,
    font_size=22,
    font_color="black",
    ax=axs[1],
)


axs[1].text(
    0.1,
    1.0,
    r"\textbf{b}",
    ha="center",
    va="center",
    transform=axs[1].transAxes,
    fontsize=32,
)
fig.savefig("figs/braess_social_cost.pdf", dpi=300, bbox_inches="tight")

# %%
