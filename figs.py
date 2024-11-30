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


fig, ax = plt.subplots(figsize=(4, 3))

alpha = 2 / 3
x = np.linspace(0, 20, 100)
y = 5 / (1 + np.exp(-(alpha * (x - 10)))) + 10

ax.plot(
    x,
    y,
    # label=r"$\frac{1}{1 + e^{-\alpha(x - 10)}}$",
    color="black",
    linewidth=2,
)
# ax.legend()
ax.grid()
ax.set_xlabel(r"Vehicle flow $f_{e}$")
ax.set_ylabel(r"Cost function $t_{e}(f_{e})$")

fig.savefig("figs/sigmoid.png", dpi=300, bbox_inches="tight")

# %%
fig, ax = plt.subplots(figsize=(4, 3))

alpha = 1 / 4
beta = 10
x = np.linspace(0, 20, 100)
y = alpha * x + beta
ax.plot(
    x,
    y,
    # label=r"$\frac{1}{1 + e^{-\alpha(x - 10)}}$",
    color="black",
    linewidth=2,
)
# ax.legend()
ax.grid()
ax.set_xlabel(r"Vehicle flow $f_{e}$")
ax.set_ylabel(r"Cost function $t_{e}(f_{e})$")

fig.savefig("figs/linear.png", dpi=300, bbox_inches="tight")

# %%


def braess_graph():
    G = nx.DiGraph()
    G.add_edge(0, 1, alpha=0, beta=5)
    G.add_edge(0, 2, alpha=1, beta=0)
    G.add_edge(2, 1, alpha=0, beta=1)
    G.add_edge(1, 3, alpha=1, beta=0)
    G.add_edge(2, 3, alpha=0, beta=5)

    pos = {0: (0, 0.5), 1: (1, 0), 2: (1, 1), 3: (2, 0.5)}
    nx.set_node_attributes(G, pos, "pos")
    return G


# %%

G = braess_graph()
P = np.zeros(G.number_of_nodes())
load = 100
P[0] = load
P[-1] = -load

f = tap.user_equilibrium(G, P, positive_constraint=True)
f_dict = dict(zip(G.edges, np.round(f, 2)))

fig, ax = plt.subplots(figsize=(4, 2))
cmap = mpl.cm.cividis
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=max(f))
nx.draw_networkx_edges(
    G,
    nx.get_node_attributes(G, "pos"),
    ax=ax,
    width=2,
    edge_color=f,
    edge_cmap=cmap,
    # edge_vmin=1e-1,
)
nx.draw_networkx_labels(G, nx.get_node_attributes(G, "pos"), ax=ax)

nodecolors = {0: "red", 1: "lightgrey", 2: "lightgrey", 3: "lightblue"}
nx.draw_networkx_nodes(
    G, nx.get_node_attributes(G, "pos"), ax=ax, node_color=nodecolors.values()
)
ax.axis("off")
fig.savefig("figs/braess.png", dpi=300, bbox_inches="tight")
# %%

total_travel_time = np.sum(
    [G.edges[e]["alpha"] * f_dict[e] + G.edges[e]["beta"] for e in G.edges]
)
print(total_travel_time)
# %%
# [G.edges[e]["alpha"] * f_dict[e] + G.edges[e]["beta"] for e in G.edges]
# %%

pos = nx.get_node_attributes(G, "pos")
nx.set_edge_attributes(G, f_dict, "flow")


# Draw the graph
fig = plt.figure(figsize=(4, 2))
nx.draw(
    G,
    pos,
    with_labels=True,
    # node_size=1000,
    node_color=nodecolors.values(),
    font_weight="bold",
    arrowsize=15,
)
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(u, v): d["flow"] for u, v, d in G.edges(data=True)},
    font_color="red",
)
fig.savefig("figs/braess_flow.png", dpi=300, bbox_inches="tight")


# %%

load = 100
G = gr.random_graph(20, 0.2, beta="random", alpha="random")
P = np.zeros(G.number_of_nodes())
P[16] = load
P[15] = -load

fue = tap.user_equilibrium(G, P, positive_constraint=True)
fso = tap.social_optimum(G, P, positive_constraint=True)


delta = fue - fso
cmap = mpl.cm.coolwarm
norm = mpl.colors.TwoSlopeNorm(vmin=np.min(delta), vcenter=0, vmax=np.max(delta))
fig, ax = plt.subplots(figsize=(8, 6))
nc = {
    n: "red" if p > 0 else "lightblue" if p < 0 else "lightgrey"
    for n, p in zip(G.nodes, P)
}
pl.graphPlot(
    G,
    ec=delta,
    cmap=cmap,
    norm=norm,
    show_labels=True,
    ax=ax,
    nc=nc,
    cbar=False,
    title="Social optimum - User equilibrium",
)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    orientation="vertical",
    fraction=0.01,
    pad=-0.1,
    extend="both",
)
cb.set_label(r"$\sum_e f^{ue}_e - f^{so}_e$")
fig.savefig("figs/social_optimum.png", dpi=300, bbox_inches="tight")
# %%

sc_ue = sc.total_social_cost(G, fue)
sc_so = sc.total_social_cost(G, fso)
# %%


sc_ue - sc_so
# %%
# Re-plotting without using the legend function to avoid the error
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Define the objective function
def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 1) ** 2


# Define the constraint function
constraint = {"type": "ineq", "fun": lambda x: 3 - (x[0] + 2 * x[1])}

# Initial guess for the variables
x0 = np.array([0.0, 0.0])

# Perform the minimization
result = minimize(objective, x0, constraints=[constraint])

# Print the results
print("Optimal solution:", result.x)
print("Objective value at optimum:", result.fun)

# Plotting the objective function
x1 = np.linspace(-1, 4, 400)
x2 = np.linspace(-1, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = objective([X1, X2])

plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=20, cmap="viridis")
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel("x1")
plt.ylabel("x2")

# Plot the constraint line
x2_constraint = (3 - x1) / 2
plt.plot(x1, x2_constraint, "r-", label=r"$x_1 + 2x_2 \leq 3$")

# Shade the feasible region
plt.fill_between(
    x1, x2_constraint, 4, where=(x2_constraint <= 4), color="lightgrey", alpha=0.5
)

# Mark the optimal solution
plt.plot(
    result.x[0],
    result.x[1],
    "x",
    label="Optimal solution",
    markersize=10,
    color="black",
)

# Add legend and show the plot
plt.legend()
plt.title("Objective Function and Constraint")
plt.grid(True)
plt.show()


# %%


G = braess_graph()
P = np.zeros(G.number_of_nodes())
load = 4
P[0] = load
P[-1] = -load


betas = np.linspace(0, 5, 100)

social_cost_list = []
for beta in betas:
    G.edges[(2, 1)]["beta"] = beta
    f = tap.user_equilibrium(G, P, positive_constraint=True)
    print(np.round(f, 2))
    social_cost_list.append(sc.total_social_cost(G, f))

G.remove_edge(2, 1)
f = tap.user_equilibrium(G, P, positive_constraint=True)
r_sc = sc.total_social_cost(G, f)
G.add_edge(2, 1, beta=5)


# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(betas, social_cost_list, color="black", linewidth=3)
axs[0].set_xlabel(r"edge length $\beta_{21}$", fontsize=22)
axs[0].set_ylabel("Social cost", fontsize=22)
axs[0].grid()
axs[0].scatter(
    betas[-1],
    r_sc,
    color="red",
    label="Removed edge (2, 1)",
    marker="x",
    zorder=3,
    s=100,
)
axs[0].legend(fontsize=18, loc="upper right")


nx.draw(
    G,
    nx.get_node_attributes(G, "pos"),
    node_size=750,
    width=2,
    with_labels=True,
    node_color=nodecolors.values(),
    font_weight="bold",
    arrowsize=18,
    ax=axs[1],
    font_size=18,
)


axs[1].annotate(
    r"$\beta_{21}$",
    xy=(0.57, 0.5),
    xycoords="axes fraction",
    fontsize=22,
    ha="center",
    va="center",
)


axs[1].annotate(
    r"x",
    xy=(0.5, 0.5),
    xycoords="axes fraction",
    fontsize=40,
    ha="center",
    va="center",
    color="red",
    zorder=3,
)

fig.savefig("figs/braess_social_cost.pdf", dpi=300, bbox_inches="tight")
# %%
