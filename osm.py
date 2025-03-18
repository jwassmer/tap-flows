# %%
import osmnx as ox
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import quad
import cvxpy as cp
import time


from src import multiCommodityTAP as mc
from src import Plotting as pl
from src import osmGraphs as og
from src import TAPOptimization as tap
from src import SocialCost as sc


def source_sink_vector(G):
    sources = G.source_nodes
    targets = G.target_nodes
    total_load = G.total_load

    node_dict = dict(zip(G.nodes, range(G.number_of_nodes())))

    sources = [node_dict[s] for s in sources]
    targets = [node_dict[t] for t in targets]

    P = np.zeros(G.number_of_nodes())
    P[sources] = total_load / len(sources)
    P[targets] = -total_load / len(targets)

    return P


def lin_potential_energy(edge):
    alpha = edge["alpha"]
    beta = edge["beta"]

    xmax = edge["xmax"]
    xmin = edge["xmin"]
    tmax = edge["tmax"]
    tmin = edge["tmin"]

    f = lambda x: 1 / 2 * alpha * x**2 + beta * x
    # f_cond = lambda x: (x * tmax if x > xmax else x * tmin if x < xmin else f(x))

    return f


def total_social_cost(G, kwd="flow"):
    return sum(
        [
            G.edges[e][kwd] * G.edges[e]["tt_function"](G.edges[e][kwd])
            for e in G.edges(keys=True)
        ]
    )


def total_potential_energy(G):
    return sum(
        [G.edges[e]["tt_function"](G.edges[e]["flow"]) for e in G.edges(keys=True)]
    )


# %%
G, boundary = og.osmGraph(
    "Vienna,Austria", return_boundary=True, heavy_boundary=True, buffer_meter=10_000
)

# Save graph as pickles
# with open("data/nippes_graph.pkl", "wb") as f:
#    pickle.dump(G, f)

nodes, edges = ox.graph_to_gdfs(G)
selected_nodes = og.select_evenly_distributed_nodes(nodes, 10)

# Read graph from pickle
# with open("data/nippes_graph.pkl", "rb") as f:
#    G = pickle.load(f)

# %%
demands = og.demand_list(nodes, commodity=selected_nodes, gamma=1e-1)
P = demands[0]
print(max(P))

f0 = tap.optimize_tap(G, P, with_capacity=False, positive_constraint=False)

# %%
F = mc.solve_multicommodity_tap(
    G, demands, social_optimum=False, verbose=True, max_iter=20_000, pos_flows=False
)

# %%

Fso = mc.solve_multicommodity_tap(
    G,
    demands,
    social_optimum=True,
    verbose=True,
)

# %%


edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges["flow"] = F - Fso
absmax = 20  # edges["flow"].abs().max()
edges = edges.sort_values(by="flow", ascending=True)

nx.set_edge_attributes(G, edges["flow"], "flow")
# edges = edges.sort_values(by="flow", ascending=True)

cmap = mpl.colormaps.get_cmap("coolwarm")
# cmap.set_under("lightgrey")
norm = mpl.colors.TwoSlopeNorm(vmin=-absmax, vmax=absmax, vcenter=0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")
edges.plot(ax=ax, column="flow", cmap=cmap, norm=norm, zorder=1, linewidth=2)
nodes[nodes["source_node"]].plot(
    ax=ax, color="black", zorder=2, label="Source nodes", marker="x"
)
boundary.boundary.plot(ax=ax, color="black", zorder=3, linewidth=1)
# nodes.plot(column="demand", cmap="coolwarm", ax=ax, zorder=2, legend=True)
# districts.plot(ax=ax, color="lightgrey", zorder=0, alpha=0.2)
# districts.boundary.plot(ax=ax, color="black", zorder=0, linewidth=0.5)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=1 / 3,
    extend="both",
    pad=-0.03,
    aspect=30,
)
ax.legend()
cbar.ax.set_ylabel(r"$f^{ue}_{e}-f^{so}_{e}$")

# print(edges.flow.sum())
fig.savefig("figs/vienna_flow.pdf", bbox_inches="tight", dpi=300)


# %%s
sc.total_social_cost(G, F) - sc.total_social_cost(G, Fso)

# %%


edge = list(G.edges(data=True))[-1][-1]
l, m, v = edge["length"], edge["lanes"], edge["speed_kph"] / 3.6
gamma, tr, d = 1, 2, 5
walking_speed = 1.4
t_max = l / walking_speed
t_min = l / v

eff_func = og.effective_travel_time(edge)
alpha, beta = og.linear_function(edge)

alpha = round(alpha, 2)
beta = round(beta, 2)

# potential_energy_func = potential_energy(edge)


xmax = edge["xmax"]
xmin = edge["xmin"]
# beta = edge["beta"]


# Generate L values
x_values = np.linspace(0, int(np.ceil(xmax * 1.2)), 100)

t_eff_values = [eff_func(x) for x in x_values]
linear_values = [alpha * x + beta for x in x_values]


# Plot the original function and the linear function
fig = plt.figure(figsize=(10, 6))
plt.plot(
    x_values,
    t_eff_values,
    label="Daganzo model $c_{D, \mathrm{eff}}(f_{e})$",
    color="blue",
)
plt.plot(
    x_values,
    linear_values,
    label=rf"Linear $c(f_e) = {alpha}[s] f_e + {beta} [s]$",
    color="red",
    linestyle="--",
)

plt.axhline(y=t_max, color="green", linestyle="-.", label="$t_{\\mathrm{max}}$")
plt.axhline(y=t_min, color="orange", linestyle="-.", label="$t_{\\mathrm{min}}$")
plt.axvline(x=xmax, color="green", linestyle="-.")
plt.axvline(x=xmin, color="orange", linestyle="-.")
plt.xlabel("$f_{e}$")
plt.ylabel("$c_{D, \mathrm{eff}}(f_{e}) [s]$")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

fig.savefig("figs/daganzo_vs_linear.pdf", bbox_inches="tight", dpi=300)

# %%
