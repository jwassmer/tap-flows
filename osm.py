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


def demand_list(G, commodity, gamma=0.1):
    nodes, edges = ox.graph_to_gdfs(G)
    nodes["source_node"] = False

    demands = []
    tot_pop = 0
    for idx, row in commodity.iterrows():
        pop_com = row["population"]
        vor_geom = row["voronoi"]
        source_node = idx
        target_nodes = nodes[~nodes["geometry"].within(vor_geom)].index

        # random_com_node = np.random.choice(com_nodes)
        nodes.loc[source_node, "source_node"] = True

        P = dict(zip(G.nodes(), np.zeros(G.number_of_nodes())))
        # for node in com_nodes:
        P[source_node] = pop_com * gamma  # / len(com_nodes)
        # P[random_com_node] = pop_com
        for node in target_nodes:
            P[node] = -pop_com * gamma / len(target_nodes)
        # print(sum(P.values()))

        demands.append(list(P.values()))
        # print(c, pop_com)
        tot_pop += pop_com
    # print("Total Population:", tot_pop)
    return demands, nodes


# %%
G, districts = og.osmGraph("Potsdam,Germany", return_districts=True)
selected_nodes = og.select_evenly_distributed_nodes(G, 30)

# %%
demands, nodes = demand_list(
    G,
    commodity=selected_nodes,
)

# %%
F = mc.solve_multicommodity_tap(
    G, demands, social_optimum=False, verbose=True, max_iter=50_000
)
# %%
vmin = 1e2

edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
edges["flow"] = F
nx.set_edge_attributes(G, edges["flow"], "flow")
edges = edges.sort_values(by="flow", ascending=True)

util = edges["flow"] / edges["xmax"]
edges["util"] = util
# edges = edges.sort_values(by="util", ascending=True)

cmap = mpl.colormaps.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = mpl.colors.LogNorm(vmin=vmin, vmax=F.max())

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")
edges.plot(ax=ax, column="flow", cmap=cmap, norm=norm, zorder=1)
nodes[nodes["source_node"]].plot(ax=ax, color="red", zorder=2)

# nodes.plot(column="demand", cmap="coolwarm", ax=ax, zorder=2, legend=True)
districts.plot(ax=ax, color="lightgrey", zorder=0, alpha=0.2)
districts.boundary.plot(ax=ax, color="black", zorder=0, linewidth=0.5)
cbar = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    shrink=1 / 3,
    extend="min",
    pad=-0.03,
    aspect=30,
)
cbar.ax.set_title(r"$f_{i \rightarrow j}$")
print("Social Cost:", total_social_cost(G))
ax.set_title("SC = {:.0f}".format(total_social_cost(G)))

# print(edges.flow.sum())
# fig.savefig("figs/cologne_flow.png", bbox_inches="tight", dpi=300)


# %%

F_list = []
for k in range(3, 150):
    print(k)
    selected_nodes = og.select_evenly_distributed_nodes(G, k)
    demands, nodes = demand_list(
        G,
        commodity=selected_nodes,
    )
    F = mc.solve_multicommodity_tap(
        G, demands, social_optimum=False, max_iter=50_000, eps_rel=1e-5
    )
    F_list.append(F)

# %%
Farr = np.array(F_list)

for i in range(Farr.shape[0]):
    plt.plot(Farr[:, i] / Farr[-1, i])

plt.grid()
plt.legend()


# %%


edge = list(G.edges(data=True))[0][-1]
l, m, v = edge["length"], edge["lanes"], edge["speed_kph"] / 3.6
gamma, tr, d = 1, 2, 5
walking_speed = 1.4
t_max = l / walking_speed
t_min = l / v

eff_func = og.effective_travel_time(edge)
linear_func = og.linear_function(edge)
# potential_energy_func = potential_energy(edge)


xmax = edge["xmax"]
xmin = edge["xmin"]
beta = edge["beta"]


# Generate L values
x_values = np.linspace(0, int(xmax * 1.2), 100)

t_eff_values = [eff_func(x) for x in x_values]
linear_values = [linear_func(x) for x in x_values]


# Plot the original function and the linear function
plt.figure(figsize=(10, 6))
plt.plot(
    x_values,
    t_eff_values,
    label="Daganzo model $c_{D, \mathrm{eff}}(f_{e})$",
    color="blue",
)
plt.plot(
    x_values,
    linear_values,
    label="Linear $c(f_e) = \\alpha_e f_e + \\beta_e$",
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

fig.savefig("figs/daganzo_vs_linear.png", bbox_inches="tight", dpi=300)

# %%


# %%
