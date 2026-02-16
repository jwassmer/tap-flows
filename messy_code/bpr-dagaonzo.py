# %%

from src import osmGraphs as og

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import cvxpy as cp
import networkx as nx

from src import Graphs as gr
from src import Plotting as pl
from src import multiCommodityTAP as mc


# %%

# bpr = lambda x: t_min * (1 + alpha * (x / c) ** beta)
teff = lambda x: (gamma * tr * l * x) / (l * m - gamma * d * x)


G = og.osmGraph("Nippes,Cologne,Germany")
nodes, edges = ox.graph_to_gdfs(G)


# %%


selected_nodes = og.select_evenly_distributed_nodes(nodes, 25)
demands = og.demand_list(nodes, commodity=selected_nodes, gamma=0.01)


f = mc.solve_multicommodity_tap(G, demands, solver=cp.MOSEK)


# %%

edges["flow"] = f
utilization = edges["flow"] / edges["capacity"]
edges["utilization"] = utilization

cmap = plt.get_cmap("cividis")
cmap.set_over("red")
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-3, vmax=1)
fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
edges.plot(ax=ax, column="utilization", cmap=cmap, norm=norm, legend=True)

edges["utilization"].describe()

# %%

edge = list(G.edges(data=True))[10][-1]
l, m, v = edge["length"], edge["lanes"], edge["speed_kph"] / 3.6
gamma, tr, d = 1, 2, 5
walking_speed = 1.4
t_max = l / walking_speed
t_min = l / v


eff_func = og.effective_travel_time(edge)
alpha, beta = og.linear_function(edge)
linear_func = lambda x: alpha * x + beta

# potential_energy_func = potential_energy(edge)


xmax = edge["xmax"]
xmin = edge["xmin"]
# beta = edge["beta"]


# Generate L values
x_values = np.linspace(0, int(np.ceil(xmax * 1.2)), 100)

t_eff_values = [eff_func(x) for x in x_values]
linear_values = [linear_func(x) for x in x_values]


# Plot the original function and the linear function
plt.figure(figsize=(10, 6))
# plt.plot(x_values, bpr_values, label="BPR Function", color="Black", linewidth=2)
plt.plot(
    x_values,
    t_eff_values,
    label="Daganzo model $c_{D, \mathrm{eff}}(f_{e})$",
    color="blue",
)
plt.plot(
    x_values,
    linear_values,
    label=rf"Linear $c(f_e) = {alpha:.2f}[s] f_e + {beta:.2f} [s]$",
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


# %%


G = gr.random_graph(3, seed=4)
rem_edges = [(1, 0), (2, 0), (2, 1), (0, 2)]
G.remove_edges_from(rem_edges)
pl.graphPlot(G)

P = [10, -5, -5]

f = mc.solve_multicommodity_tap(G, [P], solver=cp.MOSEK)
f
# %%

ebc = nx.edge_betweenness_centrality(G, normalized=False)
# %%
