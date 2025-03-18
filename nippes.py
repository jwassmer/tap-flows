# %%
from src import osmGraphs as og
from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc
from src import Plotting as pl

import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import networkx as nx


def demand_list(nodes, commodity, gamma=0.1):
    # nodes, edges = ox.graph_to_gdfs(G)
    nodes["source_node"] = False

    demands = []
    tot_pop = 0
    for idx, row in commodity.iterrows():
        source_node = nodes.loc[idx].name
        target_nodes = nodes.loc[~nodes.index.isin([idx])].index
        pop_com = row["population"]

        P = dict(zip(nodes.index, np.zeros(len(nodes))))
        # for node in com_nodes:
        P[source_node] = pop_com * gamma

        # P[random_com_node] = pop_com
        for node in target_nodes:
            P[node] = -pop_com * gamma / len(target_nodes)
        # print(sum(P.values()))

        demands.append(list(P.values()))
        # print(c, pop_com)
        tot_pop += pop_com
    # print("Total Population:", tot_pop)
    return demands


# %%s

G, boundary = og.osmGraph(
    "Nippes, Cologne, Germany",
    return_boundary=True,
    heavy_boundary=True,
    buffer_meter=10_000,
)
# %%

nodes, edges = ox.graph_to_gdfs(G)
selected_nodes = og.select_evenly_distributed_nodes(nodes, 3)

demands = demand_list(nodes, commodity=selected_nodes, gamma=1e-1)
od_matrix = np.array(demands)
# %%

f_mat, lambda_mat = mc.solve_multicommodity_tap(G, od_matrix, return_fw=True)
F = np.sum(f_mat, 0)
Fdict = dict(zip(G.edges, F))

# %%

M, x, y = mcsc.test_system(G, f_mat, lambda_mat, od_matrix, eps=1e-1)

# print(np.max(np.abs(np.linalg.pinv(M) @ y - x)))

# %%
beta_arr = np.array(list(nx.get_edge_attributes(G, "length").values()))
beta_vec = np.array([beta_arr for _ in range(len(od_matrix))]).flatten()


# %%

fig, ax = plt.subplots()
edges.plot(ax=ax, column=F)
selected_nodes.plot(ax=ax, color="red", marker="x", zorder=2)

# %%

derivatives = mcsc.derivative_social_cost(G, f_mat, od_matrix, eps=5e-1)

edges["d_social_cost"] = [derivatives.get(edge, 0) for edge in G.edges]
edges["flow"] = F

plt.scatter(edges["d_social_cost"], edges["flow"])
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.ylim(1e1, 1e5)
derivatives


# %%
delta = edges["flow"] - edges["d_social_cost"]
cmap = mpl.cm.coolwarm
norm = mpl.colors.TwoSlopeNorm(vmin=delta.min(), vmax=delta.max(), vcenter=0)

fig, ax = plt.subplots(figsize=(8, 6))
edges.plot(ax=ax, column=delta, cmap=cmap, norm=norm)


# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
edges.plot(column="d_social_cost", ax=axs[0])
edges.plot(column=F, ax=axs[1])

# %%

edge = (660192262, 215283089, 0)
print(Fdict[edge])
slope, beta, sc = mcsc.numerical_derivative(G, od_matrix, edge, var_percentage=0.05)
print(slope)
print(derivatives[edge])
plt.plot(beta, sc)
plt.grid()
# %%
