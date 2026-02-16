# %%
import osmnx as ox
import cvxpy as cp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src import multiCommodityTAP as mc
from src import Plotting as pl
from src import osmGraphs as og
from src import TAPOptimization as tap
from src import SocialCost as sc

# %%
G, boundary = og.osmGraph(
    "Niehl,Cologne,Germany",
    return_boundary=True,
    heavy_boundary=True,
    buffer_meter=10_000,
)

# Save graph as pickles
# with open("data/nippes_graph.pkl", "wb") as f:
#    pickle.dump(G, f)

nodes, edges = ox.graph_to_gdfs(G)
selected_nodes = og.select_evenly_distributed_nodes(nodes, 20)


demands = og.demand_list(nodes, commodity=selected_nodes, gamma=1e-1)
# %%

F = mc.solve_multicommodity_tap(
    G,
    demands,
    verbose=True,
    # solver=cp.MOSEK,
)

# %%
num_selected_nodes = np.linspace(3, 25, 10, dtype=int)

F_list = []

for i in num_selected_nodes:
    selected_nodes = og.select_evenly_distributed_nodes(nodes, i)
    demands = og.demand_list(nodes, commodity=selected_nodes, gamma=1e-1)
    F = mc.solve_multicommodity_tap(
        G,
        demands,
    )

    F_list.append(F)

# %%
F_mat = np.array(F_list)

plt.plot(num_selected_nodes, F_mat[:, 0:10])

plt.grid()
plt.yscale("log")

# %%

# %%
