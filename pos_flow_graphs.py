# %%
import random
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

from src import Graphs as gr
import geopandas as gpd
from shapely.geometry import Point

# %%
G = gr.random_planar_graph(5, seed=1)
nodes, edges = ox.graph_to_gdfs(G)

source = 0
sink = len(G) - 1

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
edges.plot(ax=ax)

# Plot the nodes with labels as markers
nodes.plot(ax=ax, marker="o", color="white", markersize=1000, edgecolor="black")
for x, y, label in zip(nodes.geometry.x, nodes.geometry.y, nodes.index):
    ax.text(x, y, label, fontsize=24, ha="center", va="center")

# %%


# shortest_path = nx.shortest_path(G, source, sink, weight="beta")
all_paths = list(nx.all_simple_paths(G, source, sink))
# Create a new graph to store the paths
H = nx.DiGraph()

# Add nodes and edges from all_paths to the new graph H
for path in all_paths:
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not H.has_edge(u, v):
            H.add_edge(u, v, beta=G[u][v][0]["beta"])


nx.draw(H, with_labels=True)
# %%
nx.is_directed_acyclic_graph(H)
# %%
nx.find_cycle(H)
# %%
