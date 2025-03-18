# %%
from src import multiCommoditySocialCost as mcsc
from src import osmGraphs as og
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import cvxpy as cp

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# %%
location = "Nippes,Cologne,Germany"

G, bounds = og.osmGraph(
    location,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|residential|living"]',
    return_boundary=True,
    # heavy_boundary=True,
    # buffer_meter=2_500,
)
G.flows(num_sources=25, gamma=0.04, solver=cp.MOSEK)
nodes, edges = ox.graph_to_gdfs(G)


# %%

edges.explore(column="flow")

# %%
G.derivative_social_cost(num_sources=5, gamma=0.02, eps=1e-3, solver=cp.MOSEK)

nodes, edges = ox.graph_to_gdfs(G)


# %%
vmin, vmax = min(edges["derivative_social_cost"]), max(edges["derivative_social_cost"])
cmap = plt.get_cmap("cividis")
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
edges.sort_values("derivative_social_cost", inplace=True, ascending=False)


def get_color(value):
    if value <= -1e-3:
        return "#FF0000"  # Red
    else:
        rgba = cmap(norm(value))
        return mpl.colors.to_hex(rgba)


edges["color"] = edges["derivative_social_cost"].apply(get_color)

# Now explore using the color column instead of the value
edges.explore(color=edges["color"])


# %%

G.derivative_social_cost(num_sources=25, gamma=0.02, eps=1e-3, solver=cp.MOSEK)
nodes, edges = ox.graph_to_gdfs(G)
edges.sort_values("derivative_social_cost", inplace=True, ascending=False)
edges["derivative_social_cost"]  # .hist()
# %%

primary_key_wds = ["motorway", "trunk", "primary, primary_link"]
primary = edges[edges["highway"].astype(str).str.contains("|".join(primary_key_wds))]

secondary_key_wds = [
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
]
secondary = edges[
    edges["highway"].astype(str).str.contains("|".join(secondary_key_wds))
]

residential = edges[edges["highway"].astype(str).str.contains("residential")]


# %%
