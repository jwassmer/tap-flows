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

G.derivative_social_cost(num_sources=5, gamma=0.02, eps=1e-3, solver=cp.MOSEK)
nodes, edges = ox.graph_to_gdfs(G)
edges.sort_values("derivative_social_cost", inplace=True, ascending=False)
edges["derivative_social_cost"]  # .hist()
# %%


primary_key_wds = [
    "primary",
    "trunk",
    "motorway",
    "secondary",
    "tertiary",
    "unclassified",
]
primary = edges[edges["highway"].astype(str).str.contains("|".join(primary_key_wds))]


residential_kwds = ["residential", "living"]
residential = edges[
    edges["highway"].astype(str).str.contains("|".join(residential_kwds))
]

# %%
cmap = plt.get_cmap("Reds")
cmap.set_under("blue")
vmax = max(edges["derivative_social_cost"])
norm = mpl.colors.Normalize(vmin=0, vmax=100)

fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)


primary = primary.sort_values("derivative_social_cost", inplace=False, ascending=False)
residential = residential.sort_values(
    "derivative_social_cost", inplace=False, ascending=False
)
for ax in axs:
    # Plot the primary roads
    residential.plot(
        ax=ax, column="derivative_social_cost", linewidth=2, cmap=cmap, norm=norm
    )
    primary.plot(
        ax=ax, column="derivative_social_cost", cmap=cmap, linewidth=5, norm=norm
    )
    # secondary.plot(ax=ax, column="derivative_social_cost", cmap="cividis", linewidth=3)

    ax.grid()

    primary = primary.sort_values(
        "derivative_social_cost", inplace=False, ascending=True
    )
    residential = residential.sort_values(
        "derivative_social_cost", inplace=False, ascending=True
    )

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(
    sm,
    ax=axs,
    orientation="horizontal",
    label="Derivative Social Cost",
    pad=0.01,
    aspect=50,
    shrink=0.5,
    extend="both",
)
# %%

# Concatenate all data to get common bins
all_data = np.concatenate(
    [
        primary["derivative_social_cost"].values,
        residential["derivative_social_cost"].values,
    ]
)

# Define common bins using numpy.histogram_bin_edges
max_val = 100  # <-- set your desired maximum value here
num_bins = 50  # number of bins

# Create bins manually from 0 to max_val
bins = np.linspace(min(all_data), max_val, num_bins + 1)

# Plot histograms
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(
    residential["derivative_social_cost"],
    bins=bins,
    alpha=1,
    label="Residential Roads",
    density=True,
)

ax.hist(
    primary["derivative_social_cost"],
    bins=bins,
    alpha=0.5,
    label="Primary Roads",
    density=True,
)


ax.grid()
# %%
