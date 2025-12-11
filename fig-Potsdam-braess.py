# %%
from src import Plotting as pl
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import multiCommoditySocialCost as mcsc
from src import SocialCost as sc
from src import TAPOptimization as tap
from src import osmGraphs as og
from shapely.ops import unary_union
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import cvxpy as cp
import matplotlib as mpl
import os
import geopandas as gpd
import networkx as nx

import folium


import numpy as np
import pandas as pd


def demand_list(
    nodes,
    destination_nodes,
    gamma=0.05,
    demand_col="population",
):
    """
    Generate OD demand list from nodes to given destination nodes.

    Parameters
    ----------
    nodes : GeoDataFrame or DataFrame
        Node table with at least `id_col` and `demand_col` columns.
    commodity : list or array
        IDs (or indices) of destination nodes.
    gamma : float
        Fraction of each node's population that generates demand.
    demand_col : str
        Column name for demand size (e.g. population).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    od_list : DataFrame
        Columns: ['origin', 'destination', 'demand']
    """

    origins = np.array(nodes.index)
    destinations = np.array(destination_nodes.index)
    od_mat = pd.DataFrame(0, index=origins, columns=destinations, dtype=float)

    total_pop = destination_nodes[demand_col].sum()
    destination_attractiveness = destination_nodes[demand_col] / total_pop

    for d in destinations:

        for o in origins:
            pop = nodes.loc[o, demand_col]
            demand_total = pop * gamma
            od_mat.loc[o, d] = demand_total * destination_attractiveness[d]

        od_mat.loc[d, d] = od_mat.loc[d, d] - od_mat[d].sum()

    return np.array(od_mat).T


# %%
place = "Potsdam, Germany"
G, boundary = og.osmGraph(
    place,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]',
    return_boundary=True,
    tolerance_meters=100,
)
nodes, edges = ox.graph_to_gdfs(G)

# %%
num_destinations = len(nodes)
destination_nodes = og.select_evenly_distributed_nodes(nodes, num_destinations)
od_mat = demand_list(
    nodes,
    destination_nodes,
    gamma=0.03,
    demand_col="population",
)

# %%


f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_mat, return_fw=True, pos_flows=True, solver=cp.OSQP, verbose=False
)
F = np.sum(f_mat, axis=0)


edges["flow"] = F
dsc = mcsc.derivative_social_cost(G, f_mat, od_mat, eps=1e-3, demands_to_sinks=False)
dsc_vec = list(dsc.values())
edges["derivative_social_cost"] = dsc_vec


edges["loaded_beta"] = edges["alpha"] * edges["flow"] + edges["beta"]
edges["load"] = 1 - edges["beta"] / edges["loaded_beta"]
# %%

fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)

for ax in axs:
    destination_nodes.plot(
        ax=ax,
        marker=".",
        zorder=4,
        color="black",
        markersize=25,
        label="Sources \& Destinations",
    )
ax = axs[0]
boundary.boundary.plot(linewidth=1, ax=ax, color="black", zorder=3)
ax.grid()

ax.text(
    0.02,
    1.02,
    r"$\textbf{a}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=24,
)
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=0.2, vmax=max(edges["load"]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cb = fig.colorbar(
    sm, ax=ax, orientation="horizontal", pad=0.01, shrink=2 / 3, extend="min"
)


sorted_edges = edges.sort_values("load", ascending=True, inplace=False)
sorted_edges.plot(ax=ax, linewidth=2, column="load", cmap=cmap, norm=norm, legend=False)


cb.ax.set_xlabel(r"Utilization $1-\beta_e/(\alpha_e f_e + \beta_e$)")

ax = axs[1]
boundary.boundary.plot(linewidth=1, ax=ax, color="black", zorder=3)
ax.grid()

ax.text(
    0.02,
    1.02,
    r"$\textbf{b}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=24,
)

cmap = plt.get_cmap("Reds")
cmap.set_under("#0571b0")
norm = mpl.colors.Normalize(vmin=0, vmax=max(dsc_vec))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

sorted_edges = edges.sort_values(
    "derivative_social_cost", ascending=True, inplace=False
)
sorted_edges.plot(
    ax=ax,
    linewidth=2,
    column="derivative_social_cost",
    cmap=cmap,
    norm=norm,
)

braess_edges = edges[edges["derivative_social_cost"] < 0]
if len(braess_edges) > 0:
    braess_edges.plot(
        ax=ax,
        linewidth=2,
        color="#0571b0",
        label="Braess edges",
    )

axs[1].legend(
    loc="upper right",
)

cb = fig.colorbar(
    sm,
    ax=ax,
    orientation="horizontal",
    pad=0.01,
    shrink=2 / 3,
    extend="min",
)
cb.ax.set_xlabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$")

axs[0].set_xlabel("Lon [°]")
axs[1].set_xlabel("Lon [°]")
axs[0].set_ylabel("Lat [°]")


fig.savefig(f"figs/{place.strip()}-Braess.pdf", dpi=300, bbox_inches="tight")
edges.to_file(f"figs/{place.strip()}-Braess.geojson", driver="GeoJSON")

# %%


# create interactive map
center_y, center_x = nodes.geometry.y.mean(), nodes.geometry.x.mean()
m = folium.Map(location=[center_y, center_x], zoom_start=12)  # e.g. Berlin

# Add the first GeoDataFrame (edges)
edges.explore(m=m, column="derivative_social_cost", cmap="Reds", min_zoom=12, vmin=0)

# Add the second GeoDataFrame (braess_edges)
braess_edges.explore(m=m, color="#0571b0")  # or another distinguishing style
# %%
# Re-analyze treating the uploaded Potsdam file as containing ALL edges, not just Braess edges.


edges_potsdam = edges

# Ensure CRS
if edges_potsdam.crs is None:
    edges_potsdam.set_crs(epsg=4326, inplace=True)

# Project for accurate length
edges_potsdam_utm = edges_potsdam.to_crs(epsg=32633)
edges_potsdam_utm["length_m"] = edges_potsdam_utm.geometry.length

scgc_col = "derivative_social_cost"

# Basic stats
total_edges = len(edges_potsdam_utm)
total_length_km = edges_potsdam_utm["length_m"].sum() / 1000

min_scgc = edges_potsdam_utm[scgc_col].min()
max_scgc = edges_potsdam_utm[scgc_col].max()
mean_scgc = edges_potsdam_utm[scgc_col].mean()
median_scgc = edges_potsdam_utm[scgc_col].median()

# Identify Braess edges (negative SCGC)
braess_edges = edges_potsdam_utm[edges_potsdam_utm[scgc_col] < 0]
n_braess = len(braess_edges)
total_braess_length_km = braess_edges["length_m"].sum() / 1000
median_braess_length = braess_edges["length_m"].median()

# Top 5 Braess edges
top_neg = braess_edges.sort_values(scgc_col).head(5)[
    ["name", "highway", "maxspeed", "lanes", scgc_col, "length_m"]
]

# Top 5 positive edges
top_pos = edges_potsdam_utm.sort_values(scgc_col, ascending=False).head(5)[
    ["name", "highway", "maxspeed", "lanes", scgc_col, "length_m"]
]

stats = {
    "total_edges": total_edges,
    "total_length_km": total_length_km,
    "min_scgc": float(min_scgc),
    "max_scgc": float(max_scgc),
    "mean_scgc": float(mean_scgc),
    "median_scgc": float(median_scgc),
    "n_braess": n_braess,
    "braess_share_percent": (n_braess / total_edges) * 100,
    "total_braess_length_km": total_braess_length_km,
    "median_braess_length_m": median_braess_length,
}

stats

# %%
