# %%
from src import Plotting as pl
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import multiCommoditySocialCost as mcsc
from src import SocialCost as sc
from src import TAPOptimization as tap
from src import osmGraphs as og


import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
import cvxpy as cp
import matplotlib as mpl

import time


def demand_list(nodes, target_nodes_idx_list, n_vehicles=1000):

    node_attractiveness = (
        n_vehicles
        * nodes["population"]
        / np.sum(nodes["population"])
        / len(target_nodes_idx_list)
    )
    demands = []
    for t in target_nodes_idx_list:

        source_nodes_index = nodes.index.difference([t])

        P = dict(zip(nodes.index, np.zeros(len(nodes))))

        for s in source_nodes_index:
            P[s] = node_attractiveness[s]

        P[t] = -np.sum(list(P.values()))

        demands.append(list(P.values()))
    # tot_sources = np.sum(list(P.values()))

    # for t in target_nodes_idx_list:
    # P[t] = nodes.loc[t, "population"] * (1 - gamma)
    #    P[t] = -tot_sources / len(target_nodes_idx_list)

    # print("Total Population:", tot_pop)
    return demands


# %%


G, boundary = og.osmGraph(
    "Cologne,Germany",
    return_boundary=True,
    tolerance_meters=100,
    highway_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]',
)

nodes, edges = ox.graph_to_gdfs(G)

# %%
# adress_list = ["Aachener Straße 999, 50933 Köln"]
adress_list = [
    "Salzburger Weg, 50858 Köln",
    "Wendelinstr., 50933 Köln",
    "Brauweiler Weg, 50933 Köln",
]

target_node_idx_list = []
for a in adress_list:
    location = ox.geocode(a)
    print(location)
    node_idx = ox.distance.nearest_nodes(G, location[1], location[0])
    target_node_idx_list.append(node_idx)

target_node_idx_list = np.unique(target_node_idx_list)


# %%
pop_cologne = nodes["population"].sum()

parking_lots = 7_500
n_vehicles = parking_lots * 2
od_mat = demand_list(nodes, target_node_idx_list, n_vehicles=n_vehicles)
od_mat = np.array(od_mat)
sorted(od_mat[0])
# %%

nodes, edges = ox.graph_to_gdfs(G)
f_mat, lambda_mat = mc.solve_multicommodity_tap(
    G, od_mat, return_fw=True, pos_flows=True, solver=cp.OSQP, verbose=True
)
F = np.sum(f_mat, axis=0)

edges["flow"] = F
dsc = mcsc.derivative_social_cost(G, f_mat, od_mat, eps=1e-3, demands_to_sinks=False)
dsc_vec = list(dsc.values())
edges["derivative_social_cost"] = dsc_vec
# edges.sort_values("flow", ascending=True, inplace=True)
# %%

fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)

ax = axs[0]
boundary.boundary.plot(linewidth=1, ax=ax, color="black", zorder=3)
ax.grid()

for a in axs:
    nodes.plot(
        ax=a,
        marker=".",
        color="black",
        markersize=5,
        zorder=4,
        label="Source nodes",
    )

ax.text(
    0.02,
    1.02,
    r"$\textbf{a}$",
    ha="center",
    va="center",
    transform=ax.transAxes,
    fontsize=24,
)
cmap = plt.get_cmap("viridis")
cmap.set_under("lightgrey")
norm = mpl.colors.LogNorm(vmin=100, vmax=np.mean(np.sort(F)[-30:]))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cb = fig.colorbar(
    sm, ax=ax, orientation="horizontal", pad=0.01, shrink=2 / 3, extend="both"
)


sorted_edges = edges.sort_values("flow", ascending=True)
sorted_edges.plot(ax=ax, linewidth=2, column="flow", cmap=cmap, norm=norm, legend=False)

stadium_node = nodes.loc[target_node_idx_list]

stadium_node.plot(
    ax=ax,
    marker="s",
    color="black",
    markersize=100,
    zorder=3,
)
cb.ax.set_xlabel("Vehicle flow $f_{e}$")

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
norm = mpl.colors.Normalize(vmin=0, vmax=1e3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

sorted_edges = edges.sort_values("derivative_social_cost", ascending=True)
sorted_edges.plot(
    ax=ax,
    linewidth=2,
    column="derivative_social_cost",
    cmap=cmap,
    norm=norm,
)

braess_edges = sorted_edges[sorted_edges["derivative_social_cost"] < 0]
braess_edges.plot(
    ax=ax,
    linewidth=2,
    color="#0571b0",
    label="Braess edges",
)

stadium_node.plot(
    ax=ax,
    marker="s",
    color="black",
    markersize=100,
    zorder=3,
    label="Parking lots\n near football stadium",
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
    extend="both",
)
cb.ax.set_xlabel(r"SCGC $\frac{\partial SC}{\partial \beta_e}$")

axs[0].set_xlabel("Lon [°]")
axs[1].set_xlabel("Lon [°]")
axs[0].set_ylabel("Lat [°]")

fig.savefig("figs/cologne-soccer.pdf", dpi=300, bbox_inches="tight")


# %%


edges.to_file("figs/cologne-football_edges_cologne.geojson", driver="GeoJSON")


# %%
# Read and analyze the full Cologne edges GeoJSON with both positive and negative SCGC values.
import geopandas as gpd
import pandas as pd
import numpy as np

edges_full = edges
# Ensure CRS is set
if edges_full.crs is None:
    edges_full.set_crs(epsg=4326, inplace=True)

# Project to UTM zone for Cologne for length calculation
edges_full_utm = edges_full.to_crs(epsg=32632)
edges_full_utm["length_m"] = edges_full_utm.geometry.length

# Check presence of derivative social cost column
scgc_col = "derivative_social_cost"
has_scgc = scgc_col in edges_full_utm.columns

# Basic descriptive statistics for SCGC values
if has_scgc:
    scgc_stats = {
        "count_edges": len(edges_full_utm),
        "count_braess_edges": int((edges_full_utm[scgc_col] < 0).sum()),
        "total_length_km": float(edges_full_utm["length_m"].sum() / 1000),
        "min_scgc": float(edges_full_utm[scgc_col].min()),
        "max_scgc": float(edges_full_utm[scgc_col].max()),
        "mean_scgc": float(edges_full_utm[scgc_col].mean()),
        "median_scgc": float(edges_full_utm[scgc_col].median()),
    }
else:
    scgc_stats = {"count_edges": len(edges_full_utm)}

# Top 5 negative and top 5 positive edges by SCGC
top_neg = edges_full_utm.sort_values(scgc_col).head(5) if has_scgc else None
top_pos = (
    edges_full_utm.sort_values(scgc_col, ascending=False).head(5) if has_scgc else None
)

# Select key attributes for inspection
candidate_cols = [
    "name",
    "ref",
    "highway",
    "maxspeed",
    "lanes",
    "oneway",
    scgc_col,
    "length_m",
]
present_cols = [c for c in candidate_cols if c in edges_full_utm.columns]

top_neg_table = top_neg[present_cols].copy() if top_neg is not None else None
top_pos_table = top_pos[present_cols].copy() if top_pos is not None else None


print("Top 5 Braess edges (negative SCGC)", top_neg_table.reset_index(drop=True))

print("Top 5 critical edges (positive SCGC)", top_pos_table.reset_index(drop=True))


scgc_stats


# %%
