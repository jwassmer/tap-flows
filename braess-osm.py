# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import Plotting as pl
from src import osmGraphs as og
from src import multiCommoditySocialCost as mcsc

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import osmnx as ox
import cvxpy as cp
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# %%
cologne_districts = [
    "Nippes, Cologne, Germany",
    "Ehrenfeld, Cologne, Germany",
    "Innenstadt, Cologne, Germany",
    "Lindenthal, Cologne, Germany",
    "Mülheim, Cologne, Germany",
    "Kalk, Cologne, Germany",
    "Porz, Cologne, Germany",
    "Chorweiler, Cologne, Germany",
    "Rodenkirchen, Cologne, Germany",
    "Ehrenfeld, Cologne, Germany",
]
berlin_districts = [
    "Pankow, Berlin, Germany",
    "Mitte, Berlin, Germany",
    "Neukölln, Berlin, Germany",
    "Friedrichshain-Kreuzberg, Berlin, Germany",
    "Charlottenburg-Wilmersdorf, Berlin, Germany",
    "Spandau, Berlin, Germany",
    "Steglitz-Zehlendorf, Berlin, Germany",
    "Tempelhof-Schöneberg, Berlin, Germany",
    "Treptow-Köpenick, Berlin, Germany",
    "Marzahn-Hellersdorf, Berlin, Germany",
    "Lichtenberg, Berlin, Germany",
    "Reinickendorf, Berlin, Germany",
    "Potsdam, Brandenburg, Germany",
]

# %%

gamma = 0.04

edges_list = []
nodes_list = []
boundary_list = []

for name in berlin_districts:
    print(name)
    G, bounds = og.osmGraph(name, return_boundary=True)
    G.derivative_social_cost(num_sources=20, gamma=gamma, eps=1e-3, solver=cp.MOSEK)

    nodes, edges = ox.graph_to_gdfs(G)
    edges_list.append(edges)
    nodes_list.append(nodes)
    boundary_list.append(bounds)

# %%

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
# vmin = -50
# vmax = 50
# norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
norm = mpl.colors.Normalize(vmin=100, vmax=1500)

for i, edges in enumerate(edges_list):
    braess_edges = edges[edges["derivative_social_cost"] < 0]

    selected_nodes = nodes_list[i]
    boundary = boundary_list[i]
    edges.sort_values("flow", inplace=True, ascending=True)
    edges.plot(column="flow", ax=ax, legend=False, cmap=cmap, norm=norm)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=2)
    # selected_nodes.plot(ax=ax, marker="x", color="black", markersize=10, zorder=3)
    if len(braess_edges) > 0:
        braess_edges.plot(ax=ax, color="red", linewidth=3, zorder=10)
        # print(len(braess_edges))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, extend="both", pad=0.0, shrink=1 / 2)


# %%

location = "Potsdam, Germany"

G, bounds = og.osmGraph(location, return_boundary=True)
# nodes.set_geometry("voronoi", inplace=True)
# nodes.plot("population")
G.derivative_social_cost(num_sources=25, gamma=0.03, eps=1e-3, solver=cp.MOSEK)

nodes, edges = ox.graph_to_gdfs(G)
nodes["population"].sum()
edges["length"].sum()

edges.sort_values("derivative_social_cost", inplace=True, ascending=False)


edges.explore().save("edges_explore.html")
# %%

nodes_list = []
edges_list = []

gamma_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]  # , 0.07, 0.08, 0.09]
# gamma_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17]

for gamma in gamma_list:
    G, bounds = og.osmGraph(
        location, return_boundary=True  # , heavy_boundary=True, buffer_meter=2_500
    )

    G.derivative_social_cost(num_sources=25, gamma=gamma, eps=1e-3, solver=cp.MOSEK)

    nodes, edges = ox.graph_to_gdfs(G)
    nodes_list.append(nodes)
    edges_list.append(edges)

    braess_edges = edges[edges["derivative_social_cost"] < 0]
    total_braess_lengt = braess_edges["length"].sum()
    print("Gamma:", gamma)
    print("Braess Edges:", len(braess_edges))
    print("Braess Length:", total_braess_lengt)


# %%
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axs = axs.flatten()
cmap = plt.get_cmap("Reds")
cmap.set_under("#0571b0")
# norm = mpl.colors.LogNorm(vmin=1e2, vmax=5e3)
norm = mpl.colors.Normalize(vmin=0, vmax=5e2)

labels = [
    r"\textbf{a}",
    r"\textbf{b}",
    r"\textbf{c}",
    r"\textbf{d}",
    r"\textbf{e}",
    r"\textbf{f}",
]

for i, ax in enumerate(axs):
    edges = edges_list[i]
    nodes = nodes_list[i]

    braess_edges = edges[edges["derivative_social_cost"] < 0]
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)

    edges.plot(
        column="derivative_social_cost",
        ax=ax,
        legend=False,
        cmap=cmap,
        norm=norm,
        linewidth=3,
    )
    # if len(braess_edges) > 0:
    #    braess_edges.plot(ax=ax, color="Red", zorder=3, linewidth=4)
    # bounds.plot(ax=ax, color="grey", linewidth=1.5, zorder=0, alpha=0.1)
    bounds.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=2)
    nodes[nodes["source"]].plot(
        ax=ax, marker="x", color="black", markersize=25, zorder=3
    )
    ax.set_title(
        r"Traffic Volume $\gamma = $"
        + f"{gamma_list[i]:.2f} \n Braess edge length: {braess_edges['length'].sum():.0f} m"
    )
    ax.grid()

    ax.text(
        0.02,
        1.05,
        labels[i],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
    )

    # inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=0)
    # inset_ax.axis("off")
    # norm_flow = edges["flow"] / edges["flow"].max()
    # norm_derivative = (
    #    edges["derivative_social_cost"] / edges["derivative_social_cost"].max()
    # )
    # diff = np.abs(norm_flow - norm_derivative)
    # var = np.var(diff)
    # inset_ax.scatter(
    #    norm_flow,
    #    norm_derivative,
    #    marker="o",
    #    edgecolors="black",
    #    color="lightgrey",
    #    label=f"$\sigma= {var:.1e}$",
    # )
    # inset_ax.set_xlabel(r"Traffic Flow $\|f_e\|$")
    # inset_ax.set_ylabel(r"Sensitivity $\|\partial_{\beta_e} \mathrm{sc}_e\|$")


cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=axs,
    extend="both",
    shrink=1 / 2,
    pad=0.01,
    aspect=40,
)

cbar.ax.set_ylabel(r"Sensitivity $\partial_{\beta_e} \mathrm{sc}_e$")

red_patch = Patch(color="#0571b0", label="Braess Edges")
source_marker = Line2D(
    [],
    [],
    color="black",
    marker="x",
    linestyle="None",
    markersize=10,
    label="Source Nodes",
)
axs[2].legend(
    handles=[red_patch, source_marker],
    loc="upper right",
    # bbox_to_anchor=(1.4, 1.05),  # Adjust the position as needed
    borderaxespad=0.0,
)

fig.savefig(f"figs/{location}-braess.pdf", dpi=300, bbox_inches="tight")

# %%

fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

for i, ax in enumerate(axs.flatten()):
    edges = edges_list[i]
    edges.sort_values("derivative_social_cost", inplace=True, ascending=False)

    norm_flow = edges["flow"] / edges["flow"].max()
    norm_derivative = (
        edges["derivative_social_cost"] / edges["derivative_social_cost"].max()
    )

    diff = np.abs(norm_flow - norm_derivative)
    var = np.var(diff)
    ax.scatter(
        norm_flow,
        norm_derivative,
        marker="o",
        edgecolors="black",
        color="lightgrey",
        label=f"$\sigma= {var:.1e}$",
    )
    ax.set_title(r"Traffic Volume $\gamma = $" + f"{gamma_list[i]:.2f}")
    ax.set_xlabel(r"Traffic Flow $\|f_e\|$")
    ax.set_ylabel(r"Sensitivity $\|\partial_{\beta_e} \mathrm{sc}_e\|$")
    ax.legend()
    ax.grid()
    ax.text(
        0.02,
        1.05,
        labels[i],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
    )


fig.savefig(f"figs/{location}-braess-correlation.pdf", dpi=300, bbox_inches="tight")


# plt.plot(edges["flow"], edges["derivative_social_cost"], marker="o")
# %%
