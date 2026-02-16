# %%
import networkx as nx
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import cvxpy as cp
from tqdm import tqdm

from sklearn.linear_model import LinearRegression

from src import Graphs as gr
from src import Plotting as pl
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import osmGraphs as og
from src import multiCommodityTAP as mc
from src import multiCommoditySocialCost as mcsc
from src import SocialCost as sc


def middle_value(arr):
    n = len(arr)
    if n % 2 == 1:  # Odd length
        return arr[n // 2]
    else:  # Even length
        return (arr[n // 2 - 1] + arr[n // 2]) / 2


pl.mpl_params(fontsize=14)


# %%

G = gr.random_planar_graph(50, seed=1)

G.derivative_social_cost(num_sources=len(G), gamma=0.02, eps=1e-3, solver=cp.MOSEK)
nodes, edges = ox.graph_to_gdfs(G)
edges.sort_values("derivative_social_cost", inplace=True, ascending=False)
edges["derivative_social_cost"].describe()


cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(
    vmin=min(-1e-3, edges["derivative_social_cost"].min()),
    vcenter=0,
    vmax=edges["derivative_social_cost"].max(),
)
edges.plot(column="derivative_social_cost", cmap=cmap, norm=norm, legend=True)

# %%
numerical_derivative_dict = {}
F = np.array(list(nx.get_edge_attributes(G, "flow").values()))
sc0 = sc.total_social_cost(G, F)

demand_list = og.demands(G, len(G), gamma=0.02)

for edge in tqdm(list(G.edges)):
    beta = nx.get_edge_attributes(G, "beta")
    beta0 = beta[edge]
    beta[edge] = beta0 - 0.001 * beta0
    beta_arr = np.array(list(beta.values()))
    f = mc.solve_multicommodity_tap(
        G, demand_list, pos_flows=True, beta=beta_arr, solver=cp.MOSEK
    )
    # G.flows(gamma=0.02, beta=beta_arr, solver=cp.MOSEK)
    # f = np.array(list(nx.get_edge_attributes(G, "flow").values()))
    social_cost = sc.total_social_cost(G, f, beta=beta_arr)

    sc_list = [sc0, social_cost]
    beta_list = [beta0, beta[edge]]

    grad = np.gradient(sc_list, beta_list)

    numerical_derivative_dict[edge] = np.mean(grad)

# %%

labels = [r"\textbf{a}", r"\textbf{b}"]

edges["numerical_derivative"] = pd.Series(numerical_derivative_dict)

variance = np.var(edges["numerical_derivative"] / edges["derivative_social_cost"])

cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=30)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))


axs[1].plot(edges["derivative_social_cost"], edges["numerical_derivative"], "o")
axs[1].grid()
axs[1].set_xlabel(r"Analytical derivative of social cost")
axs[1].set_ylabel(r"Numerical derivative of social cost")

axs[1].text(0.01, 1.01, labels[1], transform=axs[1].transAxes, fontsize=22)

edges.plot(
    ax=axs[0],
    column="derivative_social_cost",
    cmap=cmap,
    norm=norm,
    legend=False,
    linewidth=3,
)
axs[0].axis("off")
axs[0].text(0.05, 0.98, labels[0], transform=axs[0].transAxes, fontsize=22)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=axs[0], orientation="horizontal", shrink=0.5, pad=-0.1, extend="both"
)

tick_arr = [-5, 0, 10, 30]
cbar.set_ticks(tick_arr)
cbar.set_ticklabels(tick_arr)

cbar.ax.set_xlabel("Derivative of social cost")

fig.savefig("figs/analytical-vs-numerical-derivative-cost.pdf", bbox_inches="tight")


# %%


braess_edge = edges.iloc[-1].name

print(edges.loc[braess_edge].derivative_social_cost)

slope_list, beta_list, cost_list = mcsc.numerical_derivative(
    G, demand_list, braess_edge, var_percentage=0.2, num=50, solver=cp.MOSEK
)


print(middle_value(slope_list))
plt.plot(beta_list, cost_list, marker="o")
plt.grid()


# %%

g = og.osmGraph("Potsdam,Germany")

n, e = ox.graph_to_gdfs(g)

gamma_list = np.linspace(1e-3, 0.1, 10)

for gamma in gamma_list:
    g.derivative_social_cost(num_sources=25, gamma=gamma, eps=1e-3, solver=cp.MOSEK)

    d_social_cost_vec = np.array(
        list(nx.get_edge_attributes(g, "derivative_social_cost").values())
    )

    num_braess_edges = len(d_social_cost_vec[d_social_cost_vec < 0])
    print("Gamma:", gamma)
    print("Braess Edges:", num_braess_edges)

# %%

g = og.osmGraph(
    "Potsdam,Germany",
)
g.derivative_social_cost(num_sources=25, gamma=0.04, eps=1e-3, solver=cp.OSQP)

nodes, edges = ox.graph_to_gdfs(g)
edges.sort_values(by="derivative_social_cost", inplace=True, ascending=False)

cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(
    vmin=min(-1e-3, edges["derivative_social_cost"].min()),
    vcenter=0,
    vmax=edges["derivative_social_cost"].max(),
)

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
edges.plot(ax=ax, column="derivative_social_cost", cmap=cmap, norm=norm, legend=True)
nodes[nodes["source"]].plot(ax=ax, color="black", markersize=25, zorder=3)

num_braess_edges = len(edges[edges["derivative_social_cost"] < 0])
print("Braess Edges:", num_braess_edges)


# %%

edges["name"] = edges["name"].astype(str)

fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
# edges[edges["name"].str.contains("Neusser")].plot(ax=ax, color="red", linewidth=5)
edges.plot(ax=ax, column="flow", cmap="cividis", legend=True)
# %%


braess_edge = edges.sort_values("derivative_social_cost").iloc[0].name

print(edges.loc[braess_edge])


demand_list = og.demands(g, 30, gamma=0.04)

slope_list, beta_list, cost_list = mcsc.numerical_derivative(
    g, demand_list, braess_edge, var_percentage=0.1, num=10, solver=cp.OSQP
)


print(np.mean(slope_list))
plt.plot(beta_list, cost_list)
plt.grid()

# %%

plt.plot(edges["length"], edges["derivative_social_cost"], "o")
plt.grid()
# %%
edges.sort_values("flow")
# %%
