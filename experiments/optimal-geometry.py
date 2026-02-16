# %%
import numpy as np
import cvxpy as cp
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import ConvexOptimization as co
from src import Plotting as pl
from src import Equilibirium as eq
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc


def ODmatrix(G, load=100):
    num_nodes = G.number_of_nodes()
    A = -load * np.ones((num_nodes, num_nodes)) / (num_nodes - 1)
    np.fill_diagonal(A, load)
    return A


def effective_beta(G, F):
    alphas = nx.get_edge_attributes(G, "alpha")
    betas = nx.get_edge_attributes(G, "beta")

    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = dict(zip(G.edges, F))

    eff_beta = {e: alphas[e] * F[e] + betas[e] for e in G.edges}

    return eff_beta


# %%
loads = [1, 5, 10, 100, 200, 300, 500, 1000]

G = gr.random_graph(
    22,
    alpha="random",
    beta="random",
)
nodes = list(G.nodes)


diff_list = []
for load in loads:
    # Example graph creation

    A = ODmatrix(G, load=load)
    demands = [A[:, n] for n in nodes]

    fso0, _ = mc.solve_multicommodity_tap(
        G, demands, social_optimum=True, solver=cp.SCS
    )
    fue0, _ = mc.solve_multicommodity_tap(
        G, demands, social_optimum=False, solver=cp.SCS
    )

    diff = fso0 - fue0
    diff_list.append(diff)


vmax = max([diff.max() for diff in diff_list])
vmin = min([diff.min() for diff in diff_list])

cmap = plt.cm.coolwarm
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)

for diff in diff_list:
    pl.graphPlot(G, ec=diff, cmap=cmap, norm=norm)
    distance = np.linalg.norm(diff)
    print(f"Distance: {distance}")
# %%

A = ODmatrix(G, load=100)
demands = [A[:, n] for n in nodes]

fso0, _ = mc.solve_multicommodity_tap(G, demands, social_optimum=True, solver=cp.SCS)

beta = nx.get_edge_attributes(G, "beta")
eff_beta = effective_beta(G, fso0)

beta_diff = {e: eff_beta[e] - beta[e] for e in G.edges}

nx.set_edge_attributes(G, eff_beta, "beta")

fue, _ = mc.solve_multicommodity_tap(G, demands, social_optimum=False, solver=cp.SCS)

pl.graphPlot(G, ec=fso0 - fue, cmap=cmap, norm=norm)


cmap = plt.cm.cividis
cmap.set_under("lightgrey")
norm = mpl.colors.Normalize(vmin=1e-1, vmax=max(beta_diff.values()))


pl.graphPlot(G, ec=beta_diff, cmap=cmap, norm=norm)
# %%


G = gr.triangularLattice(
    3,
    alpha="random",
    beta="random",
)
A = ODmatrix(G, load=10)
demands = [A[:, n] for n in nodes]

iteration = range(30)
cost_diff_list = []

for _ in iteration:

    fso, _ = mc.solve_multicommodity_tap(G, demands, social_optimum=True, solver=cp.SCS)
    fue, _ = mc.solve_multicommodity_tap(
        G, demands, social_optimum=False, solver=cp.SCS, eps_rel=1e-9
    )

    total_social_cost_ue = sc.total_social_cost(G, fue)
    total_social_cost_so = sc.total_social_cost(G, fso)

    cost_diff = total_social_cost_ue - total_social_cost_so

    eff_beta = effective_beta(G, fso)
    beta = nx.set_edge_attributes(G, eff_beta, "beta")

    cost_diff_list.append(cost_diff)


# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(iteration, cost_diff_list)
ax.set_yscale("log")
ax.grid()
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\Delta$ SC")
# %%
