# %%
import networkx as nx
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl

from src import Graphs as gr
from src import Plotting as pl
from src import SIBC as sibc
from src import TAPOptimization as tap


# %%
G = gr.triangularLattice(2, beta="random", alpha="random", directed=True)
P = np.zeros(G.number_of_nodes())
P[0] = 1
# P[np.delete(np.arange(G.number_of_nodes()), 0)] = -sum(P) / (G.number_of_nodes() - 1)
P[-1] = -1

E = -nx.incidence_matrix(G, oriented=True).toarray()

f0 = sibc._single_source_interaction_betweenness_centrality(G, P=P, weight="beta")

start_time = time.time()
f = tap.user_equilibrium(G, P, positive_constraint=True)
print("Elapsed time:", time.time() - start_time)

cycle_f = f - f0
vmin, vmax = np.min(cycle_f), np.max(cycle_f)

cmap = plt.get_cmap("coolwarm")
norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
pl.graphPlot(G, ec=f, show_labels=True)

# %%
C, cycles = gr.cycle_edge_incidence_matrix(G)
C


# %%

C1, cycles1 = gr.cycle_edge_incidence_matrix(G)

# %%
