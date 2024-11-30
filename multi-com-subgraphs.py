# %%
# %%
from src import Graphs as gr
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap
from src import SocialCost as sc
from src import Plotting as pl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
G = gr.random_graph(10, num_edges=20, beta="random", alpha="random", directed=True)

od_matrix = -1 * np.ones((G.number_of_nodes(), G.number_of_nodes()))
np.fill_diagonal(od_matrix, G.number_of_nodes() - 1)


fw = mc.solve_multicommodity_tap(G, od_matrix, return_fw=True)
# %%
cmap = plt.get_cmap("cividis")
cmap.set_under("lightgrey")
norm = plt.Normalize(1e-1, max(fw[0]))
for i, f in enumerate(fw):
    pl.graphPlot(G, ec=f, cmap=cmap, norm=norm)
# %%
