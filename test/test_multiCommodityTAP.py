# %%
import numpy as np
from src import multiCommodityTAP as mc
from src import Plotting as pl
from src import TAPOptimization as tap

# %%
num_nodes = 20
G = mc.random_graph(
    seed=20,
    num_nodes=num_nodes,
    num_edges=15,
    alpha=1,
    beta=3,
)
nodes = G.nodes
load = 1000
P = np.zeros(G.number_of_nodes())
sources = [1]
P[sources] = load / len(sources)
targets = [16]  # np.delete(nodes, sources)
P[targets] = -load / len(targets)

demands = [P]
mc.price_of_anarchy(G, demands)
# %%
Fue = mc.solve_multicommodity_tap(G, demands, social_optimum=False)
Fso = mc.solve_multicommodity_tap(G, demands, social_optimum=True)
# pl.graphPlotCC(G, cc=Fue)  # , edge_labels=dict(zip(G.edges, Fue)))

# %%
print("User Equilibrium:")
fue = tap.user_equilibrium(G, P)
print("Social Optimum:")
fso = tap.social_optimum(G, P)
# %%
np.round(Fue - fue, 5)
# %%
np.round(Fso - fso, 5)
# %%
tap.social_cost(G, Fue) - tap.social_cost(G, Fso)
# %%
tap.social_cost(G, fue) - tap.social_cost(G, fso)
# %%
