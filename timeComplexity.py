# %%
import time
import numpy as np

from src import ConvexOptimization as co
from src import GraphGenerator as gg

# %%
source, target = ["d", "c"], ["b", "a"]
total_load = 1000
# Example graph creation

# %%

time_list = []
num_nodes_list = np.logspace(2, 3.7, num=20, dtype=int)
for num_nodes in num_nodes_list:
    U = gg.random_graph(
        source,
        target,
        total_load,
        seed=42,
        num_nodes=num_nodes,
        num_edges=num_nodes * 6,
        alpha="random",
        beta="random",
    )
    G = gg.to_directed_flow_graph(U)
    print(G)

    start = time.time()
    co.convex_optimization_linflow(G)
    total_time = time.time() - start
    time_list.append(total_time)

# %%

import matplotlib.pyplot as plt

plt.plot(num_nodes_list, time_list)
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.xlabel("Number of nodes")
plt.ylabel("Time (s)")
plt.plot(num_nodes_list, (num_nodes_list) ** (1.4) / (1e5), label="O(n)")

# %%
num_nodes = 5000
U = gg.random_graph(
    source,
    target,
    total_load,
    seed=42,
    num_nodes=num_nodes,
    num_edges=num_nodes * 5,
    alpha="random",
    beta="random",
)
G = gg.to_directed_flow_graph(U)
print(G)

start = time.time()
co.convex_optimization_linflow(G)
total_time = time.time() - start
print(total_time)
# %%
