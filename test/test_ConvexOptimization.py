# %%
import numpy as np
import networkx as nx
from src import Equilibirium as eq
from src import ConvexOptimization as cv
from src import GraphGenerator as gg
from src import LinAlg as la

import pytest
import numpy as np
from src import ConvexOptimization as cv


@pytest.mark.parametrize(
    "num_nodes, edge_prob, seed",
    [
        (10, 0.2, 42),
        (15, 0.3, 24),
        (20, 0.5, 10),
    ],
)
def test_convex_optimization_linflow(num_nodes, edge_prob, seed):
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Create a random Erdős-Rényi graph
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=seed, directed=False)

    # Initialize the supply/demand vector
    P = np.zeros(G.number_of_nodes())
    P[0], P[-1] = 100, -100

    # Initialize the capacity vector
    K = np.random.rand(G.number_of_edges())
    K = np.ones(G.number_of_edges())  # Setting all capacities to 1 for simplicity

    # Compute the convex optimization flow without setting attributes
    fco0 = cv.convex_optimization_linflow(G, P, K)

    # Set the graph attributes for capacities and supply/demand
    nx.set_edge_attributes(G, dict(zip(G.edges, K)), "weight")
    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")

    # Compute the convex optimization flow using the graph attributes
    fco1 = cv.convex_optimization_linflow(G)

    # Check if the two computed flows are almost equal
    assert all(
        np.abs(fco0 - fco1) < 1e-5
    ), "Mismatch between flows computed with and without graph attributes"

    # Compute the linear flow using equilibrium calculations
    flin = eq.linear_flow(G)

    # Check if the convex optimization flow is almost equal to the linear flow
    assert all(
        np.abs(fco0 - list(flin.values())) < 1e-5
    ), "Mismatch between convex optimization flow and linear flow"


# )


# %%
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(10, 0.2, seed=seed, directed=False)
    P = np.zeros(G.number_of_nodes())
    P[0], P[-1] = 100, -100
    K = np.random.rand(G.number_of_edges())
    K = np.ones(G.number_of_edges())
    # nx.draw(G)
    fco0 = cv.convex_optimization_linflow(G, P, K)
    nx.set_edge_attributes(G, dict(zip(G.edges, K)), "weight")
    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")
    fco1 = cv.convex_optimization_linflow(G)
    result0 = all(np.abs(fco0 - fco1) < 1e-5)

    assert result0 == True

    flin = eq.linear_flow(G)

    result1 = all(np.abs(fco0 - list(flin.values())) < 1e-5)
    assert result1 == True

    # flin = eq.linear_flow(G, P)
# %%
