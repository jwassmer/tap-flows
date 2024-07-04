import pytest
import numpy as np
import networkx as nx
from src import SocialCost as sc
from src import TAPOptimization as tap

from sklearn.linear_model import LinearRegression


@pytest.fixture
def setup_graph():
    def _setup(num_nodes, edge_prob, seed):
        num_edges = int(num_nodes * (num_nodes - 1) * edge_prob / 2)
        G = tap.random_graph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            seed=seed,
            alpha="random",
            beta="random",
        )
        return G

    return _setup


@pytest.mark.parametrize(
    "num_nodes, edge_prob, seed",
    [
        (10, 0.2, 42),
        (15, 0.3, 24),
        (20, 0.5, 10),
    ],
)
def test_social_cost_slope(num_nodes, edge_prob, seed, setup_graph):
    G = setup_graph(num_nodes, edge_prob, seed)
    P = np.zeros(G.number_of_nodes())
    load = 500
    np.random.seed(seed)
    source = np.random.randint(0, G.number_of_nodes())
    P[source] = load
    targets = np.delete(np.arange(G.number_of_nodes()), source)
    P[targets] = -load / len(targets)

    tt_fs = nx.get_edge_attributes(G, "tt_function")
    alpha_arr = np.array([tt_fs[e](1) - tt_fs[e](0) for e in G.edges()])

    slopes = sc.all_social_cost_derivatives(G, P, alpha_arr)

    isclose_arr = {}
    for edge, slope in slopes.items():
        m = sc.linreg_slope_sc(G, P, edge)
        isclose_arr[edge] = np.isclose(m, slope)

    assert all(
        isclose_arr
    ), "Derivative of social cost is not equal to its linreg slope."
