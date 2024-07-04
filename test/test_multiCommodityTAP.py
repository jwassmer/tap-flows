# %%
import numpy as np
import pytest
import networkx as nx
from src import multiCommodityTAP as mc
from src import TAPOptimization as tap


@pytest.fixture
def setup_graph():
    def _setup(num_nodes, edge_prob, seed):
        num_edges = int(num_nodes * (num_nodes - 1) * edge_prob / 2)
        G = mc.random_graph(
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
def test_multicommodityTAP(num_nodes, edge_prob, seed, setup_graph):
    G = setup_graph(num_nodes, edge_prob, seed)

    load = 1000
    P = np.zeros(G.number_of_nodes())
    sources = [0]
    P[sources] = load / len(sources)
    targets = np.delete(G.nodes, sources)
    P[targets] = -load / len(targets)

    demands = [P]

    Fue = mc.solve_multicommodity_tap(G, demands, social_optimum=False)
    Fso = mc.solve_multicommodity_tap(G, demands, social_optimum=True)

    fue = tap.user_equilibrium(G, P)
    fso = tap.social_optimum(G, P)

    ue_diff = np.isclose(Fue, fue)
    so_diff = np.isclose(Fso, fso)

    assert all(ue_diff), "Mismatch between multicommodity UE and tap UE flow"

    assert all(so_diff), "Mismatch between multicommodity SO and tap SO flow"


# %%
if __name__ == "__main__":
    num_nodes = 20
    G = mc.random_graph(
        seed=20,
        num_nodes=num_nodes,
        num_edges=15,
        alpha=1,
        beta=3,
    )
    # nodes = G.nodes
    load = 1000
    P = np.zeros(G.number_of_nodes())
    sources = [1]
    P[sources] = load / len(sources)
    targets = np.delete(G.nodes, sources)
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
