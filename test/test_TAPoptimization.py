# %%
import pytest
import networkx as nx
import numpy as np
import cvxpy as cp
from src import ConvexOptimization as co
from src import TAPOptimization as tap
import time


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
        A_od = tap.ODmatrix(G)
        return G, A_od

    return _setup


@pytest.mark.parametrize(
    "num_nodes, edge_prob, seed",
    [
        (10, 0.2, 42),
        (15, 0.3, 24),
        (20, 0.5, 10),
    ],
)
def test_user_equilibrium(num_nodes, edge_prob, seed, setup_graph):
    G, A = setup_graph(num_nodes, edge_prob, seed)
    B = np.zeros((num_nodes, num_nodes))
    B[:, 0] = A[:, 0]
    P = A[:, 0]
    F = tap.user_equilibrium(G, P, positive_constraint=True)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    assert (
        len(F) == G.number_of_edges()
    ), "The flow vector length should match the number of edges."
    assert np.all(
        F >= -1e-7
    ), "Flow values should be non-negative when positive constraint is applied."
    f = tap.user_equilibrium(G, P, positive_constraint=True)
    assert all(np.isclose(E @ f, B[:, 0])), "Flow must satisfy the demand constraints."


@pytest.mark.parametrize(
    "num_nodes, edge_prob, seed",
    [
        (10, 0.2, 42),
        (15, 0.3, 24),
        (20, 0.5, 10),
    ],
)
def test_linear_tap_solver(num_nodes, edge_prob, seed, setup_graph):
    G, A = setup_graph(num_nodes, edge_prob, seed)
    E = -nx.incidence_matrix(G, oriented=True).toarray()

    P = A[:, 0]
    B = np.zeros((num_nodes, num_nodes))
    B[:, 0] = P
    fue = tap.user_equilibrium(G, P, positive_constraint=False)
    flin, lamb = tap.linearTAP(G, P)

    assert np.all(
        np.abs(fue - flin) < 1e-5
    ), "Mismatch between user equilibrium and linear TAP solver."
    # print(np.isclose(E @ F, B[:, 0]))
    assert all(
        np.isclose(E @ flin, B[:, 0])
    ), "Flow must satisfy the demand constraints."


# %%

if __name__ == "__main__":
    num_nodes = 10
    num_edges = 15
    seed = 42
    G = tap.random_graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        seed=seed,
        alpha=10,
        beta=10,
    )
    A = tap.ODmatrix(G)

    P = A[:, 0]
    F = tap.user_equilibrium(G, P, positive_constraint=True)
    E = -nx.incidence_matrix(G, oriented=True).toarray()
    assert (
        len(F) == G.number_of_edges()
    ), "The flow vector length should match the number of edges."
    assert np.all(
        F >= -1e-7
    ), "Flow values should be non-negative when positive constraint is applied."
    f = tap.user_equilibrium(G, P, positive_constraint=True)
    assert all(np.isclose(E @ f, P)), "Flow must satisfy the demand constraints."

    # %%

    G = G.to_undirected()

    P = np.zeros(G.number_of_nodes())
    P[0], P[-1] = 100, -100

    fue = tap.user_equilibrium(G, P, positive_constraint=False)
    flin, lamb = tap.linearTAP(G, P)

    flin - fue
    # %%
    nx.set_node_attributes(G, dict(zip(G.nodes, P)), "P")
    feq = co.convex_optimization_kcl_tap(G, positive_constraint=False)
    feq - fue
    # %%
