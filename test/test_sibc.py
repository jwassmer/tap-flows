import networkx as nx
import numpy as np
import pytest

from src import Graphs as gr
from src import SIBC as sibc
from src import TAPOptimization as tap
from src import multiCommodityTAP as mc


@pytest.fixture
def seeded_graph():
    graph = gr.random_graph(
        seed=42,
        num_nodes=20,
        num_edges=25,
        alpha=0,
        beta="random",
    )
    betas = np.random.default_rng(42).random(graph.number_of_edges())
    nx.set_edge_attributes(graph, dict(zip(graph.edges, betas)), "beta")
    return graph


def test_interaction_betweenness_matches_networkx(seeded_graph):
    num_nodes = seeded_graph.number_of_nodes()
    od_matrix = -np.random.default_rng(42).random((num_nodes, num_nodes))
    np.fill_diagonal(od_matrix, 0.0)
    np.fill_diagonal(od_matrix, -np.sum(od_matrix, axis=0))

    ibc = sibc._interaction_betweenness_centrality(
        seeded_graph,
        weight="beta",
        od_matrix=np.abs(od_matrix),
    )
    nx_ebc = list(
        nx.edge_betweenness_centrality(
            seeded_graph, weight="beta", normalized=False
        ).values()
    )

    assert np.allclose(ibc, nx_ebc, atol=1e-8)


def test_multicommodity_flow_matches_ibc_when_alpha_zero(seeded_graph):
    num_nodes = seeded_graph.number_of_nodes()
    od_matrix = -np.random.default_rng(7).random((num_nodes, num_nodes))
    np.fill_diagonal(od_matrix, 0.0)
    np.fill_diagonal(od_matrix, -np.sum(od_matrix, axis=0))
    demands = [od_matrix[:, i] for i in range(num_nodes)]

    ibc = sibc._interaction_betweenness_centrality(
        seeded_graph,
        weight="beta",
        od_matrix=np.abs(od_matrix),
    )
    flow = mc.solve_multicommodity_tap(seeded_graph, demands)

    assert np.allclose(flow, ibc, atol=1e-6)


def test_single_source_sibc_matches_user_equilibrium():
    graph = gr.random_graph(
        seed=123,
        num_nodes=12,
        num_edges=18,
        alpha=0,
        beta="random",
    )
    P = -np.ones(graph.number_of_nodes())
    P[0] = np.sum(np.abs(P)) - 1

    sibc_flow = sibc._single_source_interaction_betweenness_centrality(
        graph, weight="beta", P=P
    )
    ue_flow = tap.user_equilibrium(graph, P, positive_constraint=True)

    assert np.allclose(ue_flow, sibc_flow, atol=1e-5)
