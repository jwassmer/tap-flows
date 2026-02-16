"""Shared graph helpers used across TAP solvers."""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np


def edge_attribute_array(
    graph: nx.Graph, attr: str, values: Sequence[float] | np.ndarray | None = None
) -> np.ndarray:
    """Return edge values as a dense array with one value per edge."""
    if values is None:
        mapping = nx.get_edge_attributes(graph, attr)
        if len(mapping) != graph.number_of_edges():
            raise ValueError(
                f"Missing edge attribute '{attr}' on at least one edge "
                f"({len(mapping)}/{graph.number_of_edges()})."
            )
        arr = np.asarray(list(mapping.values()), dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.shape != (graph.number_of_edges(),):
            raise ValueError(
                f"Expected '{attr}' with shape ({graph.number_of_edges()},), "
                f"got {arr.shape}."
            )
    return arr


def validate_node_balance(graph: nx.Graph, demands: Iterable[float]) -> np.ndarray:
    """Validate node demand vector size and return as ndarray."""
    demand_vec = np.asarray(demands, dtype=float).reshape(-1)
    expected = graph.number_of_nodes()
    if demand_vec.shape != (expected,):
        raise ValueError(f"Expected demand vector shape ({expected},), got {demand_vec.shape}.")
    return demand_vec


def random_directed_cost_graph(
    num_nodes: int = 10,
    num_edges: int = 15,
    seed: int = 42,
    alpha: float | str = 1,
    beta: float | str = 0,
) -> nx.DiGraph:
    """Create a connected random directed graph with linear edge costs."""
    connected = False
    if num_edges < num_nodes - 1:
        num_edges = num_nodes - 1

    while not connected:
        undirected = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)
        connected = nx.is_connected(undirected)
        num_edges += 1

    if isinstance(alpha, str) and alpha == "random_symmetric":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, undirected.number_of_edges())
    if isinstance(beta, str) and beta == "random_symmetric":
        np.random.seed(seed)
        beta = 100 * np.random.rand(undirected.number_of_edges())

    graph = undirected.to_directed()

    if isinstance(alpha, (int, float)):
        alpha = alpha * np.ones(graph.number_of_edges())
    if isinstance(beta, (int, float)):
        beta = beta * np.ones(graph.number_of_edges())
    if isinstance(alpha, str) and alpha == "random":
        np.random.seed(seed)
        alpha = np.random.uniform(0.1, 1, graph.number_of_edges())
    if isinstance(beta, str) and beta == "random":
        np.random.seed(seed)
        beta = 100 * np.random.rand(graph.number_of_edges())

    alpha_arr = np.asarray(alpha, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    if alpha_arr.shape != (graph.number_of_edges(),):
        raise ValueError(
            f"Expected alpha shape ({graph.number_of_edges()},), got {alpha_arr.shape}."
        )
    if beta_arr.shape != (graph.number_of_edges(),):
        raise ValueError(
            f"Expected beta shape ({graph.number_of_edges()},), got {beta_arr.shape}."
        )

    nx.set_edge_attributes(graph, dict(zip(graph.edges, alpha_arr)), "alpha")
    nx.set_edge_attributes(graph, dict(zip(graph.edges, beta_arr)), "beta")

    pos = nx.spring_layout(graph, seed=seed)
    nx.set_node_attributes(graph, pos, "pos")
    nx.set_edge_attributes(graph, "black", "color")
    nx.set_node_attributes(graph, "lightgrey", "color")
    return graph
