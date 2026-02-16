"""Social-cost utilities for linear edge travel-time functions."""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, Tuple

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

Node = Hashable
Edge = Tuple[Node, Node]



def build_braess_graph() -> nx.DiGraph:
    """Return the canonical Braess network used in analytic examples."""
    graph = nx.DiGraph()

    a, b, c, d = 0, 1, 2, 3
    graph.add_nodes_from(
        [
            (a, {"pos": (0, 0.5)}),
            (b, {"pos": (0.5, 1)}),
            (c, {"pos": (0.5, 0)}),
            (d, {"pos": (1, 0.5)}),
        ]
    )

    graph.add_edges_from(
        [
            (a, b, {"alpha": 1 / 100, "beta": 10}),
            (b, d, {"alpha": 1 / 1000, "beta": 25}),
            (a, c, {"alpha": 1 / 1000, "beta": 25}),
            (c, d, {"alpha": 1 / 100, "beta": 10}),
            (b, c, {"alpha": 1 / 1000, "beta": 1}),
        ]
    )

    nx.set_edge_attributes(graph, "black", "color")
    nx.set_node_attributes(graph, "lightgrey", "color")
    return graph



def braessGraph() -> nx.DiGraph:
    """Backward-compatible alias for :func:`build_braess_graph`."""
    return build_braess_graph()



def _edge_arrays(
    graph: nx.DiGraph,
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if alpha is None:
        alpha = np.array(list(nx.get_edge_attributes(graph, "alpha").values()), dtype=float)
    if beta is None:
        beta = np.array(list(nx.get_edge_attributes(graph, "beta").values()), dtype=float)
    return np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)



def social_cost_vec(
    graph: nx.DiGraph,
    f: Iterable[float],
    alpha: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-edge social-cost contributions ``alpha f^2 + beta f``."""
    alpha_arr, beta_arr = _edge_arrays(graph, alpha=alpha, beta=beta)
    flow = np.asarray(f, dtype=float)
    return alpha_arr * flow**2 + beta_arr * flow



def total_social_cost(graph: nx.DiGraph, f: Iterable[float], **kwargs) -> float:
    """Return total social cost for edge flows ``f``."""
    alpha = kwargs.pop("alpha", None)
    beta = kwargs.pop("beta", None)
    return float(np.sum(social_cost_vec(graph, f, alpha=alpha, beta=beta)))



def _social_cost_from_vecs(
    graph: nx.DiGraph,
    alpha: np.ndarray,
    beta: np.ndarray,
    demands: np.ndarray,
) -> float:
    """Analytic social cost using incidence-based closed form."""
    incidence = -nx.incidence_matrix(graph, oriented=True)
    num_nodes = incidence.shape[0]

    laplacian = incidence @ np.diag(1 / alpha) @ incidence.T
    nx.set_edge_attributes(graph, dict(zip(graph.edges, beta / alpha)), "gamma")
    adjacency = nx.adjacency_matrix(graph, weight="gamma")
    gamma = adjacency - adjacency.T

    multipliers = np.linalg.pinv(laplacian) @ (demands + gamma @ np.ones(num_nodes))
    delta_lambda = incidence.T @ multipliers

    sc_values = (delta_lambda**2 - delta_lambda * beta) / alpha
    return float(np.sum(sc_values))



def derivative_social_cost_edge(
    graph: nx.DiGraph,
    laplacian_inverse: np.ndarray,
    demands: np.ndarray,
    edge: Edge,
    alpha: np.ndarray,
) -> float:
    """Derivative of social cost w.r.t. ``beta_edge`` using closed form."""
    a, b = edge
    node_order = list(graph.nodes)
    edge_order = list(graph.edges)

    a_idx = node_order.index(a)
    b_idx = node_order.index(b)
    edge_idx = edge_order.index(edge)

    demand_vec = np.asarray(demands, dtype=float)
    slope = (laplacian_inverse[a_idx, :] - laplacian_inverse[b_idx, :]) @ demand_vec
    return float(slope / alpha[edge_idx])



def derivative_socia_cost_ab(
    graph: nx.DiGraph,
    laplacian_inverse: np.ndarray,
    demands: np.ndarray,
    edge: Edge,
    alpha: np.ndarray,
) -> float:
    """Backward-compatible alias with original misspelled name."""
    return derivative_social_cost_edge(graph, laplacian_inverse, demands, edge, alpha)



def all_social_cost_derivatives(
    graph: nx.DiGraph,
    demands: np.ndarray,
    alpha_arr: np.ndarray | None = None,
) -> Dict[Edge, float]:
    """Return derivatives of total social cost w.r.t all edge ``beta`` values."""
    alpha_arr, _ = _edge_arrays(graph, alpha=alpha_arr, beta=None)

    incidence = -nx.incidence_matrix(graph, oriented=True)
    laplacian = incidence @ np.diag(1 / alpha_arr) @ incidence.T
    laplacian_inverse = np.linalg.pinv(laplacian)

    slopes = {}
    for edge in graph.edges:
        slopes[edge] = derivative_social_cost_edge(
            graph,
            laplacian_inverse,
            demands,
            edge,
            alpha_arr,
        )
    return slopes



def all_derivatives_slope_social_cost(graph: nx.DiGraph, demands: np.ndarray):
    """Legacy wrapper for the analytic derivative computation."""
    return all_social_cost_derivatives(graph, demands)



def slope_social_cost(graph: nx.DiGraph, demands: np.ndarray, edge: Edge) -> float:
    """Return a single edge derivative of social cost."""
    alpha_arr, _ = _edge_arrays(graph)
    incidence = -nx.incidence_matrix(graph, oriented=True).toarray()
    laplacian = incidence @ np.diag(1 / alpha_arr) @ incidence.T
    laplacian_inverse = np.linalg.pinv(laplacian)
    return derivative_social_cost_edge(graph, laplacian_inverse, demands, edge, alpha_arr)



def all_braess_edges(graph: nx.DiGraph, demands: np.ndarray) -> Dict[Edge, float]:
    """Return edge derivatives; negative entries correspond to Braess edges."""
    return all_social_cost_derivatives(graph, demands)



def linreg_slope_sc(graph: nx.DiGraph, demands: np.ndarray, edge: Edge) -> float:
    """Estimate slope via linear regression over local beta perturbations."""
    alpha_arr, beta_arr = _edge_arrays(graph)
    edge_idx = list(graph.edges).index(edge)

    beta_values = np.linspace(-1e1, 1e1, 5)
    social_cost_values = []

    for beta_value in beta_values:
        trial_beta = beta_arr.copy()
        trial_beta[edge_idx] = beta_value
        social_cost_values.append(
            _social_cost_from_vecs(graph, alpha_arr, trial_beta, np.asarray(demands))
        )

    model = LinearRegression()
    model.fit(beta_values.reshape(-1, 1), np.asarray(social_cost_values))
    return float(model.coef_[0])
