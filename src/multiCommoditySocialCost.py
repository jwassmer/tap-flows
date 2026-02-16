"""Social-cost-gradient tooling for multicommodity TAP models.

This module keeps the original public API but consolidates it into a smaller,
well-documented implementation suitable for figure-generation workflows.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cvxpy as cp
import networkx as nx
import numpy as np
import osmnx as ox
from scipy.sparse import bmat, block_diag, csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky

from src import SocialCost as sc
from src import multiCommodityTAP as mc
from src import osmGraphs as og

np.set_printoptions(precision=3, suppress=True)



def _as_flow_matrix(f_mat: np.ndarray, num_edges: int) -> np.ndarray:
    """Normalize flow input to shape ``(num_commodities, num_edges)``."""
    flow_matrix = np.asarray(f_mat, dtype=float)
    if flow_matrix.ndim == 1:
        flow_matrix = flow_matrix.reshape(1, -1)
    if flow_matrix.ndim != 2:
        raise ValueError(f"Expected 1D/2D flow array, got shape {flow_matrix.shape}.")
    if flow_matrix.shape[1] != num_edges:
        raise ValueError(
            f"Flow matrix has {flow_matrix.shape[1]} columns, expected {num_edges}."
        )
    return flow_matrix



def _as_demand_matrix(od_matrix: Iterable[np.ndarray], num_nodes: int) -> np.ndarray:
    """Normalize demand input to shape ``(num_commodities, num_nodes)``."""
    demands = np.asarray(od_matrix, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(1, -1)
    if demands.ndim != 2:
        raise ValueError(f"Expected 1D/2D demand array, got shape {demands.shape}.")
    if demands.shape[1] != num_nodes:
        raise ValueError(
            f"Demand matrix has {demands.shape[1]} columns, expected {num_nodes}."
        )
    return demands



def _flow_support_mask(
    flow_or_mask: np.ndarray,
    num_edges: int,
    eps: float,
) -> np.ndarray:
    """Return a boolean support mask with shape ``(num_commodities, num_edges)``."""
    arr = np.asarray(flow_or_mask)

    if arr.ndim == 2 and arr.shape[1] == num_edges:
        if arr.dtype == bool:
            return arr
        return arr > eps

    if arr.ndim == 1 and arr.size % num_edges == 0:
        if arr.dtype == bool:
            return arr.reshape(-1, num_edges)
        return arr.reshape(-1, num_edges) > eps

    raise ValueError(
        f"Could not interpret mask/flow input with shape {arr.shape} for {num_edges} edges."
    )



def _layered_edge_incidence_from_masks(
    graph: nx.DiGraph,
    node_mask: np.ndarray,
    edge_mask_flat: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray]:
    """Build layered incidence matrix and remove disconnected node rows."""
    num_edges = graph.number_of_edges()
    num_layers = edge_mask_flat.size // num_edges

    incidence = -nx.incidence_matrix(graph, oriented=True)
    layered_incidence = block_diag([incidence] * num_layers, format="csr")
    layered_incidence = layered_incidence[:, edge_mask_flat]

    connected_rows = np.diff(layered_incidence.indptr) > 0
    updated_node_mask = np.asarray(node_mask, dtype=bool).copy()
    updated_node_mask &= connected_rows

    layered_incidence = layered_incidence[updated_node_mask]
    return layered_incidence.astype(float), updated_node_mask



def _legacy_layered_edge_incidence_matrix(
    graph: nx.DiGraph,
    f_mat: np.ndarray,
    eps: float,
) -> csr_matrix:
    """Backward-compatible incidence matrix helper for old exploratory scripts."""
    num_edges = graph.number_of_edges()
    edge_mask = _flow_support_mask(f_mat, num_edges=num_edges, eps=eps).reshape(-1)

    incidence = -nx.incidence_matrix(graph, oriented=True)
    num_layers = edge_mask.size // num_edges
    layered_incidence = block_diag([incidence] * num_layers, format="csr")[:, edge_mask]

    non_zero_rows = np.diff(layered_incidence.indptr) > 0
    non_zero_cols = np.diff(layered_incidence.tocsc().indptr) > 0
    return layered_incidence[non_zero_rows][:, non_zero_cols].astype(float)



def flow_mask(f_mat: np.ndarray, eps: float = 1e-1) -> csr_matrix:
    """Return legacy sparse support mask used by older coupling implementations."""
    flow_matrix = np.asarray(f_mat, dtype=float)
    if flow_matrix.ndim != 2:
        raise ValueError("flow_mask expects a 2D flow matrix.")

    binary = (flow_matrix > eps).astype(int)
    num_commodities, _ = binary.shape

    blocks = [
        [
            diags(binary[i] * binary[j], format="csr")
            for j in range(num_commodities)
        ]
        for i in range(num_commodities)
    ]
    return bmat(blocks, format="csr")



def generate_coupling_matrix(
    graph: nx.DiGraph,
    edge_mask_or_flows: np.ndarray,
    eps: float = 1e-1,
) -> csr_matrix:
    """Return the reduced coupling matrix on active layered edges."""
    num_edges = graph.number_of_edges()
    alpha = np.array(list(nx.get_edge_attributes(graph, "alpha").values()), dtype=float)
    edge_mask_stacked = _flow_support_mask(edge_mask_or_flows, num_edges=num_edges, eps=eps)

    num_layers = edge_mask_stacked.shape[0]
    full_size = num_layers * num_edges
    active_flat_mask = edge_mask_stacked.reshape(-1)

    coupling = lil_matrix((full_size, full_size), dtype=float)
    for edge_idx in range(num_edges):
        active_layers = np.flatnonzero(edge_mask_stacked[:, edge_idx])
        if active_layers.size == 0:
            continue

        value = alpha[edge_idx]
        for i in active_layers:
            row = i * num_edges + edge_idx
            for j in active_layers:
                col = j * num_edges + edge_idx
                coupling[row, col] = value

    reduced = coupling.tocsr()[active_flat_mask][:, active_flat_mask]
    return reduced



def get_kappa_x_delta(
    graph: nx.DiGraph,
    stacked_edge_mask: np.ndarray,
    delta: float = 1e-2,
) -> csr_matrix:
    """Return ``delta * (K + delta I)^{-1}`` on active layered edges."""
    num_edges = graph.number_of_edges()
    alpha = np.array(list(nx.get_edge_attributes(graph, "alpha").values()), dtype=float)
    edge_mask_stacked = _flow_support_mask(
        stacked_edge_mask,
        num_edges=num_edges,
        eps=0.0,
    )

    num_layers = edge_mask_stacked.shape[0]
    full_size = num_layers * num_edges
    active_flat_mask = edge_mask_stacked.reshape(-1)

    rows_per_edge = {
        edge_idx: np.flatnonzero(edge_mask_stacked[:, edge_idx])
        for edge_idx in range(num_edges)
    }

    kappa = lil_matrix((full_size, full_size), dtype=float)
    for edge_idx, active_layers in rows_per_edge.items():
        n_active = active_layers.size
        if n_active == 0:
            continue

        denominator = delta + alpha[edge_idx] * n_active
        diagonal_value = 1 - alpha[edge_idx] / denominator
        off_diagonal_value = -alpha[edge_idx] / denominator

        for i in active_layers:
            row = i * num_edges + edge_idx
            for j in active_layers:
                col = j * num_edges + edge_idx
                if i == j:
                    kappa[row, col] = diagonal_value
                else:
                    kappa[row, col] = off_diagonal_value

    reduced = kappa.tocsr()[active_flat_mask][:, active_flat_mask]
    return reduced



def inverse_coupling_matrix(
    graph: nx.DiGraph,
    stacked_edge_mask: np.ndarray,
    delta: float = 1e-2,
) -> csr_matrix:
    """Return ``(K + delta I)^{-1}`` on active layered edges."""
    return get_kappa_x_delta(graph, stacked_edge_mask, delta=delta) / delta



def layered_edge_incidence_matrix(
    graph: nx.DiGraph,
    node_mask_or_flows: np.ndarray,
    edge_mask: np.ndarray | None = None,
    eps: float = 1e-1,
):
    """Build layered incidence matrices.

    Supported call styles:
    - New: ``layered_edge_incidence_matrix(graph, node_mask, edge_mask)``
      where both masks are boolean arrays.
    - Legacy: ``layered_edge_incidence_matrix(graph, f_mat, eps=...)``
      where ``f_mat`` is a flow matrix. Returns only the reduced matrix.
    """
    if edge_mask is None:
        return _legacy_layered_edge_incidence_matrix(graph, node_mask_or_flows, eps=eps)

    edge_mask_flat = np.asarray(edge_mask, dtype=bool)
    node_mask = np.asarray(node_mask_or_flows, dtype=bool)
    return _layered_edge_incidence_from_masks(graph, node_mask, edge_mask_flat)



def derivative_social_cost(
    graph: nx.DiGraph,
    f_mat: np.ndarray,
    od_matrix: np.ndarray,
    eps: float = 1e-3,
    reg: float = 1e-5,
    demands_to_sinks: bool = True,
):
    """Compute social-cost gradients w.r.t. free-flow travel times ``beta_e``."""
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    flow_matrix = _as_flow_matrix(f_mat, num_edges=num_edges)
    demand_matrix = _as_demand_matrix(od_matrix, num_nodes=num_nodes)

    if flow_matrix.shape[0] != demand_matrix.shape[0]:
        raise ValueError(
            "Number of commodities in flow and demand matrices must match. "
            f"Got {flow_matrix.shape[0]} and {demand_matrix.shape[0]}."
        )

    num_layers = flow_matrix.shape[0]
    edge_mask_stacked = flow_matrix > eps
    edge_mask_flat = edge_mask_stacked.reshape(-1)

    p_vec = demand_matrix.reshape(-1)
    if demands_to_sinks:
        removed_nodes = np.where(p_vec > 0)[0]
    else:
        removed_nodes = np.where(p_vec < 0)[0]

    node_mask = np.ones(num_layers * num_nodes, dtype=bool)
    node_mask[removed_nodes] = False

    incidence, node_mask = _layered_edge_incidence_from_masks(
        graph,
        node_mask=node_mask,
        edge_mask_flat=edge_mask_flat,
    )

    gradients = {edge: 0.0 for edge in graph.edges}
    if incidence.shape[1] == 0:
        return gradients

    kappa_x_delta = get_kappa_x_delta(graph, edge_mask_stacked, delta=reg)
    mll_x_delta = incidence @ kappa_x_delta @ incidence.T
    mll_x_delta = mll_x_delta.tocsc()
    p_filtered = p_vec[node_mask]

    try:
        solve = cholesky(mll_x_delta).solve_A
        phi = solve(p_filtered)
    except Exception:
        phi = spsolve(mll_x_delta, p_filtered)

    active_slopes = np.asarray(phi @ incidence @ kappa_x_delta).reshape(-1)
    active_indices = np.flatnonzero(edge_mask_flat)

    edge_order = list(graph.edges)
    for active_idx, slope in zip(active_indices, active_slopes):
        edge = edge_order[active_idx % num_edges]
        gradients[edge] += float(slope)

    return gradients



def compute_social_cost_gradient(*args, **kwargs):
    """Named alias for :func:`derivative_social_cost`."""
    return derivative_social_cost(*args, **kwargs)



def _derivative_social_cost_and_flow(
    graph: nx.DiGraph,
    gamma: float = 0.02,
    num_sources: int | str = "all",
    eps: float = 1e-3,
    **kwargs,
):
    """Solve multicommodity TAP, then write flow and SCGC back to graph attributes."""
    nodes, _ = ox.graph_to_gdfs(graph)

    if num_sources == "all":
        num_sources = graph.number_of_nodes()

    selected_nodes = og.select_evenly_distributed_nodes(nodes, num_sources)
    nodes["source"] = nodes.index.isin(selected_nodes.index)
    nx.set_node_attributes(graph, nodes["source"], "source")

    demands = og.demand_list(nodes, commodity=selected_nodes, gamma=gamma)

    solver_kwargs = dict(kwargs)
    solver_kwargs.setdefault("solver", cp.OSQP)
    solver_kwargs.setdefault("return_fw", True)
    solver_kwargs.setdefault("pos_flows", True)

    f_mat, _ = mc.solve_multicommodity_tap(graph, demands, **solver_kwargs)
    total_flow = np.sum(f_mat, axis=0)
    nx.set_edge_attributes(graph, dict(zip(graph.edges, total_flow)), "flow")

    gradients = derivative_social_cost(graph, f_mat, demands, eps=eps)
    nx.set_edge_attributes(graph, gradients, "derivative_social_cost")
    return gradients



def numerical_derivative(
    graph: nx.DiGraph,
    od_matrix: np.ndarray,
    edge,
    num: int = 25,
    var_percentage: float = 0.1,
    show_progress: bool = False,
    **kwargs,
):
    """Finite-difference social-cost derivative for one edge's ``beta`` parameter."""
    edge_order = list(graph.edges)
    if edge not in edge_order:
        raise ValueError(f"Edge {edge} not in graph.")

    beta = np.array(list(nx.get_edge_attributes(graph, "beta").values()), dtype=float)
    edge_idx = edge_order.index(edge)
    beta_center = beta[edge_idx]
    beta_span = abs(beta_center) * var_percentage

    beta_values = np.linspace(beta_center - beta_span, beta_center + beta_span, num)

    solver = kwargs.pop("solver", cp.OSQP)
    social_cost_values = []

    iterator = beta_values
    if show_progress:
        try:
            import tqdm

            iterator = tqdm.tqdm(beta_values)
        except Exception:
            iterator = beta_values

    for beta_value in iterator:
        beta_trial = beta.copy()
        beta_trial[edge_idx] = beta_value

        flow = mc.solve_multicommodity_tap(
            graph,
            od_matrix,
            pos_flows=True,
            beta=beta_trial,
            solver=solver,
            **kwargs,
        )
        social_cost_values.append(sc.total_social_cost(graph, flow, beta=beta_trial))

    slopes = np.gradient(social_cost_values, beta_values)
    return np.asarray(slopes), np.asarray(beta_values), np.asarray(social_cost_values)



def sample_social_cost_sensitivity(*args, **kwargs):
    """Named alias for :func:`numerical_derivative`."""
    return numerical_derivative(*args, **kwargs)



def test_system(
    graph: nx.DiGraph,
    f_mat: np.ndarray,
    lambda_mat: np.ndarray,
    od_matrix: np.ndarray,
    eps: float = 1e-2,
):
    """Legacy debugging helper that assembles the reduced KKT system."""
    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    flow_matrix = _as_flow_matrix(f_mat, num_edges=num_edges)
    demand_matrix = _as_demand_matrix(od_matrix, num_nodes=num_nodes)

    edge_mask_stacked = flow_matrix > eps
    edge_mask_flat = edge_mask_stacked.reshape(-1)

    node_mask = np.ones(flow_matrix.shape[0] * num_nodes, dtype=bool)
    incidence, node_mask = _layered_edge_incidence_from_masks(
        graph,
        node_mask=node_mask,
        edge_mask_flat=edge_mask_flat,
    )

    coupling = generate_coupling_matrix(graph, edge_mask_stacked)
    zeros = csr_matrix((incidence.shape[0], incidence.shape[0]))
    M = bmat([[coupling, incidence.T], [incidence, zeros]], format="csr")

    beta = np.array(list(nx.get_edge_attributes(graph, "beta").values()), dtype=float)
    beta_vec = np.tile(beta, flow_matrix.shape[0])
    f_vec = flow_matrix.reshape(-1)
    lambda_vec = np.asarray(lambda_mat, dtype=float).reshape(-1)[node_mask]
    p_vec = demand_matrix.reshape(-1)[node_mask]

    x = np.hstack([f_vec[edge_mask_flat], lambda_vec])
    y = np.hstack([-beta_vec[edge_mask_flat], p_vec])
    return M, x, y


# Backward-compatible aliases kept intentionally.
derivative_social_cost_OLD = derivative_social_cost
generate_coupling_matrix_OLD = generate_coupling_matrix
layered_edge_incidence_matrix_OLD = layered_edge_incidence_matrix


# Graph method helpers for prior scripts.
nx.DiGraph.derivative_social_cost = _derivative_social_cost_and_flow
nx.MultiDiGraph.derivative_social_cost = _derivative_social_cost_and_flow
