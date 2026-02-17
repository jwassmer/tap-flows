"""Predict first active-set breakpoints from reduced KKT linearization.

For each selected edge parameter ``beta_k``, this script computes:
1) A raw first-order predicted local perturbation radius before active-set change.
2) A v3 seeded-refinement prediction (raw seed + short stability search).
3) An observed radius from brute-force re-solves over a delta grid or bisection.

The raw prediction uses first-order sensitivities on the fixed baseline active set.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import block_diag, eye
from scipy.sparse.linalg import splu

# Allow direct execution via `python analysis/<script>.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import Graphs as gr
from src import multiCommoditySocialCost as msc
from src import multiCommodityTAP as mc
from src.figure_style import apply_publication_style
from src.paper_examples import build_classic_braess_validation_graph

OUTPUT_FIGURE = "figs/active-set-breakpoint-prediction.pdf"
OUTPUT_TABLE = "cache/active-set-breakpoint-prediction.csv"


@dataclass
class BaselineSystem:
    graph: nx.DiGraph
    edges: list[tuple]
    demands: np.ndarray
    flow_matrix: np.ndarray
    lambda_matrix: np.ndarray
    edge_mask_stacked: np.ndarray
    edge_mask_flat: np.ndarray
    active_flat_indices: np.ndarray
    node_mask: np.ndarray
    incidence_full: Any
    matrix_solver: Any
    n_active: int
    beta: np.ndarray
    alpha: np.ndarray
    total_flow: np.ndarray
    flow_flat: np.ndarray
    slack_flat: np.ndarray
    edge_ids_flat: np.ndarray
    eps_active: float


def _format_edge(edge: tuple) -> str:
    return f"{edge[0]}→{edge[1]}"


def _active_mask(flow_matrix: np.ndarray, eps_active: float) -> np.ndarray:
    flows = np.asarray(flow_matrix, dtype=float)
    if flows.ndim == 1:
        flows = flows.reshape(1, -1)
    return flows > eps_active


def _active_set_indicator(
    flow_matrix: np.ndarray,
    eps_active: float,
    level: str,
) -> np.ndarray:
    mask = _active_mask(flow_matrix, eps_active)
    if level == "commodity":
        return mask
    if level == "edge":
        return np.sum(np.asarray(flow_matrix, dtype=float), axis=0) > eps_active
    raise ValueError(f"Unknown active-set level '{level}'.")


def _compute_slack_flat(
    graph: nx.DiGraph,
    flow_matrix: np.ndarray,
    lambda_matrix: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Return stacked complementarity slacks for all commodity-edge entries."""
    flows = np.asarray(flow_matrix, dtype=float)
    if flows.ndim == 1:
        flows = flows.reshape(1, -1)
    lambdas = np.asarray(lambda_matrix, dtype=float)
    if lambdas.ndim == 1:
        lambdas = lambdas.reshape(1, -1)

    num_commodities, num_edges = flows.shape
    incidence_single = -nx.incidence_matrix(graph, oriented=True)
    incidence_full = block_diag([incidence_single] * num_commodities, format="csr")

    alpha = np.array([graph.edges[edge]["alpha"] for edge in graph.edges], dtype=float)
    edge_ids_flat = np.arange(num_commodities * num_edges, dtype=int) % num_edges
    alpha_flat = alpha[edge_ids_flat]

    total_flow = np.sum(flows, axis=0)
    total_flat = total_flow[edge_ids_flat]
    beta_flat = np.tile(np.asarray(beta, dtype=float), num_commodities)
    lambda_flat = lambdas.reshape(-1)
    potential_drop_flat = np.asarray(
        incidence_full.T @ lambda_flat, dtype=float
    ).reshape(-1)
    return alpha_flat * total_flat + beta_flat - potential_drop_flat


def _local_stability_radius(
    delta_values: np.ndarray, stable_flags: np.ndarray
) -> float:
    deltas = np.asarray(delta_values, dtype=float)
    stable = np.asarray(stable_flags, dtype=bool)
    zero_idx = int(np.argmin(np.abs(deltas)))
    if not stable[zero_idx]:
        return 0.0

    left = zero_idx
    while left - 1 >= 0 and stable[left - 1]:
        left -= 1

    right = zero_idx
    while right + 1 < stable.size and stable[right + 1]:
        right += 1

    return float(min(abs(deltas[left]), abs(deltas[right])))


def _select_edge_indices(
    baseline_edge_flow: np.ndarray,
    strategy: str,
    max_edges: int,
    seed: int,
) -> np.ndarray:
    num_edges = baseline_edge_flow.size
    all_indices = np.arange(num_edges, dtype=int)

    if strategy == "all" or max_edges >= num_edges:
        return all_indices

    if strategy == "top-flow":
        order = np.argsort(-baseline_edge_flow)
        return np.sort(order[:max_edges])

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(all_indices, size=max_edges, replace=False))

    raise ValueError(f"Unknown edge-selection strategy '{strategy}'.")


def _synthetic_population_demands(
    graph,
    num_commodities: int,
    gamma: float,
) -> np.ndarray:
    """Create source-to-all OD demands from node populations without GeoPandas."""
    node_ids = np.array(list(graph.nodes))
    num_nodes = node_ids.size
    if num_commodities > num_nodes:
        raise ValueError(
            f"num_commodities ({num_commodities}) must be <= number of nodes ({num_nodes})."
        )

    population_values = [graph.nodes[node].get("population", 1.0) for node in node_ids]
    populations = np.array(
        [
            float(pop if pop is not None and pop > 0 else 1.0)
            for pop in population_values
        ],
        dtype=float,
    )

    source_positions = np.linspace(0, num_nodes - 1, num_commodities, dtype=int)
    demands = np.zeros((num_commodities, num_nodes), dtype=float)

    for commodity_idx, source_idx in enumerate(source_positions):
        load = max(float(populations[source_idx] * gamma), 1e-9)
        demand = np.full(num_nodes, -load / (num_nodes - 1), dtype=float)
        demand[source_idx] = load
        demands[commodity_idx] = demand

    return demands


def _build_problem(args: argparse.Namespace):
    if args.network == "braess":
        graph = build_classic_braess_validation_graph()
        load = 9.5
        demand = (
            -np.ones(graph.number_of_nodes()) * load / (graph.number_of_nodes() - 1)
        )
        demand[0] = load
        demands = np.asarray([demand], dtype=float)
        return graph, demands

    graph = gr.random_planar_graph(
        n_nodes=args.num_nodes,
        seed=args.seed,
        alpha="random",
    )
    demands = _synthetic_population_demands(
        graph=graph,
        num_commodities=args.num_commodities,
        gamma=args.gamma,
    )
    return graph, demands


def _solve_flows_and_lambdas(
    graph: nx.DiGraph,
    demands: np.ndarray,
    beta: np.ndarray,
    solver,
    disagg_reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve UE TAP and return commodity flows and dual node multipliers.

    If ``disagg_reg > 0``, adds a tiny per-commodity quadratic regularizer to make
    commodity decomposition unique and numerically stable.
    """
    demand_matrix = np.asarray(demands, dtype=float)
    if demand_matrix.ndim == 1:
        demand_matrix = demand_matrix.reshape(1, -1)

    if disagg_reg <= 0:
        flow_matrix, lambda_matrix = mc.solve_multicommodity_tap(
            graph,
            demand_matrix,
            beta=beta,
            pos_flows=True,
            return_fw=True,
            solver=solver,
        )
        return np.asarray(flow_matrix, dtype=float), np.asarray(
            lambda_matrix, dtype=float
        )

    incidence = -nx.incidence_matrix(graph, oriented=True)
    num_edges = graph.number_of_edges()
    num_commodities = demand_matrix.shape[0]
    alpha = np.array([graph.edges[edge]["alpha"] for edge in graph.edges], dtype=float)

    flows = [cp.Variable(num_edges, nonneg=True) for _ in range(num_commodities)]
    constraints = [
        incidence @ flows[k] == demand_matrix[k] for k in range(num_commodities)
    ]

    if num_commodities > 1:
        flow_matrix_expr = cp.vstack(flows)
    else:
        flow_matrix_expr = cp.reshape(flows[0], (1, num_edges), order="F")
    total_flow = cp.sum(flow_matrix_expr, axis=0)

    objective = cp.sum(
        cp.multiply(0.5 * alpha, cp.square(total_flow)) + cp.multiply(beta, total_flow)
    )
    objective += disagg_reg * cp.sum(cp.square(flow_matrix_expr))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver)

    flow_matrix = np.array(
        [np.asarray(f.value).reshape(-1) for f in flows], dtype=float
    )
    lambda_matrix = np.array(
        [np.asarray(c.dual_value).reshape(-1) for c in constraints],
        dtype=float,
    )
    return flow_matrix, lambda_matrix


def _solve_flows(
    graph: nx.DiGraph,
    demands: np.ndarray,
    beta: np.ndarray,
    solver,
    disagg_reg: float,
) -> np.ndarray:
    flow_matrix, _ = _solve_flows_and_lambdas(
        graph=graph,
        demands=demands,
        beta=beta,
        solver=solver,
        disagg_reg=disagg_reg,
    )
    return np.asarray(flow_matrix, dtype=float)


def _build_baseline_system(
    graph: nx.DiGraph,
    demands: np.ndarray,
    solver,
    eps_active: float,
    regularization: float,
    disagg_reg: float,
):
    flow_matrix, lambda_matrix = _solve_flows_and_lambdas(
        graph=graph,
        demands=demands,
        beta=np.array([graph.edges[edge]["beta"] for edge in graph.edges], dtype=float),
        solver=solver,
        disagg_reg=disagg_reg,
    )
    flow_matrix = np.asarray(flow_matrix, dtype=float)
    lambda_matrix = np.asarray(lambda_matrix, dtype=float)

    num_commodities, num_edges = flow_matrix.shape
    num_nodes = graph.number_of_nodes()
    edges = list(graph.edges)

    edge_mask_stacked = _active_mask(flow_matrix, eps_active)
    edge_mask_flat = edge_mask_stacked.reshape(-1)
    active_flat_indices = np.flatnonzero(edge_mask_flat)

    p_vec = demands.reshape(-1)
    removed_nodes = np.where(p_vec > 0)[0]
    node_mask = np.ones(num_commodities * num_nodes, dtype=bool)
    node_mask[removed_nodes] = False

    incidence_reduced, node_mask = msc._layered_edge_incidence_from_masks(
        graph,
        node_mask=node_mask,
        edge_mask_flat=edge_mask_flat,
    )
    coupling = msc.generate_coupling_matrix(graph, edge_mask_stacked)
    # Match the Hessian used by the optional disaggregation-regularized solve.
    if disagg_reg > 0:
        coupling = coupling + (2.0 * disagg_reg) * eye(coupling.shape[0], format="csr")
    coupling = coupling + regularization * eye(coupling.shape[0], format="csr")
    zeros = msc.csr_matrix((incidence_reduced.shape[0], incidence_reduced.shape[0]))
    reduced_matrix = msc.bmat(
        [[coupling, incidence_reduced.T], [incidence_reduced, zeros]],
        format="csc",
    )
    matrix_solver = splu(reduced_matrix)

    incidence_single = -nx.incidence_matrix(graph, oriented=True)
    incidence_full = block_diag([incidence_single] * num_commodities, format="csr")

    alpha = np.array([graph.edges[edge]["alpha"] for edge in edges], dtype=float)
    beta = np.array([graph.edges[edge]["beta"] for edge in edges], dtype=float)
    total_flow = np.sum(flow_matrix, axis=0)
    flow_flat = flow_matrix.reshape(-1)

    edge_ids_flat = np.arange(num_commodities * num_edges, dtype=int) % num_edges
    alpha_flat = alpha[edge_ids_flat]
    beta_flat = np.tile(beta, num_commodities)

    lambda_flat = lambda_matrix.reshape(-1)
    potential_drop_flat = np.asarray(
        incidence_full.T @ lambda_flat, dtype=float
    ).reshape(-1)
    slack_flat = (
        alpha_flat * total_flow[edge_ids_flat] + beta_flat - potential_drop_flat
    )

    return BaselineSystem(
        graph=graph,
        edges=edges,
        demands=demands,
        flow_matrix=flow_matrix,
        lambda_matrix=lambda_matrix,
        edge_mask_stacked=edge_mask_stacked,
        edge_mask_flat=edge_mask_flat,
        active_flat_indices=active_flat_indices,
        node_mask=node_mask,
        incidence_full=incidence_full,
        matrix_solver=matrix_solver,
        n_active=int(active_flat_indices.size),
        beta=beta,
        alpha=alpha,
        total_flow=total_flow,
        flow_flat=flow_flat,
        slack_flat=slack_flat,
        edge_ids_flat=edge_ids_flat,
        eps_active=eps_active,
    )


def _solve_directional_sensitivity(
    baseline: BaselineSystem,
    edge_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return derivatives (w.r.t absolute beta change) of layered flow and slack."""
    rhs = np.zeros(
        baseline.n_active + int(np.count_nonzero(baseline.node_mask)), dtype=float
    )

    active_positions = np.arange(baseline.n_active)
    active_edge_ids = baseline.active_flat_indices % baseline.graph.number_of_edges()
    affected = active_positions[active_edge_ids == edge_idx]
    rhs[affected] = -1.0

    x_dot = baseline.matrix_solver.solve(rhs)
    df_active = x_dot[: baseline.n_active]
    dlambda_reduced = x_dot[baseline.n_active :]

    num_commodities, num_edges = baseline.flow_matrix.shape
    num_nodes = baseline.graph.number_of_nodes()

    df_flat = np.zeros(num_commodities * num_edges, dtype=float)
    df_flat[baseline.active_flat_indices] = df_active

    d_total_flow = np.sum(df_flat.reshape(num_commodities, num_edges), axis=0)
    d_total_flat = d_total_flow[baseline.edge_ids_flat]

    dlambda_full = np.zeros(num_commodities * num_nodes, dtype=float)
    dlambda_full[baseline.node_mask] = dlambda_reduced
    d_potential_drop = np.asarray(
        baseline.incidence_full.T @ dlambda_full,
        dtype=float,
    ).reshape(-1)

    alpha_flat = baseline.alpha[baseline.edge_ids_flat]
    d_beta_unit = (baseline.edge_ids_flat == edge_idx).astype(float)
    d_slack_flat = alpha_flat * d_total_flat + d_beta_unit - d_potential_drop

    return df_flat, d_slack_flat


def _first_breakpoints(
    baseline: BaselineSystem,
    edge_idx: int,
    tol: float,
    active_set_level: str,
    return_meta: bool = False,
):
    """Predict first active-set breakpoints for positive/negative absolute beta steps."""
    df_flat, d_slack_flat = _solve_directional_sensitivity(baseline, edge_idx)

    eta_plus_candidates = []
    eta_minus_candidates = []
    event_plus = "none"
    event_minus = "none"

    flow_threshold = baseline.eps_active
    if active_set_level == "commodity":
        active_mask = baseline.edge_mask_flat
        inactive_mask = ~active_mask

        # Active entries leaving support.
        f0_active = baseline.flow_flat[active_mask]
        df_active = df_flat[active_mask]

        for f0, df in zip(f0_active, df_active):
            if df < -tol:
                eta = (flow_threshold - f0) / df
                if eta > tol:
                    eta_plus_candidates.append(float(eta))
            elif df > tol:
                eta = (flow_threshold - f0) / df
                if eta < -tol:
                    eta_minus_candidates.append(float(-eta))

        # Inactive entries becoming active via reduced-cost slack crossing zero.
        s0_inactive = baseline.slack_flat[inactive_mask]
        ds_inactive = d_slack_flat[inactive_mask]

        for s0, ds in zip(s0_inactive, ds_inactive):
            if s0 <= tol:
                continue
            if ds < -tol:
                eta = -s0 / ds
                if eta > tol:
                    eta_plus_candidates.append(float(eta))
            elif ds > tol:
                eta = -s0 / ds
                if eta < -tol:
                    eta_minus_candidates.append(float(-eta))

        eta_plus_sorted = np.sort(np.asarray(eta_plus_candidates, dtype=float))
        eta_minus_sorted = np.sort(np.asarray(eta_minus_candidates, dtype=float))
        eta_plus = float(eta_plus_sorted[0]) if eta_plus_sorted.size else np.inf
        eta_minus = float(eta_minus_sorted[0]) if eta_minus_sorted.size else np.inf

        if np.isfinite(eta_plus):
            first_active_leave_plus = np.any(
                np.isclose(
                    (flow_threshold - f0_active[df_active < -tol])
                    / df_active[df_active < -tol],
                    eta_plus,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )
            event_plus = "leave" if first_active_leave_plus else "enter"
        if np.isfinite(eta_minus):
            first_active_leave_minus = np.any(
                np.isclose(
                    -(
                        (flow_threshold - f0_active[df_active > tol])
                        / df_active[df_active > tol]
                    ),
                    eta_minus,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )
            event_minus = "leave" if first_active_leave_minus else "enter"
        if not return_meta:
            return eta_minus, eta_plus, event_minus, event_plus

        gap_plus = (
            float(eta_plus_sorted[1] / eta_plus_sorted[0])
            if eta_plus_sorted.size >= 2 and eta_plus_sorted[0] > 0
            else np.inf
        )
        gap_minus = (
            float(eta_minus_sorted[1] / eta_minus_sorted[0])
            if eta_minus_sorted.size >= 2 and eta_minus_sorted[0] > 0
            else np.inf
        )
        meta = {
            "n_plus_candidates": int(eta_plus_sorted.size),
            "n_minus_candidates": int(eta_minus_sorted.size),
            "gap_plus": gap_plus,
            "gap_minus": gap_minus,
            "gap_sym": float(min(gap_plus, gap_minus)),
            "active_level": active_set_level,
        }
        return eta_minus, eta_plus, event_minus, event_plus, meta

    if active_set_level != "edge":
        raise ValueError(f"Unknown active-set level '{active_set_level}'.")

    num_commodities, num_edges = baseline.flow_matrix.shape
    flows0 = baseline.flow_flat.reshape(num_commodities, num_edges)
    dfs = df_flat.reshape(num_commodities, num_edges)
    slacks0 = baseline.slack_flat.reshape(num_commodities, num_edges)
    dslacks = d_slack_flat.reshape(num_commodities, num_edges)

    total0 = np.sum(flows0, axis=0)
    dtotal = np.sum(dfs, axis=0)
    edge_active = total0 > flow_threshold

    # Active edge leaves when total flow drops below threshold.
    for edge_j in np.where(edge_active)[0]:
        dt = dtotal[edge_j]
        if dt < -tol:
            eta = (flow_threshold - total0[edge_j]) / dt
            if eta > tol:
                eta_plus_candidates.append(float(eta))
        elif dt > tol:
            eta = (flow_threshold - total0[edge_j]) / dt
            if eta < -tol:
                eta_minus_candidates.append(float(-eta))

    # Inactive edge enters when any commodity slack on that edge crosses zero.
    for edge_j in np.where(~edge_active)[0]:
        s0_vec = slacks0[:, edge_j]
        ds_vec = dslacks[:, edge_j]

        plus = [
            float(-s0 / ds)
            for s0, ds in zip(s0_vec, ds_vec)
            if s0 > tol and ds < -tol and (-s0 / ds) > tol
        ]
        minus = [
            float(-(-s0 / ds))
            for s0, ds in zip(s0_vec, ds_vec)
            if s0 > tol and ds > tol and (-s0 / ds) < -tol
        ]
        if plus:
            eta_plus_candidates.append(min(plus))
        if minus:
            eta_minus_candidates.append(min(minus))

    eta_plus_sorted = np.sort(np.asarray(eta_plus_candidates, dtype=float))
    eta_minus_sorted = np.sort(np.asarray(eta_minus_candidates, dtype=float))
    eta_plus = float(eta_plus_sorted[0]) if eta_plus_sorted.size else np.inf
    eta_minus = float(eta_minus_sorted[0]) if eta_minus_sorted.size else np.inf

    if np.isfinite(eta_plus):
        event_plus = "leave/enter"
    if np.isfinite(eta_minus):
        event_minus = "leave/enter"

    if not return_meta:
        return eta_minus, eta_plus, event_minus, event_plus

    gap_plus = (
        float(eta_plus_sorted[1] / eta_plus_sorted[0])
        if eta_plus_sorted.size >= 2 and eta_plus_sorted[0] > 0
        else np.inf
    )
    gap_minus = (
        float(eta_minus_sorted[1] / eta_minus_sorted[0])
        if eta_minus_sorted.size >= 2 and eta_minus_sorted[0] > 0
        else np.inf
    )
    meta = {
        "n_plus_candidates": int(eta_plus_sorted.size),
        "n_minus_candidates": int(eta_minus_sorted.size),
        "gap_plus": gap_plus,
        "gap_minus": gap_minus,
        "gap_sym": float(min(gap_plus, gap_minus)),
        "active_level": active_set_level,
    }
    return eta_minus, eta_plus, event_minus, event_plus, meta


def _is_active_set_stable(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    delta_abs: float,
    direction: int,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
) -> bool:
    """Return whether active set matches baseline after signed perturbation."""
    beta_trial = baseline.beta.copy()
    beta_k = baseline.beta[edge_idx]
    beta_trial[edge_idx] = max(min_beta, beta_k + direction * delta_abs)

    flow_trial = _solve_flows(
        graph=baseline.graph,
        demands=baseline.demands,
        beta=beta_trial,
        solver=solver,
        disagg_reg=disagg_reg,
    )
    trial_indicator = _active_set_indicator(
        flow_trial,
        eps_active=baseline.eps_active,
        level=active_set_level,
    )
    return bool(np.array_equal(trial_indicator, baseline_indicator))


def _observed_breakpoint_bisection(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    delta_limit: float,
    bisect_iters: int,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
) -> tuple[float, float, bool, bool]:
    """Return plus/minus observed radii and censoring flags from bisection."""
    beta_k = float(baseline.beta[edge_idx])

    def boundary(direction: int) -> tuple[float, bool]:
        if beta_k <= min_beta:
            return 0.0, False

        delta_abs_limit = delta_limit * beta_k
        stable_at_limit = _is_active_set_stable(
            baseline=baseline,
            baseline_indicator=baseline_indicator,
            edge_idx=edge_idx,
            delta_abs=delta_abs_limit,
            direction=direction,
            solver=solver,
            min_beta=min_beta,
            active_set_level=active_set_level,
            disagg_reg=disagg_reg,
        )
        if stable_at_limit:
            return delta_limit, True

        lo = 0.0
        hi = delta_abs_limit
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            if _is_active_set_stable(
                baseline=baseline,
                baseline_indicator=baseline_indicator,
                edge_idx=edge_idx,
                delta_abs=mid,
                direction=direction,
                solver=solver,
                min_beta=min_beta,
                active_set_level=active_set_level,
                disagg_reg=disagg_reg,
            ):
                lo = mid
            else:
                hi = mid
        return lo / beta_k, False

    radius_minus, cens_minus = boundary(direction=-1)
    radius_plus, cens_plus = boundary(direction=+1)
    return radius_minus, radius_plus, cens_minus, cens_plus


def _observed_breakpoint_grid(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    delta_values: np.ndarray,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
) -> tuple[float, float, bool, bool]:
    """Return plus/minus radii from a discrete delta scan."""
    stable_flags = np.zeros(delta_values.size, dtype=bool)

    beta_k = float(baseline.beta[edge_idx])
    for delta_idx, delta in enumerate(delta_values):
        delta_abs = abs(delta) * beta_k
        direction = -1 if delta < 0 else 1
        stable_flags[delta_idx] = _is_active_set_stable(
            baseline=baseline,
            baseline_indicator=baseline_indicator,
            edge_idx=edge_idx,
            delta_abs=delta_abs,
            direction=direction,
            solver=solver,
            min_beta=min_beta,
            active_set_level=active_set_level,
            disagg_reg=disagg_reg,
        )

    neg_idx = np.where(delta_values <= 0)[0]
    pos_idx = np.where(delta_values >= 0)[0]

    radius_minus = _local_stability_radius(
        np.abs(delta_values[neg_idx][::-1]),
        stable_flags[neg_idx][::-1],
    )
    radius_plus = _local_stability_radius(
        np.abs(delta_values[pos_idx]),
        stable_flags[pos_idx],
    )

    cens_minus = bool(np.all(stable_flags[neg_idx]))
    cens_plus = bool(np.all(stable_flags[pos_idx]))
    return float(radius_minus), float(radius_plus), cens_minus, cens_plus


def _boundary_from_seed(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    direction: int,
    delta_limit: float,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
    seed_rel: float | None,
    refine_iters: int,
    expand_factor: float,
    max_expand_iters: int,
) -> tuple[float, bool, int]:
    """Find boundary radius using a seeded bracket + short bisection."""
    beta_k = float(baseline.beta[edge_idx])
    if beta_k <= min_beta:
        return 0.0, False, 0

    delta_abs_limit = delta_limit * beta_k
    eval_count = 0

    def is_stable(delta_abs: float) -> bool:
        nonlocal eval_count
        eval_count += 1
        return _is_active_set_stable(
            baseline=baseline,
            baseline_indicator=baseline_indicator,
            edge_idx=edge_idx,
            delta_abs=float(delta_abs),
            direction=direction,
            solver=solver,
            min_beta=min_beta,
            active_set_level=active_set_level,
            disagg_reg=disagg_reg,
        )

    if seed_rel is None or not np.isfinite(seed_rel) or seed_rel <= 0:
        probe_abs = min(0.2 * delta_abs_limit, delta_abs_limit)
    else:
        probe_abs = min(max(seed_rel * beta_k, 1e-12), delta_abs_limit)

    if probe_abs <= 0:
        probe_abs = min(0.2 * delta_abs_limit, delta_abs_limit)

    if not is_stable(probe_abs):
        lo = 0.0
        hi = probe_abs
    else:
        lo = probe_abs
        if np.isclose(lo, delta_abs_limit):
            return delta_limit, True, eval_count

        hi = min(
            delta_abs_limit,
            max(lo * expand_factor, lo + 0.02 * delta_abs_limit, 1e-12),
        )
        for _ in range(max_expand_iters):
            if not is_stable(hi):
                break
            lo = hi
            if np.isclose(hi, delta_abs_limit):
                return delta_limit, True, eval_count
            hi = min(
                delta_abs_limit,
                max(hi * expand_factor, hi + 0.02 * delta_abs_limit),
            )
        else:
            if is_stable(delta_abs_limit):
                return delta_limit, True, eval_count
            hi = delta_abs_limit

    for _ in range(refine_iters):
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid

    return lo / beta_k, False, eval_count


def _predicted_breakpoint_v3(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    predicted_minus_rel: float,
    predicted_plus_rel: float,
    delta_limit: float,
    solver,
    min_beta: float,
    active_set_level: str,
    disagg_reg: float,
    refine_iters: int,
    expand_factor: float,
    max_expand_iters: int,
) -> tuple[float, float, bool, bool, int]:
    """Refine raw predictions by seeded boundary search in each direction."""
    radius_minus, cens_minus, eval_minus = _boundary_from_seed(
        baseline=baseline,
        baseline_indicator=baseline_indicator,
        edge_idx=edge_idx,
        direction=-1,
        delta_limit=delta_limit,
        solver=solver,
        min_beta=min_beta,
        active_set_level=active_set_level,
        disagg_reg=disagg_reg,
        seed_rel=predicted_minus_rel,
        refine_iters=refine_iters,
        expand_factor=expand_factor,
        max_expand_iters=max_expand_iters,
    )
    radius_plus, cens_plus, eval_plus = _boundary_from_seed(
        baseline=baseline,
        baseline_indicator=baseline_indicator,
        edge_idx=edge_idx,
        direction=+1,
        delta_limit=delta_limit,
        solver=solver,
        min_beta=min_beta,
        active_set_level=active_set_level,
        disagg_reg=disagg_reg,
        seed_rel=predicted_plus_rel,
        refine_iters=refine_iters,
        expand_factor=expand_factor,
        max_expand_iters=max_expand_iters,
    )
    return radius_minus, radius_plus, cens_minus, cens_plus, int(eval_minus + eval_plus)


def _verify_directional_sensitivity(
    baseline: BaselineSystem,
    edge_idx: int,
    solver,
    min_beta: float,
    step_rel: float,
    active_set_level: str,
    disagg_reg: float,
) -> dict | None:
    """Validate derivative math by central finite differences around baseline."""
    beta_k = float(baseline.beta[edge_idx])
    step_abs = max(step_rel * max(beta_k, 1e-12), 1e-8)
    if beta_k - step_abs < min_beta:
        step_abs = max(beta_k - min_beta, 1e-10)
    if step_abs <= 0:
        return None

    beta_plus = baseline.beta.copy()
    beta_minus = baseline.beta.copy()
    beta_plus[edge_idx] = beta_k + step_abs
    beta_minus[edge_idx] = max(min_beta, beta_k - step_abs)
    actual_step = 0.5 * (beta_plus[edge_idx] - beta_minus[edge_idx])
    if actual_step <= 0:
        return None

    flow_plus, lambda_plus = _solve_flows_and_lambdas(
        graph=baseline.graph,
        demands=baseline.demands,
        beta=beta_plus,
        solver=solver,
        disagg_reg=disagg_reg,
    )
    flow_minus, lambda_minus = _solve_flows_and_lambdas(
        graph=baseline.graph,
        demands=baseline.demands,
        beta=beta_minus,
        solver=solver,
        disagg_reg=disagg_reg,
    )
    flow_plus = np.asarray(flow_plus, dtype=float)
    flow_minus = np.asarray(flow_minus, dtype=float)
    lambda_plus = np.asarray(lambda_plus, dtype=float)
    lambda_minus = np.asarray(lambda_minus, dtype=float)

    baseline_indicator = _active_set_indicator(
        baseline.flow_matrix,
        eps_active=baseline.eps_active,
        level=active_set_level,
    )
    if not np.array_equal(
        _active_set_indicator(flow_plus, baseline.eps_active, active_set_level),
        baseline_indicator,
    ):
        return None
    if not np.array_equal(
        _active_set_indicator(flow_minus, baseline.eps_active, active_set_level),
        baseline_indicator,
    ):
        return None

    df_model, dslack_model = _solve_directional_sensitivity(baseline, edge_idx=edge_idx)
    df_fd = (flow_plus.reshape(-1) - flow_minus.reshape(-1)) / (2.0 * actual_step)

    slack_plus = _compute_slack_flat(
        baseline.graph,
        flow_plus,
        lambda_plus,
        beta_plus,
    )
    slack_minus = _compute_slack_flat(
        baseline.graph,
        flow_minus,
        lambda_minus,
        beta_minus,
    )
    dslack_fd = (slack_plus - slack_minus) / (2.0 * actual_step)

    active_mask = baseline.edge_mask_flat
    inactive_mask = ~active_mask

    flow_model_active = df_model[active_mask]
    flow_fd_active = df_fd[active_mask]
    flow_scale_active = np.maximum(np.abs(flow_model_active), np.abs(flow_fd_active))
    flow_support = flow_scale_active > 1e-4
    if np.any(flow_support):
        flow_num = np.linalg.norm(
            flow_model_active[flow_support] - flow_fd_active[flow_support]
        )
        flow_den = max(
            np.linalg.norm(flow_fd_active[flow_support]),
            np.linalg.norm(flow_model_active[flow_support]),
            1e-10,
        )
    else:
        flow_num = np.linalg.norm(flow_model_active - flow_fd_active)
        flow_den = max(
            np.linalg.norm(flow_fd_active),
            np.linalg.norm(flow_model_active),
            1e-10,
        )

    slack_model_inactive = dslack_model[inactive_mask]
    slack_fd_inactive = dslack_fd[inactive_mask]
    slack_scale_inactive = np.maximum(
        np.abs(slack_model_inactive), np.abs(slack_fd_inactive)
    )
    slack_support = slack_scale_inactive > 1e-4
    if np.any(slack_support):
        slack_num = np.linalg.norm(
            slack_model_inactive[slack_support] - slack_fd_inactive[slack_support]
        )
        slack_den = max(
            np.linalg.norm(slack_fd_inactive[slack_support]),
            np.linalg.norm(slack_model_inactive[slack_support]),
            1e-10,
        )
    else:
        slack_num = np.linalg.norm(slack_model_inactive - slack_fd_inactive)
        slack_den = max(
            np.linalg.norm(slack_fd_inactive),
            np.linalg.norm(slack_model_inactive),
            1e-10,
        )

    return {
        "edge_index": int(edge_idx),
        "flow_rel_error": float(flow_num / flow_den),
        "slack_rel_error": float(slack_num / slack_den),
        "flow_abs_error": float(flow_num),
        "slack_abs_error": float(slack_num),
        "flow_scale": float(flow_den),
        "slack_scale": float(slack_den),
        "flow_support_size": int(np.count_nonzero(flow_support)),
        "slack_support_size": int(np.count_nonzero(slack_support)),
        "step_abs": float(actual_step),
    }


def _write_table(
    output_path: str,
    rows: list[dict],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "trial",
                "seed",
                "edge_index",
                "edge",
                "beta",
                "baseline_edge_flow",
                "pred_raw_radius_minus_pct",
                "pred_raw_radius_plus_pct",
                "pred_raw_radius_sym_pct",
                "pred_raw_radius_sym_clipped_pct",
                "pred_v3_radius_minus_pct",
                "pred_v3_radius_plus_pct",
                "pred_v3_radius_sym_pct",
                "pred_v3_radius_sym_clipped_pct",
                "pred_used_radius_minus_pct",
                "pred_used_radius_plus_pct",
                "pred_used_radius_sym_pct",
                "pred_used_radius_sym_clipped_pct",
                "predictor_mode",
                "pred_gap_sym",
                "pred_n_plus_candidates",
                "pred_n_minus_candidates",
                "pred_winner_event",
                "obs_radius_minus_pct",
                "obs_radius_plus_pct",
                "obs_minus_censored",
                "obs_plus_censored",
                "obs_radius_pct",
                "obs_is_censored",
                "is_confident",
                "is_mismatch_raw",
                "is_mismatch_v3",
                "is_mismatch",
                "abs_error_raw_clipped_pct",
                "abs_error_v3_clipped_pct",
                "abs_error_used_clipped_pct",
                "first_event_minus",
                "first_event_plus",
                "v3_solver_evals",
            ]
        )
        for row in rows:
            obs = row["obs_radius_pct"]
            pred_raw_clip = row["pred_raw_radius_sym_clipped_pct"]
            pred_v3_clip = row["pred_v3_radius_sym_clipped_pct"]
            pred_used_clip = row["pred_used_radius_sym_clipped_pct"]
            abs_err_raw = abs(pred_raw_clip - obs)
            abs_err_v3 = abs(pred_v3_clip - obs)
            abs_err_used = abs(pred_used_clip - obs)
            writer.writerow(
                [
                    int(row["trial"]),
                    int(row["seed"]),
                    int(row["edge_index"]),
                    row["edge"],
                    float(row["beta"]),
                    float(row["baseline_edge_flow"]),
                    float(row["pred_raw_radius_minus_pct"]),
                    float(row["pred_raw_radius_plus_pct"]),
                    float(row["pred_raw_radius_sym_pct"]),
                    pred_raw_clip,
                    float(row["pred_v3_radius_minus_pct"]),
                    float(row["pred_v3_radius_plus_pct"]),
                    float(row["pred_v3_radius_sym_pct"]),
                    pred_v3_clip,
                    float(row["pred_used_radius_minus_pct"]),
                    float(row["pred_used_radius_plus_pct"]),
                    float(row["pred_used_radius_sym_pct"]),
                    pred_used_clip,
                    row["predictor_mode"],
                    float(row["pred_gap_sym"]),
                    int(row["pred_n_plus_candidates"]),
                    int(row["pred_n_minus_candidates"]),
                    row["pred_winner_event"],
                    float(row["obs_radius_minus_pct"]),
                    float(row["obs_radius_plus_pct"]),
                    int(bool(row["obs_minus_censored"])),
                    int(bool(row["obs_plus_censored"])),
                    obs,
                    int(bool(row["obs_is_censored"])),
                    int(bool(row["is_confident"])),
                    int(bool(row["is_mismatch_raw"])),
                    int(bool(row["is_mismatch_v3"])),
                    int(bool(row["is_mismatch"])),
                    abs_err_raw,
                    abs_err_v3,
                    abs_err_used,
                    row["first_event_minus"],
                    row["first_event_plus"],
                    int(row["v3_solver_evals"]),
                ]
            )


def _plot_results(
    output_figure: str,
    observed_minus_pct: np.ndarray,
    observed_plus_pct: np.ndarray,
    predicted_raw_minus_pct: np.ndarray,
    predicted_raw_plus_pct: np.ndarray,
    predicted_mode_minus_pct: np.ndarray,
    predicted_mode_plus_pct: np.ndarray,
    censored_minus_mask: np.ndarray,
    censored_plus_mask: np.ndarray,
    confident_mask: np.ndarray,
    delta_limit: float,
    predictor_mode: str,
    show_raw_overlay: bool,
) -> None:
    path = Path(output_figure)
    path.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=14)
    fig, ax_scatter = plt.subplots(1, 1, figsize=(7.2, 6.2))

    lim_pct = 100.0 * delta_limit
    obs_minus_clip = np.minimum(np.asarray(observed_minus_pct, dtype=float), lim_pct)
    obs_plus_clip = np.minimum(np.asarray(observed_plus_pct, dtype=float), lim_pct)
    pred_mode_minus_clip = np.minimum(
        np.asarray(predicted_mode_minus_pct, dtype=float),
        lim_pct,
    )
    pred_mode_plus_clip = np.minimum(
        np.asarray(predicted_mode_plus_pct, dtype=float),
        lim_pct,
    )
    pred_raw_minus_clip = np.minimum(
        np.asarray(predicted_raw_minus_pct, dtype=float),
        lim_pct,
    )
    pred_raw_plus_clip = np.minimum(
        np.asarray(predicted_raw_plus_pct, dtype=float),
        lim_pct,
    )

    x = np.concatenate((-obs_minus_clip, obs_plus_clip))
    y = np.concatenate((-pred_mode_minus_clip, pred_mode_plus_clip))
    y_raw = np.concatenate((-pred_raw_minus_clip, pred_raw_plus_clip))
    censored = np.concatenate(
        (
            np.asarray(censored_minus_mask, dtype=bool),
            np.asarray(censored_plus_mask, dtype=bool),
        )
    )
    confident_edge = np.asarray(confident_mask, dtype=bool)
    confident = np.concatenate((confident_edge, confident_edge))
    uncensored = ~censored

    if np.any(uncensored):
        ax_scatter.scatter(
            x[uncensored],
            y[uncensored],
            s=42,
            c="#1b9e77",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=f"{predictor_mode} predictor",
        )
    if np.any(uncensored & confident):
        ax_scatter.scatter(
            x[uncensored & confident],
            y[uncensored & confident],
            s=58,
            facecolors="none",
            edgecolors="#1f78b4",
            linewidth=1.1,
            alpha=0.9,
            label="High-confidence subset",
        )
    if np.any(censored):
        ax_scatter.scatter(
            x[censored],
            y[censored],
            s=38,
            facecolors="none",
            edgecolors="0.45",
            linewidth=0.9,
            alpha=0.85,
            label="Right-censored (stable up to scan limit)",
        )
    if show_raw_overlay and predictor_mode != "raw":
        ax_scatter.scatter(
            x[uncensored] if np.any(uncensored) else x,
            y_raw[uncensored] if np.any(uncensored) else y_raw,
            s=36,
            c="#d95f02",
            alpha=0.6,
            marker="x",
            label="Raw linear predictor (uncensored only)",
        )
    ax_scatter.plot(
        [-lim_pct, lim_pct],
        [-lim_pct, lim_pct],
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    ax_scatter.set_xlim(-lim_pct, lim_pct)
    ax_scatter.set_ylim(-lim_pct, lim_pct)
    ax_scatter.set_xlabel(r"Observed directional breakpoint [\%]")
    ax_scatter.set_ylabel(
        rf"Predicted directional breakpoint [{predictor_mode}] [\%]"
    )
    ax_scatter.set_title("Directional first-breakpoint prediction vs scan")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.legend(loc="upper left")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict active-set breakpoint radii from reduced KKT sensitivities.",
    )
    parser.add_argument(
        "--network", choices=["synthetic", "braess"], default="synthetic"
    )
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-commodities", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)

    parser.add_argument(
        "--edge-selection", choices=["top-flow", "random", "all"], default="top-flow"
    )
    parser.add_argument("--max-edges", type=int, default=30)

    parser.add_argument("--delta-min", type=float, default=-0.05)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument(
        "--observed-method", choices=["bisection", "grid"], default="bisection"
    )
    parser.add_argument("--bisect-iters", type=int, default=14)
    parser.add_argument(
        "--active-set-level", choices=["commodity", "edge"], default="commodity"
    )
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--disagg-reg", type=float, default=1e-6)
    parser.add_argument("--confidence-gap-min", type=float, default=2.0)
    parser.add_argument("--mismatch-tol-pct", type=float, default=0.5)
    parser.add_argument("--predictor-mode", choices=["raw", "v3"], default="v3")
    parser.add_argument(
        "--show-raw-overlay",
        action="store_true",
        help="Overlay raw linear predictor points in panel a.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small, fast configuration for quick smoke tests.",
    )
    parser.add_argument("--v3-refine-iters", type=int, default=6)
    parser.add_argument("--v3-expand-factor", type=float, default=1.8)
    parser.add_argument("--v3-max-expand-iters", type=int, default=6)
    parser.add_argument("--verify-sensitivity-edges", type=int, default=0)
    parser.add_argument("--verify-step-rel", type=float, default=1e-4)
    parser.add_argument("--min-beta", type=float, default=1e-8)
    parser.add_argument("--solver", choices=["osqp", "mosek"], default="osqp")
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)

    args, unknown = parser.parse_known_args()

    remaining = []
    skip_next = False
    for token in unknown:
        if skip_next:
            skip_next = False
            continue
        if token in ("-f", "--f"):
            skip_next = True
            continue
        if token.startswith("--f="):
            continue
        remaining.append(token)

    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if args.smoke:
        args.network = "synthetic"
        args.num_nodes = min(args.num_nodes, 20)
        args.num_commodities = min(args.num_commodities, 3)
        args.max_edges = min(args.max_edges, 8)
        args.num_trials = 1
        args.seed_step = 1
        args.bisect_iters = min(args.bisect_iters, 6)
        args.num_points = min(args.num_points, 11)
        args.verify_sensitivity_edges = min(args.verify_sensitivity_edges, 2)
    if args.num_points < 3:
        parser.error("--num-points must be at least 3.")
    if args.max_edges < 1:
        parser.error("--max-edges must be at least 1.")
    if args.delta_min > args.delta_max:
        parser.error("--delta-min must be <= --delta-max.")
    if args.num_nodes < 5:
        parser.error("--num-nodes must be at least 5.")
    if args.num_commodities < 1:
        parser.error("--num-commodities must be at least 1.")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1.")
    if args.seed_step < 1:
        parser.error("--seed-step must be at least 1.")
    if args.bisect_iters < 1:
        parser.error("--bisect-iters must be at least 1.")
    if args.verify_sensitivity_edges < 0:
        parser.error("--verify-sensitivity-edges must be >= 0.")
    if args.verify_step_rel <= 0:
        parser.error("--verify-step-rel must be > 0.")
    if args.disagg_reg < 0:
        parser.error("--disagg-reg must be >= 0.")
    if args.confidence_gap_min <= 0:
        parser.error("--confidence-gap-min must be > 0.")
    if args.mismatch_tol_pct < 0:
        parser.error("--mismatch-tol-pct must be >= 0.")
    if args.v3_refine_iters < 1:
        parser.error("--v3-refine-iters must be at least 1.")
    if args.v3_expand_factor <= 1.0:
        parser.error("--v3-expand-factor must be > 1.")
    if args.v3_max_expand_iters < 1:
        parser.error("--v3-max-expand-iters must be at least 1.")
    return args


def main() -> None:
    args = _parse_args()
    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK
    delta_values = np.linspace(args.delta_min, args.delta_max, args.num_points)
    delta_limit = max(abs(args.delta_min), abs(args.delta_max))

    if args.network == "braess" and args.num_trials > 1:
        print("Warning: braess network is deterministic; forcing --num-trials=1.")
        args.num_trials = 1
    if args.smoke:
        print(
            "Smoke mode active: "
            f"nodes={args.num_nodes}, commodities={args.num_commodities}, "
            f"edges/test={args.max_edges}, trials={args.num_trials}"
        )

    rows: list[dict] = []
    observed_pct_all: list[float] = []
    predicted_raw_pct_all: list[float] = []
    predicted_raw_clipped_pct_all: list[float] = []
    predicted_v3_pct_all: list[float] = []
    predicted_v3_clipped_pct_all: list[float] = []
    predicted_used_pct_all: list[float] = []
    predicted_used_clipped_pct_all: list[float] = []
    observed_minus_pct_all: list[float] = []
    observed_plus_pct_all: list[float] = []
    observed_minus_censored_all: list[bool] = []
    observed_plus_censored_all: list[bool] = []
    predicted_raw_minus_pct_all: list[float] = []
    predicted_raw_plus_pct_all: list[float] = []
    predicted_used_minus_pct_all: list[float] = []
    predicted_used_plus_pct_all: list[float] = []
    confident_all: list[bool] = []
    total_edge_tests = 0
    total_v3_solver_evals = 0
    verify_rows: list[dict] = []

    for trial_idx in range(args.num_trials):
        trial_seed = args.seed + trial_idx * args.seed_step
        trial_args = argparse.Namespace(**vars(args))
        trial_args.seed = trial_seed

        graph, demands = _build_problem(trial_args)
        baseline = _build_baseline_system(
            graph=graph,
            demands=demands,
            solver=solver,
            eps_active=args.eps_active,
            regularization=args.regularization,
            disagg_reg=args.disagg_reg,
        )

        baseline_edge_flow = np.sum(baseline.flow_matrix, axis=0)
        selected_indices = _select_edge_indices(
            baseline_edge_flow=baseline_edge_flow,
            strategy=args.edge_selection,
            max_edges=args.max_edges,
            seed=trial_seed,
        )
        total_edge_tests += int(selected_indices.size)
        baseline_indicator = _active_set_indicator(
            baseline.flow_matrix,
            eps_active=args.eps_active,
            level=args.active_set_level,
        )

        pred_raw_minus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_raw_plus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_raw_sym_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_v3_minus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_v3_plus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_v3_sym_rel = np.full(selected_indices.size, np.inf, dtype=float)
        pred_gap_sym = np.full(selected_indices.size, np.inf, dtype=float)
        pred_n_plus = np.zeros(selected_indices.size, dtype=int)
        pred_n_minus = np.zeros(selected_indices.size, dtype=int)
        observed_minus_rel = np.full(selected_indices.size, delta_limit, dtype=float)
        observed_plus_rel = np.full(selected_indices.size, delta_limit, dtype=float)
        observed_radii_rel = np.full(selected_indices.size, delta_limit, dtype=float)
        cens_minus_obs = [False] * selected_indices.size
        cens_plus_obs = [False] * selected_indices.size
        cens_minus_v3 = [False] * selected_indices.size
        cens_plus_v3 = [False] * selected_indices.size
        v3_solver_evals = np.zeros(selected_indices.size, dtype=int)
        event_minus: list[str] = []
        event_plus: list[str] = []

        for local_idx, edge_idx in enumerate(selected_indices):
            beta_k = float(baseline.beta[edge_idx])
            if beta_k <= args.min_beta:
                event_minus.append("undefined")
                event_plus.append("undefined")
                continue

            eta_minus, eta_plus, evt_minus, evt_plus, pred_meta = _first_breakpoints(
                baseline=baseline,
                edge_idx=int(edge_idx),
                tol=args.tol,
                active_set_level=args.active_set_level,
                return_meta=True,
            )
            pred_raw_minus_rel[local_idx] = eta_minus / beta_k
            pred_raw_plus_rel[local_idx] = eta_plus / beta_k
            pred_raw_sym_rel[local_idx] = min(
                pred_raw_minus_rel[local_idx],
                pred_raw_plus_rel[local_idx],
            )
            pred_gap_sym[local_idx] = float(pred_meta["gap_sym"])
            pred_n_plus[local_idx] = int(pred_meta["n_plus_candidates"])
            pred_n_minus[local_idx] = int(pred_meta["n_minus_candidates"])

            (
                pred_v3_minus_rel[local_idx],
                pred_v3_plus_rel[local_idx],
                cens_minus_v3[local_idx],
                cens_plus_v3[local_idx],
                v3_solver_evals[local_idx],
            ) = _predicted_breakpoint_v3(
                baseline=baseline,
                baseline_indicator=baseline_indicator,
                edge_idx=int(edge_idx),
                predicted_minus_rel=pred_raw_minus_rel[local_idx],
                predicted_plus_rel=pred_raw_plus_rel[local_idx],
                delta_limit=delta_limit,
                solver=solver,
                min_beta=args.min_beta,
                active_set_level=args.active_set_level,
                disagg_reg=args.disagg_reg,
                refine_iters=args.v3_refine_iters,
                expand_factor=args.v3_expand_factor,
                max_expand_iters=args.v3_max_expand_iters,
            )
            pred_v3_sym_rel[local_idx] = min(
                pred_v3_minus_rel[local_idx],
                pred_v3_plus_rel[local_idx],
            )

            if args.observed_method == "bisection":
                obs_minus, obs_plus, c_minus, c_plus = _observed_breakpoint_bisection(
                    baseline=baseline,
                    baseline_indicator=baseline_indicator,
                    edge_idx=int(edge_idx),
                    delta_limit=delta_limit,
                    bisect_iters=args.bisect_iters,
                    solver=solver,
                    min_beta=args.min_beta,
                    active_set_level=args.active_set_level,
                    disagg_reg=args.disagg_reg,
                )
            else:
                obs_minus, obs_plus, c_minus, c_plus = _observed_breakpoint_grid(
                    baseline=baseline,
                    baseline_indicator=baseline_indicator,
                    edge_idx=int(edge_idx),
                    delta_values=delta_values,
                    solver=solver,
                    min_beta=args.min_beta,
                    active_set_level=args.active_set_level,
                    disagg_reg=args.disagg_reg,
                )
            observed_minus_rel[local_idx] = obs_minus
            observed_plus_rel[local_idx] = obs_plus
            observed_radii_rel[local_idx] = min(obs_minus, obs_plus)
            cens_minus_obs[local_idx] = c_minus
            cens_plus_obs[local_idx] = c_plus
            event_minus.append(evt_minus)
            event_plus.append(evt_plus)

        if args.verify_sensitivity_edges > 0 and selected_indices.size > 0:
            verify_indices = selected_indices[
                : min(args.verify_sensitivity_edges, selected_indices.size)
            ]
            for edge_idx in verify_indices:
                check = _verify_directional_sensitivity(
                    baseline=baseline,
                    edge_idx=int(edge_idx),
                    solver=solver,
                    min_beta=args.min_beta,
                    step_rel=args.verify_step_rel,
                    active_set_level=args.active_set_level,
                    disagg_reg=args.disagg_reg,
                )
                if check is not None:
                    check["trial"] = trial_idx
                    check["seed"] = trial_seed
                    verify_rows.append(check)

        pred_raw_sym_clipped_rel = np.minimum(pred_raw_sym_rel, delta_limit)
        pred_v3_sym_clipped_rel = np.minimum(pred_v3_sym_rel, delta_limit)
        if args.predictor_mode == "v3":
            pred_used_minus_rel = pred_v3_minus_rel
            pred_used_plus_rel = pred_v3_plus_rel
            pred_used_sym_rel = pred_v3_sym_rel
        else:
            pred_used_minus_rel = pred_raw_minus_rel
            pred_used_plus_rel = pred_raw_plus_rel
            pred_used_sym_rel = pred_raw_sym_rel
        pred_used_minus_clipped_rel = np.minimum(pred_used_minus_rel, delta_limit)
        pred_used_plus_clipped_rel = np.minimum(pred_used_plus_rel, delta_limit)
        pred_used_sym_clipped_rel = np.minimum(pred_used_sym_rel, delta_limit)

        for local_idx, edge_idx in enumerate(selected_indices):
            edge_label = _format_edge(baseline.edges[edge_idx])
            obs_pct = 100.0 * float(observed_radii_rel[local_idx])
            obs_minus_pct = 100.0 * float(observed_minus_rel[local_idx])
            obs_plus_pct = 100.0 * float(observed_plus_rel[local_idx])
            obs_minus_censored = bool(cens_minus_obs[local_idx])
            obs_plus_censored = bool(cens_plus_obs[local_idx])

            pred_raw_minus_pct = 100.0 * float(pred_raw_minus_rel[local_idx])
            pred_raw_plus_pct = 100.0 * float(pred_raw_plus_rel[local_idx])
            pred_raw_pct = 100.0 * float(pred_raw_sym_rel[local_idx])
            pred_raw_clip_pct = 100.0 * float(pred_raw_sym_clipped_rel[local_idx])
            pred_v3_minus_pct = 100.0 * float(pred_v3_minus_rel[local_idx])
            pred_v3_plus_pct = 100.0 * float(pred_v3_plus_rel[local_idx])
            pred_v3_pct = 100.0 * float(pred_v3_sym_rel[local_idx])
            pred_v3_clip_pct = 100.0 * float(pred_v3_sym_clipped_rel[local_idx])
            pred_used_minus_pct = 100.0 * float(pred_used_minus_rel[local_idx])
            pred_used_plus_pct = 100.0 * float(pred_used_plus_rel[local_idx])
            pred_used_pct = 100.0 * float(pred_used_sym_rel[local_idx])
            pred_used_clip_pct = 100.0 * float(pred_used_sym_clipped_rel[local_idx])

            obs_is_censored = bool(obs_minus_censored and obs_plus_censored)
            is_confident = bool(
                np.isfinite(pred_gap_sym[local_idx])
                and pred_gap_sym[local_idx] >= args.confidence_gap_min
            )
            is_mismatch_raw = bool(abs(pred_raw_clip_pct - obs_pct) > args.mismatch_tol_pct)
            is_mismatch_v3 = bool(abs(pred_v3_clip_pct - obs_pct) > args.mismatch_tol_pct)
            is_mismatch_used = bool(
                abs(pred_used_clip_pct - obs_pct) > args.mismatch_tol_pct
            )
            pred_winner_event = (
                event_minus[local_idx]
                if pred_raw_minus_rel[local_idx] <= pred_raw_plus_rel[local_idx]
                else event_plus[local_idx]
            )

            rows.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed,
                    "edge_index": int(edge_idx),
                    "edge": edge_label,
                    "beta": float(baseline.beta[edge_idx]),
                    "baseline_edge_flow": float(baseline_edge_flow[edge_idx]),
                    "pred_raw_radius_minus_pct": pred_raw_minus_pct,
                    "pred_raw_radius_plus_pct": pred_raw_plus_pct,
                    "pred_raw_radius_sym_pct": pred_raw_pct,
                    "pred_raw_radius_sym_clipped_pct": pred_raw_clip_pct,
                    "pred_v3_radius_minus_pct": pred_v3_minus_pct,
                    "pred_v3_radius_plus_pct": pred_v3_plus_pct,
                    "pred_v3_radius_sym_pct": pred_v3_pct,
                    "pred_v3_radius_sym_clipped_pct": pred_v3_clip_pct,
                    "pred_used_radius_minus_pct": pred_used_minus_pct,
                    "pred_used_radius_plus_pct": pred_used_plus_pct,
                    "pred_used_radius_sym_pct": pred_used_pct,
                    "pred_used_radius_sym_clipped_pct": pred_used_clip_pct,
                    "predictor_mode": args.predictor_mode,
                    "pred_gap_sym": float(pred_gap_sym[local_idx]),
                    "pred_n_plus_candidates": int(pred_n_plus[local_idx]),
                    "pred_n_minus_candidates": int(pred_n_minus[local_idx]),
                    "pred_winner_event": pred_winner_event,
                    "obs_radius_minus_pct": obs_minus_pct,
                    "obs_radius_plus_pct": obs_plus_pct,
                    "obs_minus_censored": obs_minus_censored,
                    "obs_plus_censored": obs_plus_censored,
                    "obs_radius_pct": obs_pct,
                    "obs_is_censored": obs_is_censored,
                    "is_confident": is_confident,
                    "is_mismatch_raw": is_mismatch_raw,
                    "is_mismatch_v3": is_mismatch_v3,
                    "is_mismatch": is_mismatch_used,
                    "first_event_minus": event_minus[local_idx],
                    "first_event_plus": event_plus[local_idx],
                    "v3_solver_evals": int(v3_solver_evals[local_idx]),
                }
            )
            observed_pct_all.append(obs_pct)
            predicted_raw_pct_all.append(pred_raw_pct)
            predicted_raw_clipped_pct_all.append(pred_raw_clip_pct)
            predicted_v3_pct_all.append(pred_v3_pct)
            predicted_v3_clipped_pct_all.append(pred_v3_clip_pct)
            predicted_used_pct_all.append(pred_used_pct)
            predicted_used_clipped_pct_all.append(pred_used_clip_pct)
            observed_minus_pct_all.append(obs_minus_pct)
            observed_plus_pct_all.append(obs_plus_pct)
            observed_minus_censored_all.append(obs_minus_censored)
            observed_plus_censored_all.append(obs_plus_censored)
            predicted_raw_minus_pct_all.append(pred_raw_minus_pct)
            predicted_raw_plus_pct_all.append(pred_raw_plus_pct)
            predicted_used_minus_pct_all.append(pred_used_minus_pct)
            predicted_used_plus_pct_all.append(pred_used_plus_pct)
            confident_all.append(is_confident)
            total_v3_solver_evals += int(v3_solver_evals[local_idx])

        print(
            f"Trial {trial_idx + 1}/{args.num_trials} (seed={trial_seed}): "
            f"tested {selected_indices.size} edges"
        )

    observed_pct_arr = np.asarray(observed_pct_all, dtype=float)
    pred_raw_pct_arr = np.asarray(predicted_raw_pct_all, dtype=float)
    pred_raw_clipped_pct_arr = np.asarray(predicted_raw_clipped_pct_all, dtype=float)
    pred_v3_pct_arr = np.asarray(predicted_v3_pct_all, dtype=float)
    pred_v3_clipped_pct_arr = np.asarray(predicted_v3_clipped_pct_all, dtype=float)
    pred_used_pct_arr = np.asarray(predicted_used_pct_all, dtype=float)
    pred_used_clipped_pct_arr = np.asarray(predicted_used_clipped_pct_all, dtype=float)
    observed_minus_pct_arr = np.asarray(observed_minus_pct_all, dtype=float)
    observed_plus_pct_arr = np.asarray(observed_plus_pct_all, dtype=float)
    observed_minus_censored_arr = np.asarray(observed_minus_censored_all, dtype=bool)
    observed_plus_censored_arr = np.asarray(observed_plus_censored_all, dtype=bool)
    pred_raw_minus_pct_arr = np.asarray(predicted_raw_minus_pct_all, dtype=float)
    pred_raw_plus_pct_arr = np.asarray(predicted_raw_plus_pct_all, dtype=float)
    pred_used_minus_pct_arr = np.asarray(predicted_used_minus_pct_all, dtype=float)
    pred_used_plus_pct_arr = np.asarray(predicted_used_plus_pct_all, dtype=float)
    confident_arr = np.asarray(confident_all, dtype=bool)

    _write_table(
        output_path=args.output_table,
        rows=rows,
    )
    _plot_results(
        output_figure=args.output_figure,
        observed_minus_pct=observed_minus_pct_arr,
        observed_plus_pct=observed_plus_pct_arr,
        predicted_raw_minus_pct=pred_raw_minus_pct_arr,
        predicted_raw_plus_pct=pred_raw_plus_pct_arr,
        predicted_mode_minus_pct=pred_used_minus_pct_arr,
        predicted_mode_plus_pct=pred_used_plus_pct_arr,
        censored_minus_mask=observed_minus_censored_arr,
        censored_plus_mask=observed_plus_censored_arr,
        confident_mask=confident_arr,
        delta_limit=delta_limit,
        predictor_mode=args.predictor_mode,
        show_raw_overlay=args.show_raw_overlay,
    )

    uncensored = np.array(
        [not bool(row["obs_is_censored"]) for row in rows], dtype=bool
    )
    confident = np.array([bool(row["is_confident"]) for row in rows], dtype=bool)
    confident_uncensored = confident & uncensored
    winner_events = np.array([row["pred_winner_event"] for row in rows], dtype=object)

    def _metric_block(pred_clipped: np.ndarray) -> dict:
        finite = np.isfinite(pred_clipped)
        if np.any(finite):
            mae = float(np.mean(np.abs(pred_clipped[finite] - observed_pct_arr[finite])))
        else:
            mae = float("nan")
        if np.count_nonzero(finite) >= 2:
            corr = float(np.corrcoef(pred_clipped[finite], observed_pct_arr[finite])[0, 1])
        else:
            corr = float("nan")

        unc = uncensored & finite
        if np.any(unc):
            mae_unc = float(np.mean(np.abs(pred_clipped[unc] - observed_pct_arr[unc])))
        else:
            mae_unc = float("nan")
        if np.count_nonzero(unc) >= 2:
            corr_unc = float(np.corrcoef(pred_clipped[unc], observed_pct_arr[unc])[0, 1])
        else:
            corr_unc = float("nan")
        mismatch = np.abs(pred_clipped - observed_pct_arr) > args.mismatch_tol_pct
        mismatch_rate = float(np.mean(mismatch[finite])) if np.any(finite) else float("nan")
        mismatch_rate_unc = (
            float(np.mean(mismatch[unc])) if np.any(unc) else float("nan")
        )
        return {
            "mae": mae,
            "corr": corr,
            "mae_unc": mae_unc,
            "corr_unc": corr_unc,
            "mismatch_rate": mismatch_rate,
            "mismatch_rate_unc": mismatch_rate_unc,
            "mismatch_mask": mismatch,
            "finite_mask": finite,
            "unc_mask": unc,
        }

    raw_metrics = _metric_block(pred_raw_clipped_pct_arr)
    v3_metrics = _metric_block(pred_v3_clipped_pct_arr)
    used_metrics = _metric_block(pred_used_clipped_pct_arr)

    print(f"Network: {args.network}")
    print(f"Active-set level: {args.active_set_level}")
    print(f"Predictor mode: {args.predictor_mode}")
    print(f"Observed method: {args.observed_method}")
    print(f"Disaggregation regularization: {args.disagg_reg:g}")
    print(
        "v3 seeded refine params: "
        f"iters={args.v3_refine_iters}, "
        f"expand_factor={args.v3_expand_factor:.2f}, "
        f"max_expand={args.v3_max_expand_iters}"
    )
    print(f"Trials: {args.num_trials}")
    print(f"Total tested edges: {total_edge_tests}")
    print(f"Total v3 stability checks: {total_v3_solver_evals}")
    censored_share = float(np.mean(~uncensored)) if len(rows) else float("nan")
    print(f"Censored share (stable up to scan limit): {100.0 * censored_share:.1f}%")
    print(
        f"Raw linear: MAE={raw_metrics['mae']:.3f} pp, "
        f"corr={raw_metrics['corr']:.3f}, "
        f"mismatch={100.0 * raw_metrics['mismatch_rate']:.1f}%"
    )
    print(
        f"Raw linear (uncensored): MAE={raw_metrics['mae_unc']:.3f} pp, "
        f"corr={raw_metrics['corr_unc']:.3f}, "
        f"mismatch={100.0 * raw_metrics['mismatch_rate_unc']:.1f}%"
    )
    print(
        f"v3 refined: MAE={v3_metrics['mae']:.3f} pp, "
        f"corr={v3_metrics['corr']:.3f}, "
        f"mismatch={100.0 * v3_metrics['mismatch_rate']:.1f}%"
    )
    print(
        f"v3 refined (uncensored): MAE={v3_metrics['mae_unc']:.3f} pp, "
        f"corr={v3_metrics['corr_unc']:.3f}, "
        f"mismatch={100.0 * v3_metrics['mismatch_rate_unc']:.1f}%"
    )
    print(
        f"Selected '{args.predictor_mode}' predictor: MAE={used_metrics['mae']:.3f} pp, "
        f"corr={used_metrics['corr']:.3f}, "
        f"mismatch={100.0 * used_metrics['mismatch_rate']:.1f}% "
        f"(threshold {args.mismatch_tol_pct:.2f} pp)"
    )

    print("Raw linear mismatch by winning event (uncensored only):")
    for event_name in ("leave", "enter", "leave/enter", "none", "undefined"):
        mask_evt = uncensored & (winner_events == event_name)
        if not np.any(mask_evt):
            continue
        evt_mismatch = np.mean(raw_metrics["mismatch_mask"][mask_evt])
        evt_mae = np.mean(np.abs(pred_raw_clipped_pct_arr[mask_evt] - observed_pct_arr[mask_evt]))
        print(
            f"  {event_name:11s} n={int(np.count_nonzero(mask_evt)):4d} "
            f"mismatch={100.0 * evt_mismatch:5.1f}% "
            f"MAE={evt_mae:5.3f} pp"
        )

    coverage_conf = (
        float(np.mean(confident_uncensored))
        if confident_uncensored.size
        else float("nan")
    )
    if np.any(confident_uncensored):
        mae_conf_raw = float(
            np.mean(
                np.abs(
                    pred_raw_clipped_pct_arr[confident_uncensored]
                    - observed_pct_arr[confident_uncensored]
                )
            )
        )
        corr_conf_raw = (
            float(
                np.corrcoef(
                    pred_raw_clipped_pct_arr[confident_uncensored],
                    observed_pct_arr[confident_uncensored],
                )[0, 1]
            )
            if np.count_nonzero(confident_uncensored) >= 2
            else float("nan")
        )
        mismatch_rate_conf_raw = float(
            np.mean(raw_metrics["mismatch_mask"][confident_uncensored])
        )
    else:
        mae_conf_raw = float("nan")
        corr_conf_raw = float("nan")
        mismatch_rate_conf_raw = float("nan")

    print(
        f"High-confidence subset (gap >= {args.confidence_gap_min:.2f}) "
        f"coverage: {100.0 * coverage_conf:.1f}%"
    )
    print(f"High-confidence raw MAE (uncensored): {mae_conf_raw:.3f} percentage points")
    print(f"High-confidence raw corr (uncensored): {corr_conf_raw:.3f}")
    print(
        f"High-confidence raw mismatch rate (uncensored): "
        f"{100.0 * mismatch_rate_conf_raw:.1f}%"
    )
    if verify_rows:
        flow_err = np.array([row["flow_rel_error"] for row in verify_rows], dtype=float)
        slack_err = np.array(
            [row["slack_rel_error"] for row in verify_rows], dtype=float
        )
        flow_abs = np.array([row["flow_abs_error"] for row in verify_rows], dtype=float)
        slack_abs = np.array(
            [row["slack_abs_error"] for row in verify_rows], dtype=float
        )
        flow_scale = np.array([row["flow_scale"] for row in verify_rows], dtype=float)
        slack_scale = np.array([row["slack_scale"] for row in verify_rows], dtype=float)
        flow_support = np.array(
            [row["flow_support_size"] for row in verify_rows], dtype=float
        )
        slack_support = np.array(
            [row["slack_support_size"] for row in verify_rows], dtype=float
        )
        good_flow = flow_scale > 1e-2
        good_slack = slack_scale > 1e-2

        flow_rel_str = (
            f"{np.median(flow_err[good_flow]):.2e}"
            if np.any(good_flow)
            else "n/a (tiny flow sensitivity scale)"
        )
        slack_rel_str = (
            f"{np.median(slack_err[good_slack]):.2e}"
            if np.any(good_slack)
            else "n/a (tiny slack sensitivity scale)"
        )
        print(
            "Sensitivity check (finite-difference, local-regime only): "
            f"n={len(verify_rows)}, "
            f"median flow rel. err={flow_rel_str}, "
            f"median slack rel. err={slack_rel_str}, "
            f"median flow abs. err={np.median(flow_abs):.2e}, "
            f"median slack abs. err={np.median(slack_abs):.2e}, "
            f"median support sizes: flow={np.median(flow_support):.0f}, "
            f"slack={np.median(slack_support):.0f}"
        )
    elif args.verify_sensitivity_edges > 0:
        print(
            "Sensitivity check: no usable local-regime samples "
            "(active set changed for +/- finite-difference step)."
        )
    print(f"Saved figure to {args.output_figure}")
    print(f"Saved table to {args.output_table}")


if __name__ == "__main__":
    main()
