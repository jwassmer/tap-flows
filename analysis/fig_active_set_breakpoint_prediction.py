"""Predict first active-set breakpoints from reduced KKT linearization.

For each selected edge parameter ``beta_k``, this script computes:
1) A predicted local perturbation radius before the active set changes.
2) An observed radius from brute-force re-solves over a delta grid.

The prediction uses first-order sensitivities on the fixed baseline active set.
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
from src.figure_style import add_panel_label, apply_publication_style
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
    potential_drop_flat = np.asarray(incidence_full.T @ lambda_flat, dtype=float).reshape(-1)
    return alpha_flat * total_flat + beta_flat - potential_drop_flat


def _local_stability_radius(delta_values: np.ndarray, stable_flags: np.ndarray) -> float:
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
        [float(pop if pop is not None and pop > 0 else 1.0) for pop in population_values],
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
        demand = -np.ones(graph.number_of_nodes()) * load / (graph.number_of_nodes() - 1)
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


def _solve_flows(
    graph: nx.DiGraph,
    demands: np.ndarray,
    beta: np.ndarray,
    solver,
) -> np.ndarray:
    flow_matrix, _ = mc.solve_multicommodity_tap(
        graph,
        demands,
        beta=beta,
        pos_flows=True,
        return_fw=True,
        solver=solver,
    )
    return np.asarray(flow_matrix, dtype=float)


def _build_baseline_system(
    graph: nx.DiGraph,
    demands: np.ndarray,
    solver,
    eps_active: float,
    regularization: float,
):
    flow_matrix, lambda_matrix = mc.solve_multicommodity_tap(
        graph,
        demands,
        pos_flows=True,
        return_fw=True,
        solver=solver,
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
    potential_drop_flat = np.asarray(incidence_full.T @ lambda_flat, dtype=float).reshape(-1)
    slack_flat = alpha_flat * total_flow[edge_ids_flat] + beta_flat - potential_drop_flat

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
    rhs = np.zeros(baseline.n_active + int(np.count_nonzero(baseline.node_mask)), dtype=float)

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

        eta_plus = float(min(eta_plus_candidates)) if eta_plus_candidates else np.inf
        eta_minus = float(min(eta_minus_candidates)) if eta_minus_candidates else np.inf

        if np.isfinite(eta_plus):
            first_active_leave_plus = np.any(
                np.isclose(
                    (flow_threshold - f0_active[df_active < -tol]) / df_active[df_active < -tol],
                    eta_plus,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )
            event_plus = "leave" if first_active_leave_plus else "enter"
        if np.isfinite(eta_minus):
            first_active_leave_minus = np.any(
                np.isclose(
                    -((flow_threshold - f0_active[df_active > tol]) / df_active[df_active > tol]),
                    eta_minus,
                    rtol=1e-4,
                    atol=1e-8,
                )
            )
            event_minus = "leave" if first_active_leave_minus else "enter"
        return eta_minus, eta_plus, event_minus, event_plus

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

    eta_plus = float(min(eta_plus_candidates)) if eta_plus_candidates else np.inf
    eta_minus = float(min(eta_minus_candidates)) if eta_minus_candidates else np.inf

    if np.isfinite(eta_plus):
        event_plus = "leave/enter"
    if np.isfinite(eta_minus):
        event_minus = "leave/enter"

    return eta_minus, eta_plus, event_minus, event_plus


def _is_active_set_stable(
    baseline: BaselineSystem,
    baseline_indicator: np.ndarray,
    edge_idx: int,
    delta_abs: float,
    direction: int,
    solver,
    min_beta: float,
    active_set_level: str,
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


def _verify_directional_sensitivity(
    baseline: BaselineSystem,
    edge_idx: int,
    solver,
    min_beta: float,
    step_rel: float,
    active_set_level: str,
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

    flow_plus, lambda_plus = mc.solve_multicommodity_tap(
        baseline.graph,
        baseline.demands,
        beta=beta_plus,
        pos_flows=True,
        return_fw=True,
        solver=solver,
    )
    flow_minus, lambda_minus = mc.solve_multicommodity_tap(
        baseline.graph,
        baseline.demands,
        beta=beta_minus,
        pos_flows=True,
        return_fw=True,
        solver=solver,
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
        flow_num = np.linalg.norm(flow_model_active[flow_support] - flow_fd_active[flow_support])
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
    slack_scale_inactive = np.maximum(np.abs(slack_model_inactive), np.abs(slack_fd_inactive))
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
                "pred_radius_minus_pct",
                "pred_radius_plus_pct",
                "pred_radius_sym_pct",
                "pred_radius_sym_clipped_pct",
                "obs_radius_pct",
                "obs_is_censored",
                "abs_error_clipped_pct",
                "first_event_minus",
                "first_event_plus",
            ]
        )
        for row in rows:
            obs = row["obs_radius_pct"]
            pred_clip = row["pred_radius_sym_clipped_pct"]
            abs_err = abs(pred_clip - obs)
            writer.writerow(
                [
                    int(row["trial"]),
                    int(row["seed"]),
                    int(row["edge_index"]),
                    row["edge"],
                    float(row["beta"]),
                    float(row["baseline_edge_flow"]),
                    float(row["pred_radius_minus_pct"]),
                    float(row["pred_radius_plus_pct"]),
                    float(row["pred_radius_sym_pct"]),
                    pred_clip,
                    obs,
                    int(bool(row["obs_is_censored"])),
                    abs_err,
                    row["first_event_minus"],
                    row["first_event_plus"],
                ]
            )


def _plot_results(
    output_figure: str,
    observed_radius_pct: np.ndarray,
    predicted_sym_pct: np.ndarray,
    predicted_sym_clipped_pct: np.ndarray,
    censored_mask: np.ndarray,
    labels: list[str],
    delta_limit: float,
) -> None:
    path = Path(output_figure)
    path.parent.mkdir(parents=True, exist_ok=True)

    apply_publication_style(font_size=14)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.0),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax_scatter, ax_error = axes

    add_panel_label(ax_scatter, r"\textbf{a}", x=0.04, y=1.02, fontsize=22)
    add_panel_label(ax_error, r"\textbf{b}", x=0.04, y=1.02, fontsize=22)

    x = np.asarray(observed_radius_pct, dtype=float)
    y = np.asarray(predicted_sym_clipped_pct, dtype=float)
    y_raw = np.asarray(predicted_sym_pct, dtype=float)
    censored = np.asarray(censored_mask, dtype=bool)
    uncensored = ~censored

    lim = max(float(np.max(x)), float(np.max(y)), 100.0 * delta_limit, 1e-9)
    if np.any(uncensored):
        ax_scatter.scatter(
            x[uncensored],
            y[uncensored],
            s=42,
            c="#1b9e77",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label="Uncensored (observed change in scan range)",
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
    ax_scatter.scatter(
        x[uncensored] if np.any(uncensored) else x,
        y_raw[uncensored] if np.any(uncensored) else y_raw,
        s=36,
        c="#d95f02",
        alpha=0.6,
        marker="x",
        label="Predicted raw (uncensored only)",
    )
    ax_scatter.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.5)
    ax_scatter.set_xlim(-0.1, lim * 1.02)
    ax_scatter.set_ylim(-0.1, lim * 1.02)
    ax_scatter.set_xlabel("Observed local radius [%]")
    ax_scatter.set_ylabel("Predicted local radius [%]")
    ax_scatter.set_title("First-breakpoint prediction vs scan")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.legend(loc="upper left")

    errors = y - x
    order = np.argsort(np.abs(errors))[::-1]
    top = order[: min(12, order.size)]
    ax_error.barh(
        np.arange(top.size),
        errors[top],
        color=np.where(errors[top] >= 0, "#66a61e", "#e7298a"),
        alpha=0.9,
    )
    ax_error.axvline(0.0, color="black", linewidth=1.0)
    ax_error.set_yticks(np.arange(top.size), [labels[i] for i in top])
    ax_error.invert_yaxis()
    ax_error.set_xlabel("Prediction error (clipped) [%]")
    ax_error.set_title("Largest absolute errors")
    ax_error.grid(axis="x", alpha=0.3)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict active-set breakpoint radii from reduced KKT sensitivities.",
    )
    parser.add_argument("--network", choices=["synthetic", "braess"], default="synthetic")
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--num-commodities", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)

    parser.add_argument("--edge-selection", choices=["top-flow", "random", "all"], default="top-flow")
    parser.add_argument("--max-edges", type=int, default=30)

    parser.add_argument("--delta-min", type=float, default=-0.05)
    parser.add_argument("--delta-max", type=float, default=0.05)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--observed-method", choices=["bisection", "grid"], default="bisection")
    parser.add_argument("--bisect-iters", type=int, default=14)
    parser.add_argument("--active-set-level", choices=["commodity", "edge"], default="commodity")
    parser.add_argument("--eps-active", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--regularization", type=float, default=1e-5)
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
    return args


def main() -> None:
    args = _parse_args()
    solver = cp.OSQP if args.solver == "osqp" else cp.MOSEK
    delta_values = np.linspace(args.delta_min, args.delta_max, args.num_points)
    delta_limit = max(abs(args.delta_min), abs(args.delta_max))

    if args.network == "braess" and args.num_trials > 1:
        print("Warning: braess network is deterministic; forcing --num-trials=1.")
        args.num_trials = 1

    rows: list[dict] = []
    observed_pct_all: list[float] = []
    predicted_pct_all: list[float] = []
    predicted_clipped_pct_all: list[float] = []
    labels_all: list[str] = []
    total_edge_tests = 0
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

        predicted_minus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        predicted_plus_rel = np.full(selected_indices.size, np.inf, dtype=float)
        predicted_sym_rel = np.full(selected_indices.size, np.inf, dtype=float)
        observed_radii_rel = np.full(selected_indices.size, delta_limit, dtype=float)
        cens_minus = [False] * selected_indices.size
        cens_plus = [False] * selected_indices.size
        event_minus: list[str] = []
        event_plus: list[str] = []

        for local_idx, edge_idx in enumerate(selected_indices):
            beta_k = float(baseline.beta[edge_idx])
            if beta_k <= args.min_beta:
                event_minus.append("undefined")
                event_plus.append("undefined")
                continue

            eta_minus, eta_plus, evt_minus, evt_plus = _first_breakpoints(
                baseline=baseline,
                edge_idx=int(edge_idx),
                tol=args.tol,
                active_set_level=args.active_set_level,
            )
            predicted_minus_rel[local_idx] = eta_minus / beta_k
            predicted_plus_rel[local_idx] = eta_plus / beta_k
            predicted_sym_rel[local_idx] = min(
                predicted_minus_rel[local_idx],
                predicted_plus_rel[local_idx],
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
                )
            observed_radii_rel[local_idx] = min(obs_minus, obs_plus)
            cens_minus[local_idx] = c_minus
            cens_plus[local_idx] = c_plus
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
                )
                if check is not None:
                    check["trial"] = trial_idx
                    check["seed"] = trial_seed
                    verify_rows.append(check)

        predicted_sym_clipped_rel = np.minimum(predicted_sym_rel, delta_limit)

        for local_idx, edge_idx in enumerate(selected_indices):
            edge_label = _format_edge(baseline.edges[edge_idx])
            obs_pct = 100.0 * float(observed_radii_rel[local_idx])
            pred_pct = 100.0 * float(predicted_sym_rel[local_idx])
            pred_clip_pct = 100.0 * float(predicted_sym_clipped_rel[local_idx])
            rows.append(
                {
                    "trial": trial_idx,
                    "seed": trial_seed,
                    "edge_index": int(edge_idx),
                    "edge": edge_label,
                    "beta": float(baseline.beta[edge_idx]),
                    "baseline_edge_flow": float(baseline_edge_flow[edge_idx]),
                    "pred_radius_minus_pct": 100.0 * float(predicted_minus_rel[local_idx]),
                    "pred_radius_plus_pct": 100.0 * float(predicted_plus_rel[local_idx]),
                    "pred_radius_sym_pct": pred_pct,
                    "pred_radius_sym_clipped_pct": pred_clip_pct,
                    "obs_radius_pct": obs_pct,
                    "obs_is_censored": bool(cens_minus[local_idx] and cens_plus[local_idx]),
                    "first_event_minus": event_minus[local_idx],
                    "first_event_plus": event_plus[local_idx],
                }
            )
            observed_pct_all.append(obs_pct)
            predicted_pct_all.append(pred_pct)
            predicted_clipped_pct_all.append(pred_clip_pct)
            labels_all.append(f"t{trial_idx}:{edge_label}")

        print(
            f"Trial {trial_idx + 1}/{args.num_trials} (seed={trial_seed}): "
            f"tested {selected_indices.size} edges"
        )

    observed_pct_arr = np.asarray(observed_pct_all, dtype=float)
    predicted_pct_arr = np.asarray(predicted_pct_all, dtype=float)
    predicted_clipped_pct_arr = np.asarray(predicted_clipped_pct_all, dtype=float)

    _write_table(
        output_path=args.output_table,
        rows=rows,
    )
    _plot_results(
        output_figure=args.output_figure,
        observed_radius_pct=observed_pct_arr,
        predicted_sym_pct=predicted_pct_arr,
        predicted_sym_clipped_pct=predicted_clipped_pct_arr,
        censored_mask=np.array(
            [bool(row["obs_is_censored"]) for row in rows],
            dtype=bool,
        ),
        labels=labels_all,
        delta_limit=delta_limit,
    )

    finite = np.isfinite(predicted_clipped_pct_arr)
    if np.any(finite):
        mae = np.mean(
            np.abs(
                predicted_clipped_pct_arr[finite]
                - observed_pct_arr[finite]
            )
        )
    else:
        mae = float("nan")

    if np.count_nonzero(finite) >= 2:
        corr = float(
            np.corrcoef(
                predicted_clipped_pct_arr[finite],
                observed_pct_arr[finite],
            )[0, 1]
        )
    else:
        corr = float("nan")
    print(f"Network: {args.network}")
    print(f"Active-set level: {args.active_set_level}")
    print(f"Observed method: {args.observed_method}")
    print(f"Trials: {args.num_trials}")
    print(f"Total tested edges: {total_edge_tests}")
    print(f"MAE(predicted_clipped vs observed): {mae:.3f} percentage points")
    print(f"Correlation(predicted_clipped, observed): {corr:.3f}")

    uncensored = np.array([not bool(row["obs_is_censored"]) for row in rows], dtype=bool)
    if np.any(uncensored):
        mae_unc = np.mean(
            np.abs(
                predicted_clipped_pct_arr[uncensored] - observed_pct_arr[uncensored]
            )
        )
    else:
        mae_unc = float("nan")
    if np.count_nonzero(uncensored) >= 2:
        corr_unc = float(
            np.corrcoef(
                predicted_clipped_pct_arr[uncensored],
                observed_pct_arr[uncensored],
            )[0, 1]
        )
    else:
        corr_unc = float("nan")
    censored_share = float(np.mean(~uncensored)) if len(rows) else float("nan")
    print(f"Censored share (stable up to scan limit): {100.0 * censored_share:.1f}%")
    print(f"MAE uncensored only: {mae_unc:.3f} percentage points")
    print(f"Correlation uncensored only: {corr_unc:.3f}")
    if verify_rows:
        flow_err = np.array([row["flow_rel_error"] for row in verify_rows], dtype=float)
        slack_err = np.array([row["slack_rel_error"] for row in verify_rows], dtype=float)
        flow_abs = np.array([row["flow_abs_error"] for row in verify_rows], dtype=float)
        slack_abs = np.array([row["slack_abs_error"] for row in verify_rows], dtype=float)
        flow_scale = np.array([row["flow_scale"] for row in verify_rows], dtype=float)
        slack_scale = np.array([row["slack_scale"] for row in verify_rows], dtype=float)
        flow_support = np.array([row["flow_support_size"] for row in verify_rows], dtype=float)
        slack_support = np.array([row["slack_support_size"] for row in verify_rows], dtype=float)
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
