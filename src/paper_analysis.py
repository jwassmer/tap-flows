"""Shared analysis workflows for publication figure generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cvxpy as cp
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc
from src import SocialCost as sc

PRIMARY_ROAD_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified"]'
)


@dataclass
class CityCaseResult:
    """Container for solved city experiments."""

    graph: nx.MultiDiGraph
    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    od_matrix: np.ndarray
    commodity_flows: np.ndarray
    lagrange_multipliers: np.ndarray


@dataclass
class InterventionBenchmarkResult:
    """Container with intervention-validation outputs."""

    interventions: pd.DataFrame
    summary: pd.DataFrame
    edge_effects: pd.DataFrame
    rank_correlation: pd.DataFrame



def _required_column(values: gpd.GeoDataFrame, column: str) -> np.ndarray:
    if column not in values:
        raise ValueError(f"Missing required column '{column}'.")
    return values[column].to_numpy(dtype=float)



def destination_population_demands(
    nodes: gpd.GeoDataFrame,
    destination_nodes: gpd.GeoDataFrame,
    gamma: float = 0.03,
    demand_column: str = "population",
) -> np.ndarray:
    """Create one commodity per destination weighted by destination population."""
    populations = _required_column(nodes, demand_column)
    destination_pop = _required_column(destination_nodes, demand_column)

    if np.any(populations < 0):
        raise ValueError(f"Column '{demand_column}' contains negative values.")

    total_destination_pop = destination_pop.sum()
    if total_destination_pop <= 0:
        raise ValueError("Destination population sum must be positive.")

    destination_weights = destination_pop / total_destination_pop
    node_index = nodes.index.to_numpy()
    node_position = {node_id: i for i, node_id in enumerate(node_index)}

    od_matrix = np.zeros((len(destination_nodes), len(nodes)), dtype=float)
    base_demand = gamma * populations

    for commodity_idx, destination_id in enumerate(destination_nodes.index):
        demand = base_demand * destination_weights[commodity_idx]
        destination_pos = node_position[destination_id]
        demand[destination_pos] -= demand.sum()
        od_matrix[commodity_idx] = demand

    return od_matrix



def stadium_vehicle_demands(
    nodes: gpd.GeoDataFrame,
    sink_node_ids: Sequence[int],
    total_vehicles: float,
    demand_column: str = "population",
) -> np.ndarray:
    """Distribute total vehicles from all origins to each sink node separately."""
    sink_node_ids = np.unique(np.asarray(sink_node_ids, dtype=int))
    if sink_node_ids.size == 0:
        raise ValueError("At least one sink node is required.")

    populations = _required_column(nodes, demand_column)
    total_population = populations.sum()
    if total_population <= 0:
        raise ValueError("Population sum must be positive.")

    origin_inflow = (total_vehicles * populations / total_population) / sink_node_ids.size

    node_index = nodes.index.to_numpy()
    node_position = {node_id: i for i, node_id in enumerate(node_index)}

    demands = np.zeros((sink_node_ids.size, nodes.shape[0]), dtype=float)
    for commodity_idx, sink_node_id in enumerate(sink_node_ids):
        if sink_node_id not in node_position:
            raise ValueError(f"Sink node id {sink_node_id} not found in node table.")

        demand = origin_inflow.copy()
        sink_pos = node_position[int(sink_node_id)]
        demand[sink_pos] = 0.0
        demand[sink_pos] = -np.sum(demand)
        demands[commodity_idx] = demand

    return demands



def nearest_graph_nodes(graph: nx.MultiDiGraph, addresses: Iterable[str]) -> np.ndarray:
    """Geocode addresses and map them to nearest graph node ids."""
    node_ids = []
    for address in addresses:
        latitude, longitude = ox.geocode(address)
        node_id = ox.distance.nearest_nodes(graph, longitude, latitude)
        node_ids.append(node_id)
    return np.unique(np.asarray(node_ids, dtype=int))



def solve_city_multicommodity_case(
    graph: nx.MultiDiGraph,
    od_matrix: np.ndarray,
    eps: float = 1e-3,
    demands_to_sinks: bool = False,
    **solver_kwargs,
) -> CityCaseResult:
    """Solve multicommodity TAP and annotate edges with flow and SCGC."""
    solver_kwargs = dict(solver_kwargs)
    solver_kwargs.setdefault("solver", cp.OSQP)
    solver_kwargs.setdefault("pos_flows", True)
    solver_kwargs.setdefault("return_fw", True)

    commodity_flows, lagrange_multipliers = mc.solve_multicommodity_tap(
        graph,
        od_matrix,
        **solver_kwargs,
    )
    aggregate_flow = np.sum(commodity_flows, axis=0)

    dsc = mcsc.derivative_social_cost(
        graph,
        commodity_flows,
        od_matrix,
        eps=eps,
        demands_to_sinks=demands_to_sinks,
    )

    nodes, edges = ox.graph_to_gdfs(graph)
    edges = edges.copy()

    edge_order = list(graph.edges(keys=True))
    if isinstance(edges.index, pd.MultiIndex) and edges.index.nlevels >= 3:
        edges = edges.reindex(edge_order)

    edges["edge_index"] = np.arange(len(edge_order), dtype=int)
    edges["flow"] = aggregate_flow
    edges["derivative_social_cost"] = list(dsc.values())

    return CityCaseResult(
        graph=graph,
        nodes=nodes,
        edges=edges,
        od_matrix=np.asarray(od_matrix, dtype=float),
        commodity_flows=np.asarray(commodity_flows, dtype=float),
        lagrange_multipliers=np.asarray(lagrange_multipliers, dtype=float),
    )



def add_utilization_columns(
    edges: gpd.GeoDataFrame,
    flow_column: str = "flow",
    alpha_column: str = "alpha",
    beta_column: str = "beta",
    utilization_column: str = "utilization",
) -> gpd.GeoDataFrame:
    """Annotate edge table with loaded travel time and utilization."""
    edges = edges.copy()
    edges["loaded_beta"] = edges[alpha_column] * edges[flow_column] + edges[beta_column]
    edges[utilization_column] = 1 - edges[beta_column] / edges["loaded_beta"]
    return edges



def braess_edge_subset(
    edges: gpd.GeoDataFrame,
    gradient_column: str = "derivative_social_cost",
) -> gpd.GeoDataFrame:
    """Return the subset of edges with negative SCGC values."""
    return edges[edges[gradient_column] < 0]


def summarize_edge_metrics(
    edges: gpd.GeoDataFrame,
    gradient_column: str = "derivative_social_cost",
    flow_column: str = "flow",
    length_column: str = "length",
    utilization_column: str | None = None,
    high_utilization_threshold: float = 0.8,
    low_utilization_threshold: float = 0.2,
) -> dict[str, float]:
    """Return compact descriptive metrics used in manuscript result sections."""
    if gradient_column not in edges or flow_column not in edges:
        raise ValueError(
            f"Expected columns '{gradient_column}' and '{flow_column}' in edge table."
        )

    gradient = edges[gradient_column].to_numpy(dtype=float)
    flow = edges[flow_column].to_numpy(dtype=float)
    n_edges = int(len(edges))
    if n_edges == 0:
        raise ValueError("Edge table is empty.")

    braess_mask = gradient < 0.0
    metrics = {
        "num_edges": float(n_edges),
        "num_braess_edges": float(np.count_nonzero(braess_mask)),
        "braess_share_pct": 100.0 * float(np.mean(braess_mask)),
        "scgc_min": float(np.min(gradient)),
        "scgc_max": float(np.max(gradient)),
        "scgc_mean": float(np.mean(gradient)),
        "scgc_median": float(np.median(gradient)),
        "flow_mean": float(np.mean(flow)),
        "flow_median": float(np.median(flow)),
        "flow_max": float(np.max(flow)),
    }

    if length_column in edges:
        length_m = edges[length_column].to_numpy(dtype=float)
        total_length_km = float(np.sum(length_m) / 1000.0)
        braess_length_km = float(np.sum(length_m[braess_mask]) / 1000.0)
        metrics["total_length_km"] = total_length_km
        metrics["braess_length_km"] = braess_length_km
        metrics["braess_length_share_pct"] = (
            100.0 * braess_length_km / total_length_km if total_length_km > 0 else np.nan
        )

    if utilization_column is not None and utilization_column in edges:
        util = edges[utilization_column].to_numpy(dtype=float)
        metrics["utilization_mean"] = float(np.mean(util))
        metrics["utilization_median"] = float(np.median(util))
        metrics["utilization_max"] = float(np.max(util))
        high = util >= high_utilization_threshold
        low = util <= low_utilization_threshold
        metrics["num_edges_high_utilization"] = float(np.count_nonzero(high))
        metrics["num_edges_low_utilization"] = float(np.count_nonzero(low))
        metrics["high_utilization_share_pct"] = 100.0 * float(np.mean(high))
        metrics["low_utilization_share_pct"] = 100.0 * float(np.mean(low))
        if length_column in edges:
            length_m = edges[length_column].to_numpy(dtype=float)
            metrics["high_utilization_length_km"] = float(np.sum(length_m[high]) / 1000.0)
            metrics["low_utilization_length_km"] = float(np.sum(length_m[low]) / 1000.0)

    return metrics


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _select_candidate_indices(
    flow: np.ndarray,
    gradient: np.ndarray,
    mode: str,
    max_candidates: int,
    random_seed: int,
) -> np.ndarray:
    n = int(flow.size)
    all_indices = np.arange(n, dtype=int)
    if mode == "all" or max_candidates >= n:
        return all_indices

    if mode == "top-flow":
        order = np.argsort(-np.asarray(flow, dtype=float))
        return np.sort(order[:max_candidates])

    if mode == "top-abs-scgc":
        order = np.argsort(-np.abs(np.asarray(gradient, dtype=float)))
        return np.sort(order[:max_candidates])

    if mode == "random":
        rng = np.random.default_rng(random_seed)
        return np.sort(rng.choice(all_indices, size=max_candidates, replace=False))

    raise ValueError(f"Unknown candidate mode '{mode}'.")


def _score_to_order(
    values: np.ndarray,
    candidate_indices: np.ndarray,
    target: str,
) -> np.ndarray:
    candidates = np.asarray(candidate_indices, dtype=int)
    if target == "increase":
        order = np.argsort(-values[candidates], kind="mergesort")
        return candidates[order]
    if target == "decrease":
        order = np.argsort(values[candidates], kind="mergesort")
        return candidates[order]
    raise ValueError(f"Unknown target '{target}'.")


def run_intervention_benchmark(
    graph: nx.MultiDiGraph,
    od_matrix: np.ndarray,
    edges: gpd.GeoDataFrame,
    rel_step: float = 0.01,
    direction: int = +1,
    target: str = "increase",
    max_budget: int = 5,
    max_candidates: int = 60,
    candidate_mode: str = "top-flow",
    random_repeats: int = 16,
    random_seed: int = 42,
    min_beta: float = 1e-8,
    flow_column: str = "flow",
    gradient_column: str = "derivative_social_cost",
    solver=cp.OSQP,
    **solver_kwargs,
) -> InterventionBenchmarkResult:
    """Benchmark SCGC-based top-k interventions against simple baselines.

    Returns four tables:
    - `interventions`: realized multi-edge intervention outcomes per method and budget.
    - `summary`: compact aggregate metrics across methods and budgets.
    - `edge_effects`: single-edge predicted vs observed effects on social cost.
    - `rank_correlation`: Spearman rank agreement between each heuristic score and
      observed single-edge realized effects.
    """
    if rel_step <= 0 or rel_step >= 1:
        raise ValueError("rel_step must be in (0, 1).")
    if direction not in (-1, +1):
        raise ValueError("direction must be -1 or +1.")
    if max_budget < 1:
        raise ValueError("max_budget must be at least 1.")
    if random_repeats < 1:
        raise ValueError("random_repeats must be at least 1.")

    if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
        edge_list = list(graph.edges(keys=True))
    else:
        edge_list = list(graph.edges)
    n_edges = len(edge_list)
    if n_edges == 0:
        raise ValueError("Graph has no edges.")

    alpha = np.array([float(graph.edges[e]["alpha"]) for e in edge_list], dtype=float)
    beta = np.array([float(graph.edges[e]["beta"]) for e in edge_list], dtype=float)
    flow = edges[flow_column].to_numpy(dtype=float)
    gradient = edges[gradient_column].to_numpy(dtype=float)
    if flow.size != n_edges or gradient.size != n_edges:
        raise ValueError(
            "Edge table does not align with graph edge order (length mismatch)."
        )

    base_sc = sc.total_social_cost(graph, flow, alpha=alpha, beta=beta)
    base_scale = max(abs(base_sc), 1e-12)
    betweenness_map = nx.edge_betweenness_centrality(graph, normalized=True)
    betweenness = np.array([float(betweenness_map.get(e, 0.0)) for e in edge_list])

    candidate_indices = _select_candidate_indices(
        flow=flow,
        gradient=gradient,
        mode=candidate_mode,
        max_candidates=max_candidates,
        random_seed=random_seed,
    )
    if candidate_indices.size == 0:
        raise ValueError("No candidate edges selected.")

    delta_beta = direction * rel_step * beta
    predicted_abs = gradient * delta_beta
    predicted_pct = 100.0 * predicted_abs / base_scale

    solver_kwargs = dict(solver_kwargs)
    solver_kwargs.setdefault("solver", solver)
    solver_kwargs.setdefault("pos_flows", True)

    cache: dict[tuple[int, ...], float] = {}

    def simulate(edge_ids: Sequence[int]) -> float:
        key = tuple(sorted(set(int(i) for i in edge_ids)))
        if key in cache:
            return cache[key]
        beta_trial = beta.copy()
        if key:
            idx = np.asarray(key, dtype=int)
            beta_trial[idx] = np.maximum(min_beta, beta[idx] + delta_beta[idx])
        flow_trial = mc.solve_multicommodity_tap(
            graph,
            od_matrix,
            alpha=alpha,
            beta=beta_trial,
            **solver_kwargs,
        )
        sc_trial = sc.total_social_cost(graph, flow_trial, alpha=alpha, beta=beta_trial)
        delta_pct = 100.0 * float((sc_trial - base_sc) / base_scale)
        cache[key] = delta_pct
        return delta_pct

    edge_rows = []
    observed_single = np.zeros(candidate_indices.size, dtype=float)
    for pos, edge_idx in enumerate(candidate_indices):
        obs_pct = simulate([int(edge_idx)])
        observed_single[pos] = obs_pct
        edge = edge_list[int(edge_idx)]
        edge_rows.append(
            {
                "edge_index": int(edge_idx),
                "edge": str(edge),
                "flow": float(flow[edge_idx]),
                "scgc": float(gradient[edge_idx]),
                "betweenness": float(betweenness[edge_idx]),
                "predicted_delta_sc_pct": float(predicted_pct[edge_idx]),
                "observed_delta_sc_pct": float(obs_pct),
            }
        )

    method_scores = {
        "scgc_linear": predicted_pct,
        "flow": flow,
        "betweenness": betweenness,
    }

    corr_rows = []
    observed_for_corr = observed_single
    for method, score in method_scores.items():
        corr_rows.append(
            {
                "method": method,
                "target": target,
                "direction": int(direction),
                "n_candidates": int(candidate_indices.size),
                "spearman": _safe_spearman(score[candidate_indices], observed_for_corr),
            }
        )

    orders = {
        method: _score_to_order(score, candidate_indices, target=target)
        for method, score in method_scores.items()
    }
    max_budget_eff = min(int(max_budget), int(candidate_indices.size))
    budgets = list(range(1, max_budget_eff + 1))

    intervention_rows = []
    rng = np.random.default_rng(random_seed + 11_873)
    for budget in budgets:
        for method in ["scgc_linear", "flow", "betweenness", "random"]:
            if method == "random":
                random_deltas = []
                for _ in range(random_repeats):
                    chosen = rng.permutation(candidate_indices)[:budget]
                    random_deltas.append(simulate(chosen))
                delta_mean = float(np.mean(random_deltas))
                delta_std = float(np.std(random_deltas))
                selected_edges = ""
            else:
                chosen = orders[method][:budget]
                delta_mean = float(simulate(chosen))
                delta_std = float("nan")
                selected_edges = ";".join(str(int(idx)) for idx in chosen)

            intervention_rows.append(
                {
                    "method": method,
                    "target": target,
                    "direction": int(direction),
                    "budget": int(budget),
                    "n_candidates": int(candidate_indices.size),
                    "rel_step_pct": float(100.0 * rel_step),
                    "delta_sc_pct": delta_mean,
                    "delta_sc_std_pct": delta_std,
                    "selected_edge_indices": selected_edges,
                }
            )

    interventions_df = pd.DataFrame(intervention_rows)
    edge_effects_df = pd.DataFrame(edge_rows)
    corr_df = pd.DataFrame(corr_rows)

    if interventions_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = interventions_df.groupby("method", as_index=False).agg(
            mean_delta_sc_pct=("delta_sc_pct", "mean"),
            median_delta_sc_pct=("delta_sc_pct", "median"),
            max_delta_sc_pct=("delta_sc_pct", "max"),
            min_delta_sc_pct=("delta_sc_pct", "min"),
        )
        if target == "increase":
            winner_idx = interventions_df.groupby("budget")["delta_sc_pct"].idxmax()
        else:
            winner_idx = interventions_df.groupby("budget")["delta_sc_pct"].idxmin()
        win_counts = (
            interventions_df.loc[winner_idx, "method"]
            .value_counts()
            .rename_axis("method")
            .reset_index(name="win_count")
        )
        summary_df = summary_df.merge(win_counts, on="method", how="left")
        summary_df["win_count"] = summary_df["win_count"].fillna(0).astype(int)
        summary_df["n_budgets"] = int(interventions_df["budget"].nunique())
        summary_df["win_share_pct"] = (
            100.0 * summary_df["win_count"] / summary_df["n_budgets"]
        )

        summary_df["target"] = target
        summary_df["direction"] = int(direction)
        summary_df["n_candidates"] = int(candidate_indices.size)
        summary_df["rel_step_pct"] = float(100.0 * rel_step)

        summary_df = summary_df.merge(corr_df[["method", "spearman"]], on="method", how="left")

    return InterventionBenchmarkResult(
        interventions=interventions_df,
        summary=summary_df,
        edge_effects=edge_effects_df,
        rank_correlation=corr_df,
    )
