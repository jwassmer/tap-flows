"""Shared analysis workflows for publication figure generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cvxpy as cp
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox

from src import multiCommoditySocialCost as mcsc
from src import multiCommodityTAP as mc

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
