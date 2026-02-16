"""Reusable toy graphs used in publication figure scripts."""

from __future__ import annotations

from typing import Dict, Hashable, Tuple

import networkx as nx
import numpy as np

Node = Hashable
Edge = Tuple[Node, Node]


def build_classic_braess_social_cost_graph() -> nx.DiGraph:
    """Braess graph used for social-cost-vs-edge-parameter scans."""
    graph = nx.DiGraph()
    graph.add_edge("A", "B", alpha=0.0, beta=5.0)
    graph.add_edge("A", "C", alpha=0.1, beta=0.0)
    graph.add_edge("C", "B", alpha=0.0, beta=1.0)
    graph.add_edge("B", "D", alpha=0.1, beta=0.0)
    graph.add_edge("C", "D", alpha=0.0, beta=5.0)

    positions = {
        "A": (0.0, 0.5),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (2.0, 0.5),
    }
    nx.set_node_attributes(graph, positions, "pos")
    return graph


def build_classic_braess_validation_graph() -> nx.DiGraph:
    """Braess graph used for SCGC-vs-numerical-derivative validation."""
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "B")]
    alpha = np.array([0.1, 1.0, 1.0, 0.1, 0.1])
    beta = np.array([5.0, 1.0, 1.0, 5.0, 1.0])

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    nx.set_edge_attributes(graph, dict(zip(edges, alpha)), "alpha")
    nx.set_edge_attributes(graph, dict(zip(edges, beta)), "beta")

    positions = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.5, 1.0), "D": (1.5, 1.0)}
    nx.set_node_attributes(graph, positions, "pos")
    return graph


def build_social_optimum_demo_graph() -> nx.DiGraph:
    """Graph used to compare user equilibrium and social optimum."""
    edges = [("A", "B"), ("A", "C"), ("C", "D"), ("B", "D"), ("B", "C")]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    nx.set_edge_attributes(graph, {edge: 1.0 for edge in edges}, "alpha")
    nx.set_edge_attributes(graph, {edge: 0.0 for edge in edges}, "beta")

    positions = {"A": (0.0, 0.5), "D": (1.0, 0.5), "B": (0.5, 0.0), "C": (0.5, 1.0)}
    nx.set_node_attributes(graph, positions, "pos")
    return graph


def build_multicommodity_demo_graph() -> nx.DiGraph:
    """Graph used to illustrate commodity-specific and aggregated flows."""
    edges = [
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("C", "D"),
        ("D", "C"),
        ("B", "D"),
        ("D", "B"),
        ("B", "C"),
    ]
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    nx.set_edge_attributes(graph, {edge: 1.0 for edge in edges}, "alpha")
    nx.set_edge_attributes(graph, {edge: 0.0 for edge in edges}, "beta")

    positions = {"A": (0.0, 0.5), "D": (1.0, 0.5), "B": (0.5, 0.0), "C": (0.5, 1.0)}
    nx.set_node_attributes(graph, positions, "pos")
    return graph


def linear_travel_time_label(edge: Edge, alpha: float, beta: float) -> str:
    """Return a TeX label for a linear travel-time function on an edge."""
    edge_name = "".join(map(str, edge))
    expression = ""

    if alpha != 0:
        if alpha == 1:
            expression += f"f_{{{edge_name}}}"
        elif alpha == -1:
            expression += f"-f_{{{edge_name}}}"
        else:
            expression += f"{alpha:g}f_{{{edge_name}}}"

    if beta != 0:
        sign = " + " if beta > 0 and expression else ""
        if beta < 0:
            sign = " - " if expression else "-"
        expression += f"{sign}{abs(beta):g}"

    if not expression:
        expression = "0"

    return rf"$t_{{{edge_name}}}(f_{{{edge_name}}})={expression}$"


def default_node_colors() -> Dict[str, str]:
    """Color scheme for classic four-node Braess graphs."""
    return {"A": "red", "B": "grey", "C": "grey", "D": "lightblue"}
