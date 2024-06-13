# %%
import unittest
from src import GraphGenerator as gg
from src import Plotting as pl
from src import LinAlg as la
import networkx as nx
import numpy as np


def generate_graph(
    source_node, target_node, total_load, seed=42, num_nodes=10, num_edges=15
):
    U = gg.random_graph(
        source_node, target_node, total_load, seed, num_nodes, num_edges
    )
    G = gg.to_directed_flow_graph(U)
    return U, G


def calculate_matrices(G, U):
    Pvec = np.array(list(nx.get_node_attributes(G, "P").values()))
    F = nx.get_edge_attributes(G, "flow")
    weights = list(nx.get_edge_attributes(U, "weight").values())
    K = np.diag(weights)
    L = nx.laplacian_matrix(U, weight="weight").toarray()
    I = -nx.incidence_matrix(U, oriented=True, weight="weight").toarray()
    psi = np.linalg.pinv(L) @ Pvec
    return Pvec, F, K, L, I, psi


def validate_edge_incidence(psi, I, F):
    return any(I.T @ psi - list(F.values()) <= 1e-5)


def validate_path_link_incidence(path_link_matrix, pathflow, F):
    return any(path_link_matrix @ list(pathflow.values()) - list(F.values()) < 1e-7)


def validate_od_path_incidence(od_path_matrix, pathflow, total_load):
    return sum(od_path_matrix @ list(pathflow.values())) - total_load < 1e-7


class TestGraphFlow(unittest.TestCase):
    def setUp(self):
        self.source_node = ["a", "b"]
        self.target_node = ["c", "j"]
        self.total_load = 1000
        self.U, self.G = generate_graph(
            self.source_node, self.target_node, self.total_load
        )
        self.path_link_matrix = la.path_link_incidence_matrix(self.G)
        self.od_path_matrix = la.od_path_incidence_matrix(self.G)
        self.cycle_link_matrix = la.cycle_link_incidence_matrix(self.G)

        self.Pvec, self.F, self.K, self.L, self.I, self.psi = calculate_matrices(
            self.G, self.U
        )
        self.pathflow = la.path_flows(self.G)

    def test_edge_incidence(self):
        self.assertTrue(
            validate_edge_incidence(self.psi, self.I, self.F),
            "Edge incidence validation failed",
        )

    def test_path_link_incidence(self):
        self.assertTrue(
            validate_path_link_incidence(self.path_link_matrix, self.pathflow, self.F),
            "Path link incidence validation failed",
        )

    def test_od_path_incidence(self):
        self.assertTrue(
            validate_od_path_incidence(
                self.od_path_matrix, self.pathflow, self.total_load
            ),
            "OD path incidence validation faPiled",
        )


# %%

if __name__ == "__main__":
    unittest.main()

# %%
