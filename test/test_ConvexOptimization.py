# %%
import numpy as np
import networkx as nx
from src import Equilibirium as eq
from src import ConvexOptimization as cv
from src import GraphGenerator as gg
from src import LinAlg as la

import unittest
import numpy as np
from src import ConvexOptimization as cv


class TestConvexOptimization(unittest.TestCase):
    def test_cv(self):
        # Create a test graph
        source, target = ["a", "b", "e", "k", "g"], ["j", "c", "d", "f"]
        total_load = 1000
        # Example graph creation
        U = gg.random_graph(
            source, target, total_load, seed=42, num_nodes=20, num_edges=30
        )
        G = gg.to_directed_flow_graph(U)

        # Call the function under test
        cv_lin_flow = cv.convex_optimization_linflow(G)
        cv_tap_flow = cv.convex_optimization_TAP(G)

        # Perform assertions
        la_linflow = la_linflow = list(eq.linear_flow(G).values())
        self.assertTrue(
            all(np.abs(la_linflow - cv_lin_flow) < 1e-5),
            "Convex optimization of KCL yields similar results than LA",
        )

        self.assertTrue(
            all(np.abs(cv_tap_flow - cv_lin_flow) < 1e-5),
            "Convex optimization of KCL yields similar results than TAP",
        )


if __name__ == "__main__":
    unittest.main()
