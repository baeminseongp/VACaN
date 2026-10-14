import unittest

from vacan.planning.graph import GraphNode, ZoneGraph
from vacan.planning.planner import PlanningWeights, shortest_path


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        nodes = {
            name: GraphNode(name, row, column, x, y)
            for name, row, column, x, y in (
                ("A", 0, 0, 0, 0), ("B", 0, 1, 1, 0),
                ("C", 1, 0, 0, 1), ("D", 1, 1, 1, 1),
            )
        }
        self.graph = ZoneGraph(nodes, {
            "A": (("B", 1.0), ("C", 1.0)),
            "B": (("A", 1.0), ("D", 1.0)),
            "C": (("A", 1.0), ("D", 1.0)),
            "D": (("B", 1.0), ("C", 1.0)),
        })
        self.weights = PlanningWeights(2.0, 1.0, 4.0, {
            "FIXED_COVERED": 0.0, "ROBOT_RECOVERED": 0.1,
            "STALE": 0.5, "UNKNOWN": 1.0,
        })

    def test_baseline_takes_deterministic_shortest_route(self) -> None:
        path, cost = shortest_path(self.graph, "A", "D")
        self.assertEqual(path, ["A", "B", "D"])
        self.assertEqual(cost, 2.0)

    def test_congestion_aware_route_avoids_dense_node(self) -> None:
        observations = {
            zone: {"density_per_m2": "0", "observation_state": "FIXED_COVERED"}
            for zone in self.graph.nodes
        }
        observations["B"]["density_per_m2"] = "2"
        path, _ = shortest_path(self.graph, "A", "D", observations, self.weights)
        self.assertEqual(path, ["A", "C", "D"])


if __name__ == "__main__":
    unittest.main()
