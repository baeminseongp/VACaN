import unittest

from vacan.evaluation.metrics import bootstrap_mean_ci, sensing_time_metrics


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_brackets_mean(self):
        values = {"t1": -1.0, "t2": -2.0, "t3": -3.0, "t4": -4.0}
        first = bootstrap_mean_ci(values, samples=1000, confidence=0.95, seed=7)
        second = bootstrap_mean_ci(values, samples=1000, confidence=0.95, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(-2.5, first[0])
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_unknown_is_excluded_not_converted_to_zero(self):
        truth = {
            "A": {"density_per_m2": "2", "congestion_level": "HIGH"},
            "B": {"density_per_m2": "1", "congestion_level": "LOW"},
        }
        observed = {
            "A": {"density_per_m2": "", "congestion_level": "UNKNOWN", "observation_state": "UNKNOWN"},
            "B": {"density_per_m2": "1", "congestion_level": "LOW", "observation_state": "FIXED_COVERED"},
        }
        result = sensing_time_metrics(truth, observed, {"A": 3.0, "B": 1.0}, {"FIXED_COVERED"})
        self.assertEqual(0.25, result["coverage_fraction"])
        self.assertEqual(0.0, result["density_mae"])
        self.assertEqual(0.0, result["hotspot_recall"])

    def test_robot_observation_increases_hotspot_recall(self):
        truth = {"A": {"density_per_m2": "2", "congestion_level": "HIGH"}}
        observed = {"A": {"density_per_m2": "2", "congestion_level": "HIGH", "observation_state": "ROBOT_RECOVERED"}}
        fixed = sensing_time_metrics(truth, observed, {"A": 1.0}, {"FIXED_COVERED"})
        collaborative = sensing_time_metrics(truth, observed, {"A": 1.0}, {"FIXED_COVERED", "ROBOT_RECOVERED"})
        self.assertEqual(0.0, fixed["hotspot_recall"])
        self.assertEqual(1.0, collaborative["hotspot_recall"])


if __name__ == "__main__":
    unittest.main()
