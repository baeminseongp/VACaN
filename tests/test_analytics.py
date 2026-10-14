import csv
import tempfile
import unittest
from pathlib import Path

from vacan.analytics.congestion import aggregate_congestion
from vacan.analytics.coverage import build_observed_congestion
from vacan.analytics.zones import filter_zones_by_trajectory_support, generate_grid_zones
from vacan.datasets.atc.map import load_ros_map
from vacan.datasets.atc.trajectory import normalize_trajectories


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Path(__file__).parent / "fixtures" / "atc"
        self.ros_map = load_ros_map(self.fixture / "localization_grid.yaml")
        self.zones = generate_grid_zones(
            self.ros_map, zone_size_m=2.0, minimum_walkable_fraction=0.2
        )

    def test_grid_zones_use_walkable_area(self) -> None:
        self.assertTrue(self.zones.zones)
        self.assertTrue(all(zone.walkable_area_m2 > 0 for zone in self.zones.zones))
        self.assertIsNotNone(self.zones.zone_for_point(0, 0))

    def test_density_uses_mean_simultaneous_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized = root / "trajectories.csv"
            ground_truth = root / "ground_truth.csv"
            normalize_trajectories(self.fixture / "atc-sample.csv", normalized)
            report = aggregate_congestion(
                normalized,
                ground_truth,
                self.zones,
                time_bin_seconds=60,
                low_upper_exclusive=0.5,
                medium_upper_exclusive=1.5,
            )
            self.assertEqual(report["input_rows"], 3)
            self.assertEqual(report["frame_count"], 3)
            self.assertEqual(report["time_bin_count"], 1)

    def test_empirical_thresholds_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized = root / "trajectories.csv"
            normalize_trajectories(self.fixture / "atc-sample.csv", normalized)
            report = aggregate_congestion(
                normalized, root / "ground_truth.csv", self.zones,
                time_bin_seconds=60, low_upper_exclusive=0.5,
                medium_upper_exclusive=1.5, empirical_positive_quantiles=(0.5, 0.85),
            )
            self.assertEqual(report["threshold_mode"], "empirical_positive_quantiles")
            self.assertLessEqual(
                float(report["low_upper_exclusive"]),
                float(report["medium_upper_exclusive"]),
            )

    def test_trajectory_support_filters_unvisited_zones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            normalized = Path(temporary) / "trajectories.csv"
            normalize_trajectories(self.fixture / "atc-sample.csv", normalized)
            supported, report = filter_zones_by_trajectory_support(
                normalized, self.zones, minimum_rows_per_zone=1
            )
            self.assertGreater(len(supported.zones), 0)
            self.assertLessEqual(len(supported.zones), len(self.zones.zones))
            self.assertEqual(report["supported_zone_count"], len(supported.zones))

    def test_observation_mask_preserves_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized = root / "trajectories.csv"
            ground_truth = root / "ground_truth.csv"
            observed = root / "observed.csv"
            normalize_trajectories(self.fixture / "atc-sample.csv", normalized)
            aggregate_congestion(
                normalized, ground_truth, self.zones, time_bin_seconds=60,
                low_upper_exclusive=0.5, medium_upper_exclusive=1.5,
            )
            build_observed_congestion(
                ground_truth, observed, root / "summary.csv", root / "observations.csv",
                self.zones.zones, fixed_area_fraction=0.0, selection_key="test",
                robot_zones_per_bin=0, stale_after_seconds=30,
            )
            with observed.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row["observation_state"] == "UNKNOWN" for row in rows))
            self.assertTrue(all(row["density_per_m2"] == "" for row in rows))


if __name__ == "__main__":
    unittest.main()
