import csv
import json
import tempfile
import unittest
from pathlib import Path

from vacan.datasets.atc.alignment import alignment_report
from vacan.datasets.atc.map import load_ros_map, occupancy_state, ros_map_to_geojson, write_map_outputs
from vacan.datasets.atc.trajectory import normalize_trajectories


class AtcMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Path(__file__).parent / "fixtures" / "atc"

    def test_ros_map_metadata_and_row_inversion(self) -> None:
        ros_map = load_ros_map(self.fixture / "localization_grid.yaml")
        self.assertEqual((ros_map.width, ros_map.height), (4, 3))
        self.assertEqual(occupancy_state(0, ros_map), "OCCUPIED")
        self.assertEqual(occupancy_state(254, ros_map), "FREE")
        self.assertEqual(occupancy_state(205, ros_map), "UNKNOWN")
        geojson = ros_map_to_geojson(ros_map)
        first_ring = geojson["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(first_ring[0], [-1.0, 1.0])

    def test_map_outputs_are_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_map_outputs(self.fixture / "localization_grid.yaml", output)
            self.assertEqual(json.loads((output / "map.json").read_text())["resolution_m"], 1.0)
            self.assertEqual(json.loads((output / "occupancy_runs.geojson").read_text())["type"], "FeatureCollection")


class AtcTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Path(__file__).parent / "fixtures" / "atc"

    def test_units_timestamp_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "trajectories.csv"
            count = normalize_trajectories(self.fixture / "atc-sample.csv", output)
            self.assertEqual(count, 3)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["x_m"], "0.000000")
            self.assertEqual(rows[0]["z_m"], "1.700000")
            self.assertEqual(rows[0]["timestamp"], "2012-11-14T00:00:00.125000Z")
            report = alignment_report(load_ros_map(self.fixture / "localization_grid.yaml"), output)
            self.assertEqual(report["inside_map_fraction"], 1.0)
            self.assertEqual(report["status"], "PASS")

    def test_bad_column_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.csv"
            source.write_text("1,2,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 8 columns"):
                normalize_trajectories(source, Path(temporary) / "output.csv")


if __name__ == "__main__":
    unittest.main()
