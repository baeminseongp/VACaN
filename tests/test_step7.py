import csv
import json
import tempfile
import unittest
from pathlib import Path

from vacan.datasets.jrdb import evaluate_jrdb
from vacan.datasets.mall import evaluate_mall


ROOT = Path(__file__).resolve().parents[1]


class Step7EvaluationTests(unittest.TestCase):
    def test_jrdb_fixture_produces_mobile_sensing_metrics(self):
        source = json.loads((ROOT / "configs/evaluation/step7-jrdb-fixture.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source["output_directory"] = directory
            config = Path(directory) / "config.json"
            source["sequences"][0]["annotations"] = str(ROOT / "tests/fixtures/jrdb/stationary_labels.json")
            source["sequences"][1]["annotations"] = str(ROOT / "tests/fixtures/jrdb/dynamic_labels.json")
            config.write_text(json.dumps(source))
            report = evaluate_jrdb(config)
            self.assertEqual(report["evaluation_family"], "mobile_robot_sensing")
            self.assertEqual(report["frame_count"], 6)
            with (Path(directory) / "summary_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            motions = {row["sensor_motion"] for row in rows}
            self.assertEqual(motions, {"ALL", "DYNAMIC", "STATIONARY"})

    def test_mall_fixture_produces_independent_vlm_metrics(self):
        source = json.loads((ROOT / "configs/evaluation/step7-mall-fixture.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source["output_directory"] = directory
            source["ground_truth"] = str(ROOT / "tests/fixtures/mall/head_points.csv")
            source["vlm_predictions"] = str(ROOT / "tests/fixtures/mall/vlm_predictions.csv")
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(source))
            report = evaluate_mall(config)
            self.assertEqual(report["evaluation_family"], "dense_crowd_vlm")
            self.assertEqual(report["frame_count"], 4)
            self.assertAlmostEqual(report["metrics"]["count_mae"], 0.75)
            self.assertAlmostEqual(report["metrics"]["congestion_level_accuracy"], 1.0)

    def test_mall_rejects_missing_prediction_frames(self):
        source = json.loads((ROOT / "configs/evaluation/step7-mall-fixture.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            source["output_directory"] = str(Path(directory) / "output")
            source["ground_truth"] = str(ROOT / "tests/fixtures/mall/head_points.csv")
            predictions = Path(directory) / "predictions.csv"
            predictions.write_text("frame_id,predicted_count,predicted_level,description,grounded_evidence\n1,3,LOW,a,b\n")
            source["vlm_predictions"] = str(predictions)
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "frame mismatch"):
                evaluate_mall(config)
